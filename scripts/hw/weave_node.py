#!/usr/bin/env python3
"""The hardware node: ReSpeaker XVF3800 + coin motors on a Pi Zero 2 W.

Runs ON THE PI. Captures from the array, ships audio and direction to the Mac
running `autocwi live --node`, and drives the motor ring from the haptic cues
that come back.

    python3 scripts/hw/weave_node.py --host 192.168.0.10
    python3 scripts/hw/weave_node.py --host 192.168.0.10 \
        --ring 17,27,22,23 --doa-cmd "xvf_host GET_DOA"
    python3 scripts/hw/weave_node.py --host ... --no-motors   # audio only

Why the split is this way: the Pi Zero 2 W has 512 MB of RAM and the
recognizers are 600 MB per model, so the Pi cannot caption. It captures and
actuates; the Mac decides. That also puts the salience decision where the
speaker tracker already lives -- the node never analyses anything, which is the
standing rule for the haptic module.

Run `probe_array.py` and `probe_motor.py` first. This script assumes you have
already established the array's device index, the DoA command, and the motor
pins, because guessing any of them wastes a demo.
"""

from __future__ import annotations

import argparse
import select
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autocwi import netaudio as na          # noqa: E402
from autocwi.config import load_config       # noqa: E402
from autocwi.haptics import (                # noqa: E402
    MotorLayout, bearing_weights, layout_from_config,
)

SR = 16_000          # what the pipeline wants; the node resamples to it
BLOCK = 1024         # ~64 ms, matching the Mac's capture cadence
# Bound on a stalled uplink. Generous on purpose: a WiFi hiccup must not drop
# the link (the capture queue growing is the designed response, and the Mac
# reads it as a sequence gap), but a genuinely dead link still has to surface
# instead of hanging until TCP exhausts its retransmits.
SEND_TIMEOUT_S = 10.0


# --------------------------------------------------------------------------
# Motors
# --------------------------------------------------------------------------

class MotorRing:
    """The physical motors. The bearing -> motor mapping is NOT here.

    `autocwi.haptics.bearing_weights` owns that, so it can be tested offline and
    so both ends of the link agree on what a bearing means. This class only
    opens the pins and shapes the pulses in time.
    """

    def __init__(self, layout: MotorLayout):
        self.layout = layout
        self._devices = None
        self._direct_until = 0.0
        self._direct_next_at = 0.0
        if layout.pins:
            try:
                from gpiozero import PWMOutputDevice
                self._devices = [PWMOutputDevice(p) for p in layout.pins]
            except Exception as e:                            # noqa: BLE001
                print(f"[node] motors unavailable ({e}) — running without haptics")
        if layout.pins and not layout.can_encode_direction:
            print("[node] one motor: direction cannot be expressed, so every "
                  "cue pulses it and says nothing about where")

    @property
    def enabled(self) -> bool:
        return self._devices is not None

    def _weights(self, bearing: float | None) -> list[float]:
        if not self._devices:
            return []
        return bearing_weights(self.layout, bearing)

    def pulse(self, flag: str, bearing: float | None, intensity: float) -> None:
        """Actuate one cue. Shapes differ so the two flags are distinguishable.

        `speaker_change` is two short taps -- a turn boundary is an event.
        `emphasis` is one soft swell -- it is a property of a word.
        """
        if not self._devices:
            return
        weights = self._weights(bearing)
        if flag == "speaker_change":
            for _ in range(2):
                self._set(weights, intensity)
                # Coin motors need appreciable spin-up time. The previous
                # 55 ms tap reached the driver but was commonly imperceptible.
                time.sleep(0.18)
                self._set(weights, 0.0)
                time.sleep(0.14)
        else:
            # A gentle swell with a perceptible plateau, rather than a brief
            # PWM peak that ends before the motor reaches useful amplitude.
            for level in (0.25, 0.5, 0.75, 1.0):
                self._set(weights, intensity * level)
                time.sleep(0.05)
            time.sleep(0.16)
            for level in (0.75, 0.5, 0.25):
                self._set(weights, intensity * level)
                time.sleep(0.05)
            self._set(weights, 0.0)

    def follow(self, bearing: float | None, intensity: float,
               pulse_s: float, interval_s: float) -> None:
        """Drive motors directly from the array's current speech bearing.

        This is the low-latency path: the Pi receives a DoA update roughly
        every 120 ms and applies it locally, without waiting for ASR or a
        network round trip through the Mac. No array speech reading means off;
        unlike an event cue, absence must never activate every motor.
        """
        if not self._devices:
            return
        if bearing is None:
            self._set([0.0] * len(self._devices), 0.0)
            self._direct_until = 0.0
            self._direct_next_at = 0.0
            return
        now = time.monotonic()
        if now >= self._direct_next_at:
            self._direct_until = now + pulse_s
            self._direct_next_at = now + interval_s
        if now < self._direct_until:
            self._set(self._weights(bearing), intensity)
        else:
            self._set([0.0] * len(self._devices), 0.0)

    def _set(self, weights: list[float], intensity: float) -> None:
        for dev, w in zip(self._devices, weights):
            dev.value = max(0.0, min(1.0, w * intensity))

    def close(self) -> None:
        for dev in self._devices or []:
            dev.value = 0.0
            dev.close()


# --------------------------------------------------------------------------
# Direction of arrival
# --------------------------------------------------------------------------

class DoAReader:
    """Polls the array's control interface for a bearing.

    Two paths, direct first. A stock XVF3800 answers vendor control transfers
    on EP0, so `xvf_control` reads the bearing with no host binary and no
    subprocess -- which matters here, because the shell path spawned a process
    every 120 ms on a Pi Zero 2 W. `--doa-cmd` remains for other firmware.

    **The bearing is published only while the board reports speech.** The
    XVF3800 HOLDS its last bearing when a room goes quiet rather than blanking
    it, so a reader that just forwards whatever it read points confidently at
    a talker who has left, for as long as the silence lasts. Reporting None
    stops the node sending, the Mac's TTL lapses, and the compass returns to
    `awaiting array` -- absent, which is the honest state, rather than stale.
    """

    def __init__(self, command: str | None, period_s: float = 0.12,
                 offset_deg: float = 0.0):
        self.command = shlex.split(command) if command else None
        self.period_s = period_s
        self.offset_deg = offset_deg
        self.latest: float | None = None
        # Every steered talker beam currently carrying speech, not just the
        # dominant one. Empty is the honest state and means the same as
        # `latest is None`: nothing measured, so nothing claimed.
        self.beams: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.failures = 0
        self._array = None

    def start(self) -> None:
        try:
            from xvf_control import open_array
            self._array = open_array()
        except ImportError:
            self._array = None

        if self._array is not None:
            try:
                print(f"[node] DoA: XVF3800 control lane, firmware "
                      f"{self._array.version()}")
            except Exception as e:                            # noqa: BLE001
                print(f"[node] DoA: control lane found but unreadable ({e})")
                print("[node] on Linux this is usually permissions — add a "
                      "udev rule for 2886:001a")
                self._array = None

        if self._array is None and not self.command:
            print("[node] no DoA source — direction will stay 'awaiting array'")
            return
        target = self._loop_direct if self._array is not None else self._loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._array is not None:
            try:
                self._array.close()
            except Exception:                                 # noqa: BLE001
                pass

    def _loop_direct(self) -> None:
        while not self._stop.is_set():
            try:
                # Beam first: it carries a bearing on ~50% of reads against the
                # VAD gate's ~5%, and direction is only useful for separating a
                # speaker the voice lane has not decided on if it is actually
                # there when that speaker starts. Falls back to the VAD gate,
                # which is the more precise of the two when it does report.
                # One snapshot serves both: the auto-select beam is the
                # dominant bearing, the steered beams are the separate talkers.
                # Reading them apart would double the poll traffic and could
                # straddle a re-steer.
                snapshot = self._array.read_beams()
                bearing = None
                talkers: list[float] = []
                if snapshot is not None:
                    if snapshot[3][1] > 0.0:
                        bearing = snapshot[3][0]
                    talkers = [snapshot[i][0]
                               for i in type(self._array).TALKER_BEAMS
                               if snapshot[i][1] > 0.0]
                if bearing is None:
                    bearing = self._array.read_bearing()
                # None means the board heard no speech for this read. Forward
                # the absence; do NOT hold the previous bearing.
                self.latest = (None if bearing is None
                               else (bearing + self.offset_deg) % 360.0)
                self.beams = [(t + self.offset_deg) % 360.0 for t in talkers]
                self.failures = 0
            except Exception:                                 # noqa: BLE001
                self.failures += 1
                self.latest = None
                self.beams = []
                if self.failures == 10:
                    print("[node] DoA reads failing — was the array unplugged?")
            self._stop.wait(self.period_s)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                p = subprocess.run(self.command, capture_output=True,
                                   text=True, timeout=1.0)
                value = self._parse(p.stdout + p.stderr)
                if value is not None:
                    self.latest = (value + self.offset_deg) % 360.0
                    self.failures = 0
                else:
                    self.failures += 1
            except Exception:                                 # noqa: BLE001
                self.failures += 1
            if self.failures == 10:
                print("[node] DoA command failing — check --doa-cmd")
            self._stop.wait(self.period_s)

    @staticmethod
    def _parse(text: str) -> float | None:
        """Pull the first number out of the utility's reply.

        Deliberately forgiving: XMOS host apps print `DOA: 137`, `137.0`, or a
        key/value line depending on build, and the probe records which. A
        stricter parser would break on a firmware update for no benefit.
        """
        for token in reversed(text.replace(":", " ").replace(",", " ").split()):
            try:
                return float(token)
            except ValueError:
                continue
        return None


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

def resolve_rate(device, requested: int | None) -> int:
    """The rate the array actually offers, not the one we assumed.

    A fixed 48 kHz default is wrong for this hardware: the XVF3800 exposes
    16 kHz and nothing else, so opening it at 48 k fails with PortAudio's
    `Invalid sample rate [-9997]` -- an error that names neither the device
    nor the rate it wanted, and reads like a broken array rather than a bad
    default. Ask the device instead, and keep --rate as an override for
    hardware that misreports.
    """
    if requested is not None:
        return requested
    import sounddevice as sd
    try:
        info = sd.query_devices(device if device is not None
                                else sd.default.device[0])
        return int(info["default_samplerate"])
    except Exception:                                        # noqa: BLE001
        return 48_000


def open_stream(device, rate: int, channels: int):
    import sounddevice as sd
    return sd.InputStream(samplerate=rate, channels=channels, dtype="float32",
                          blocksize=int(BLOCK * rate / SR), device=device)


def downmix(block: np.ndarray, channel: int | None) -> np.ndarray:
    """Array frames -> one mono channel.

    The XVF3800 exposes a processed/beamformed channel alongside raw mics;
    `probe_array.py` tells you which. Averaging raw mics instead would undo the
    beamforming the board just did, so a specific channel is preferred and
    averaging is only the fallback.
    """
    if block.ndim == 1:
        return block
    if channel is not None and channel < block.shape[1]:
        return block[:, channel]
    return block.mean(axis=1)


def resample(block: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resample to the pipeline's rate.

    Linear is adequate here and cheap on a Zero 2 W: the array delivers a
    band-limited, already-processed signal, and the recognizers run at 16 k. If
    this ever measures as a recognition cost, move it to `soxr`, not to a
    hand-rolled polyphase filter.
    """
    if src == dst:
        return block.astype(np.float32, copy=False)
    n = int(round(len(block) * dst / src))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    x = np.linspace(0, len(block) - 1, n, dtype=np.float64)
    return np.interp(x, np.arange(len(block)), block).astype(np.float32)


# --------------------------------------------------------------------------
# Cue receiver
# --------------------------------------------------------------------------

def cue_loop(conn: socket.socket, ring: MotorRing, stop: threading.Event,
             verbose: bool, mac_cues: bool) -> None:
    """Drain optional Mac cues without interfering with direct DoA control."""
    reader = na.FrameReader()
    # Do NOT settimeout() here. This is the SAME socket object the capture
    # thread ships audio on, and a timeout is socket-wide, not per-call -- so
    # a short one set here silently becomes sendall's budget and any brief
    # uplink stall tears the link down as "link lost (timed out)". select()
    # waits for readability without touching shared socket state.
    while not stop.is_set():
        try:
            ready, _, _ = select.select([conn], [], [], 0.5)
        except OSError:
            return
        if not ready:
            continue
        try:
            data = conn.recv(4096)
        except OSError:
            return
        if not data:
            return
        for frame in reader.feed(data):
            if frame.kind != na.KIND_CUE:
                continue
            if not mac_cues:
                continue
            body = frame.json()
            bearing = body.get("direction_deg")
            if verbose:
                where = f"{bearing:.0f}deg" if bearing is not None else "no bearing"
                print(f"[node] cue {body['flag']:15s} {where}")
            ring.pulse(body["flag"], bearing, float(body.get("intensity", 0.8)))


# --------------------------------------------------------------------------

def run(args) -> int:
    configured_layout = layout_from_config(load_config(args.config))
    if args.no_motors:
        layout = MotorLayout()
    elif args.ring is None:
        layout = configured_layout
    else:
        pins = [int(p) for p in args.ring.split(",") if p.strip()]
        # Retain an explicitly configured physical placement for its matching
        # pin list; --ring still supports temporary probe arrangements.
        layout = configured_layout if pins == configured_layout.pins else MotorLayout(pins=pins)
    ring = MotorRing(layout)
    doa = DoAReader(args.doa_cmd, offset_deg=args.doa_offset)
    doa.start()
    stop = threading.Event()

    print(f"[node] connecting to {args.host}:{args.port}")
    while not stop.is_set():
        try:
            conn = socket.create_connection((args.host, args.port), timeout=5.0)
        except OSError as e:
            print(f"[node] no host ({e}) — retrying in 2s")
            time.sleep(2.0)
            continue

        print("[node] connected")
        # The 5 s connect budget must not carry into streaming: sendall is
        # meant to wait out a stalled uplink, not raise and reconnect.
        conn.settimeout(SEND_TIMEOUT_S)
        try:
            with conn:
                conn.sendall(na.pack_hello(SR, BLOCK))
                threading.Thread(target=cue_loop,
                                 args=(conn, ring, stop, args.verbose,
                                       args.mac_cues),
                                 daemon=True).start()
                _stream(conn, args, doa, ring, stop)
        except (OSError, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                stop.set()
                break
            print(f"[node] link lost ({e}) — reconnecting")
            time.sleep(1.0)

    doa.stop()
    ring.close()
    print("[node] stopped")
    return 0


def _stream(conn: socket.socket, args, doa: DoAReader, ring: MotorRing,
            stop: threading.Event) -> None:
    """Capture and ship until the link drops.

    The sequence number is monotonic and never reset while connected, so the
    Mac can tell a genuine node-side drop from a reconnect. Blocks are sent as
    they are captured; if the uplink stalls, `sendall` blocks and the capture
    queue grows -- which shows up as a sequence gap at the far end rather than
    as silently mistimed audio.
    """
    # Resolve onto args, not into a local: the resampler downstream reads
    # args.rate, and a local would leave it None there.
    args.rate = rate = resolve_rate(args.device, args.rate)
    stream = open_stream(args.device, rate, args.channels)
    seq = 0
    sent_doa_at = 0.0
    with stream:
        print(f"[node] capturing {rate}Hz x{args.channels} "
              f"-> {SR}Hz mono, block {BLOCK}"
              + ("  (no resampling)" if rate == SR else ""))
        while not stop.is_set():
            raw, overflowed = stream.read(stream.blocksize)
            if overflowed:
                # The Pi could not keep up. Do NOT paper over it: skipping the
                # sequence number is how the Mac learns audio was lost, and
                # live capture is lossless by rule.
                seq += 1
                print("[node] input overflow — a block was dropped")
                continue
            mono = resample(downmix(raw, args.channel), args.rate, SR)
            conn.sendall(na.pack_audio(seq, mono))

            # Local low-latency haptics: the speaker's current bearing drives
            # the physical layout without waiting for caption finalization.
            ring.follow(doa.latest, args.intensity, args.direct_pulse_s,
                        args.direct_interval_s)

            now = time.monotonic()
            if doa.latest is not None and now - sent_doa_at >= 0.1:
                conn.sendall(na.pack_doa(seq, doa.latest, beams=doa.beams))
                sent_doa_at = now
            seq += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True,
                    help="the Mac running `autocwi live --node --host 0.0.0.0`")
    ap.add_argument("--port", type=int, default=7338)
    ap.add_argument("--device", default=None,
                    help="array capture device index or name (see probe_array.py)")
    ap.add_argument("--rate", type=int, default=None,
                    help="array capture rate; default asks the device, resampled to 16k")
    ap.add_argument("--channels", type=int, default=1,
                    help="channels to open on the array (default: 1)")
    ap.add_argument("--channel", type=int, default=None,
                    help="which channel carries processed speech; default averages")
    ap.add_argument("--ring", default=None,
                    help="override configured motor BCM pins, clockwise from front")
    ap.add_argument("--config", default=None,
                    help="path to config.yaml (default: repository config.yaml)")
    ap.add_argument("--no-motors", action="store_true",
                    help="audio and direction only")
    ap.add_argument("--intensity", type=float, default=0.8, metavar="0..1",
                    help="direct DoA vibration strength (default: 0.8)")
    ap.add_argument("--direct-pulse-ms", type=float, default=100.0,
                    metavar="MS", help="direct DoA pulse length (default: 100)")
    ap.add_argument("--direct-interval-ms", type=float, default=300.0,
                    metavar="MS", help="time between direct DoA pulses (default: 300)")
    ap.add_argument("--mac-cues", action="store_true",
                    help="also play delayed caption cues from the Mac")
    ap.add_argument("--doa-cmd", default=None, metavar="CMD",
                    help="shell command printing the bearing, from probe_array.py")
    ap.add_argument("--doa-offset", type=float, default=0.0, metavar="DEG",
                    help="rotate bearings so 0deg matches the case's front")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    args.intensity = max(0.0, min(1.0, args.intensity))
    args.direct_pulse_s = max(0.02, args.direct_pulse_ms / 1000.0)
    args.direct_interval_s = max(args.direct_pulse_s, args.direct_interval_ms / 1000.0)

    if args.device is not None and str(args.device).lstrip("-").isdigit():
        args.device = int(args.device)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
