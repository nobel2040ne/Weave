"""Live captions: microphone -> true streaming ASR + prosody -> SSE events.

The default recognizer is the accuracy-first 1120 ms sherpa-onnx Nemotron 0.6B
profile. It supplies provisional spoken-onset cues and stable commits; a
Parakeet endpoint pass owns durable text. Explicit readahead mode additionally
loads a 160 ms profile for highly revisable early drafts. Fast mode deliberately
does not pay for inference whose output it would hide.

Word events share the CaptionSpec word shape (text/start/end/speaker/
loudness/pitch/loudness_db/pitch_hz/conf) plus an absolute `t` wall-clock
onset, stable ``word_id``, and revision-capable speaker metadata. Only final
word events are appended to ``live_events.jsonl``. A later full word record
with the same id may revise speaker attribution without duplicating text.
``hypothesis`` and ``cue`` events are optional live-display extensions and may
be replaced at any time. Cues improve visual synchronization but are never
written to the durable log or intended for haptic actuation.

Everything runs locally: sounddevice (mic) -> sherpa-onnx -> parselmouth,
served to the browser over plain stdlib HTTP + Server-Sent Events. The former
pause-segmented Whisper path remains available with ``live --whisper MODEL``
for comparison and keeps its adaptive threshold/VAD filtering.
"""

from __future__ import annotations

import http.server
import difflib
import json
import math
import mimetypes
import os
import platform
import queue
import re
import socket
import threading
import time
import unicodedata
import warnings
import webbrowser
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import numpy as np

from autocwi import haptics, netaudio

SR = 16_000
BLOCK = 1024  # samples per audio block (~64 ms)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(x**2))), 1e-8))


def _span_db(x: np.ndarray, percentile: float = 90.0, frame_s: float = 0.03) -> float:
    """How loud a WORD was — the level where the voice actually is.

    THE WHOLE-SPAN RMS CANNOT TELL A SHOUT FROM A WHISPER, and that is why the
    size channel never worked. A word's span includes its stops, its unvoiced
    consonants and the gaps between phones, all near-silent whatever the
    speaker is doing. Measured on the PR film, whole-span median makes the
    emphasised "louder" read QUIETER than the calm words around it; scored on
    each word's loud frames instead, the same audio separates "louder" from
    "softer" by 12.2 dB, in the right direction, from level alone.

    **The percentile stays 90.** The median ranks better within a single line
    and shipped WORSE — the crest comes from a word's position in the
    speaker's running percentile window, so lowering every word's dB moves the
    window with it. "louder" fell to the bare pop. Reverted.
    """
    if not len(x):
        return -80.0
    step = max(1, int(frame_s * SR))
    if len(x) < step * 2:
        return _rms_db(x)
    frames = [x[i:i + step] for i in range(0, len(x) - step + 1, step)]
    levels = np.array([float(np.sqrt(np.mean(f**2))) for f in frames])
    return 20 * np.log10(max(float(np.percentile(levels, percentile)), 1e-8))


def _realtime_voice_features(
    samples: np.ndarray,
    pitch_floor_hz: float = 75.0,
    pitch_ceiling_hz: float = 500.0,
) -> tuple[float, float, float]:
    """Return ``(f0_hz, periodicity, spectral_centroid_hz)`` for one block.

    This intentionally small autocorrelation estimator serves the continuous
    UI indicator, not the durable word prosody model. It stays on the capture
    thread, has no model dependency, and gives the browser a fresh observation
    every ~64 ms—well before ASR emits text.
    """

    frame = np.asarray(samples, dtype=np.float64)
    if len(frame) < 256 or float(np.max(np.abs(frame))) < 1e-6:
        return 0.0, 0.0, 0.0
    frame = frame - float(np.mean(frame))
    windowed = frame * np.hanning(len(frame))
    energy = float(np.dot(windowed, windowed))
    if energy < 1e-10:
        return 0.0, 0.0, 0.0

    # The centroid is a deliberately coarse timbre/brightness cue.
    magnitude = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(windowed), 1.0 / SR)
    mag_sum = float(np.sum(magnitude))
    centroid = (
        float(np.dot(magnitude, frequencies) / mag_sum)
        if mag_sum > 1e-10 else 0.0
    )

    # FFT autocorrelation avoids an O(n²) pass on every capture block.
    fft_size = 1 << (2 * len(windowed) - 1).bit_length()
    spectrum = np.fft.rfft(windowed, fft_size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), fft_size)[:len(frame)]
    if correlation[0] <= 1e-10:
        return 0.0, 0.0, centroid
    correlation /= correlation[0]
    min_lag = max(1, int(SR / max(pitch_ceiling_hz, 1.0)))
    max_lag = min(len(correlation) - 2, int(SR / max(pitch_floor_hz, 1.0)))
    if max_lag <= min_lag:
        return 0.0, 0.0, centroid
    region = correlation[min_lag:max_lag + 1]
    peak_offset = int(np.argmax(region))
    lag = min_lag + peak_offset
    periodicity = float(np.clip(correlation[lag], 0.0, 1.0))
    if periodicity < 0.28:
        return 0.0, periodicity, centroid

    # Sub-sample parabolic interpolation removes visible quantization in the
    # circle bead without pretending this UI estimator is a phonetic model.
    before, center, after = correlation[lag - 1:lag + 2]
    denominator = float(before - 2.0 * center + after)
    offset = (
        0.5 * float(before - after) / denominator
        if abs(denominator) > 1e-12 else 0.0
    )
    refined_lag = lag + float(np.clip(offset, -0.5, 0.5))
    return float(SR / refined_lag), periodicity, centroid


def _delivery_profile(
    *,
    force: float,
    attack: float,
    contour: float,
    flow: float,
    texture: float,
    confidence: float,
    duration_s: float,
    cfg: dict,
) -> str:
    """Name an audible delivery shape without inferring an inner emotion.

    DIAGNOSTIC ONLY, and it must never choose a caption motion family. CWI 2.2.3
    gives every word the same cue, while the continuous type axes in
    `web/src/lib/caption-motion.ts` read the acoustics directly. Thresholds here
    are unavoidable because a label is discrete; the type axes are not and must
    not inherit their dead zone.
    """

    profile = dict(cfg.get("profile", {}) or {})
    if confidence < float(profile.get("min_confidence", 0.22)):
        return "steady"
    contour_threshold = float(profile.get("contour_threshold", 0.25))
    if contour >= contour_threshold:
        return "rising"
    if contour <= -contour_threshold:
        return "falling"
    if (
        flow >= float(profile.get("sustained_flow_min", 0.68))
        and duration_s >= float(profile.get("sustained_duration_s", 0.34))
    ):
        return "sustained"
    if force >= float(profile.get("forceful_force_min", 0.68)):
        return "forceful"
    if (
        force <= float(profile.get("gentle_force_max", 0.32))
        and attack <= float(profile.get("gentle_attack_max", 0.24))
        and texture >= float(profile.get("gentle_texture_min", 0.60))
    ):
        return "gentle"
    if texture >= float(profile.get("textured_min", 0.60)):
        return "textured"
    return "steady"


# ---------------------------------------------------------------------------
# Audio sources: microphone or a wav file paced in real time (demo/testing)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioChunk:
    """Captured audio with its source-clock position.

    ``discontinuity`` is reserved for a real capture-device gap. Normal decoder
    backlog is losslessly batched and never converted into a discontinuity.

    ``samples`` always stays at the true captured level: prosody measures
    ``loudness_db`` from it, and that drives the CWI volume -> type-size
    channel. ``asr_samples`` is the optionally gained copy handed to the
    recognizer, so amplifying a quiet talker can never make a whisper render
    at the same size as a shout.
    """

    samples: np.ndarray
    source_start: float
    discontinuity: bool = False
    dropped_s: float = 0.0
    asr_samples: np.ndarray | None = None

    @property
    def source_end(self) -> float:
        return self.source_start + len(self.samples) / SR

    @property
    def recognizer_samples(self) -> np.ndarray:
        return self.samples if self.asr_samples is None else self.asr_samples


def coalesce_audio_chunks(chunks: Iterable[AudioChunk],
                          previous_end: float) -> AudioChunk:
    """Losslessly batch every queued capture block for decoder catch-up."""

    pending = list(chunks)
    if not pending:
        raise ValueError("at least one audio chunk is required")
    source_start = pending[0].source_start
    gap = max(0.0, source_start - previous_end)
    samples = np.concatenate([chunk.samples for chunk in pending]).astype(np.float32,
                                                                          copy=False)
    asr_samples = None
    if any(chunk.asr_samples is not None for chunk in pending):
        asr_samples = np.concatenate(
            [chunk.recognizer_samples for chunk in pending]
        ).astype(np.float32, copy=False)
    return AudioChunk(
        samples=samples,
        source_start=source_start,
        discontinuity=gap > 1.5 / SR or any(chunk.discontinuity for chunk in pending),
        dropped_s=gap + sum(chunk.dropped_s for chunk in pending),
        asr_samples=asr_samples,
    )


def list_input_devices() -> str:
    """Render the selectable capture devices for ``live --list-devices``."""

    import sounddevice as sd

    default = sd.default.device[0]
    lines = []
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        mark = " (default)" if index == default else ""
        lines.append(f"  {index}: {dev['name']}{mark}")
    return "\n".join(lines) or "  (no input devices found)"


def mic_blocks(stop: threading.Event, device: int | str | None = None):
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        raise SystemExit(
            f"microphone capture needs sounddevice ({e}) — pip install sounddevice"
        ) from e

    # The PortAudio callback must never block behind recognition. The consumer
    # drains and batches every pending block on each pass; because the selected
    # recognizers run faster than real time, transient backlog converges without
    # either dropping words or accumulating permanent delay.
    q: queue.Queue[AudioChunk] = queue.Queue()
    captured_samples = 0

    def cb(indata, frames, t, status):
        nonlocal captured_samples
        samples = indata[:, 0].copy()
        chunk = AudioChunk(samples, captured_samples / SR)
        captured_samples += len(samples)
        q.put_nowait(chunk)

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK, latency="low", callback=cb,
                        device=device):
        try:
            name = sd.query_devices(device if device is not None
                                    else sd.default.device[0])["name"]
        except Exception:
            name = "default input"
        print(f"[live] microphone open: {name} — speak! (Ctrl-C to quit)")
        previous_end = 0.0
        while not stop.is_set():
            try:
                pending = [q.get(timeout=0.2)]
            except queue.Empty:
                continue
            while True:
                try:
                    pending.append(q.get_nowait())
                except queue.Empty:
                    break
            chunk = coalesce_audio_chunks(pending, previous_end)
            previous_end = chunk.source_end
            if chunk.discontinuity:
                print(f"[live] capture device lost {chunk.dropped_s:.2f}s — resyncing")
            yield chunk


class NodeLink:
    """The connection to the hardware node (ReSpeaker array + Pi Zero 2 W).

    The Pi cannot host the recognizers (512 MB against 600 MB-per-model
    weights), so with the array plugged into the Pi the capture path crosses the
    network. This owns that socket in both directions: audio and
    direction-of-arrival arrive, haptic cues go back.

    It is a THIRD audio source beside ``mic_blocks`` and ``file_blocks`` and
    yields the same ``AudioChunk``, so nothing downstream knows the difference.

    Direction expires. ``direction_deg`` returns None once the newest
    observation is older than ``doa_ttl_s``, so an unplugged array or a node
    that stopped reporting falls back to the compass's `awaiting array` instead
    of freezing a stale bearing on screen. Never fabricate direction.
    """

    # Reject a span's bearing below this mean resultant length. 0.5 is two
    # readings ~120 deg apart: past that they describe two directions, not one
    # noisy one, and an averaged answer would name a place nobody spoke from.
    MIN_BEARING_CONCENTRATION = 0.5

    def __init__(self, host: str = "0.0.0.0", port: int = 7338,
                 doa_ttl_s: float = 1.5, history: int = 4096) -> None:
        self.host = host
        self.port = port
        self.doa_ttl_s = doa_ttl_s
        self._doa: tuple[float, float] | None = None   # (deg, monotonic stamp)
        # (source_time, deg) for attribution, which scores a WORD SPAN and so
        # needs the bearing during that span, not the latest one. Bearings are
        # recorded against the source clock the words are timed on; the DoA
        # frame's position in the stream is its position in time, because the
        # node interleaves it with the audio on one ordered connection.
        self._doa_history: deque[tuple[float, float]] = deque(maxlen=history)
        self._doa_lock = threading.Lock()
        self._conn: socket.socket | None = None
        self._send_lock = threading.Lock()
        self.dropped_blocks = 0

    @property
    def direction_deg(self) -> float | None:
        with self._doa_lock:
            if self._doa is None:
                return None
            deg, stamp = self._doa
            if time.monotonic() - stamp > self.doa_ttl_s:
                return None
            return deg

    def bearing_for_span(self, start_s: float, end_s: float) -> float | None:
        """The array's bearing while this word was spoken, or None.

        None when the array reported nothing across the span -- because it was
        silent, absent, or the reading was rejected. Absent is the honest
        answer and the caller must not substitute the latest bearing: a word
        spoken while the array heard nothing has no direction, and borrowing a
        neighbouring one is exactly the fabrication 'omit it' forbids.

        Averaged as a CIRCULAR mean. Bearings are angles, so 359 and 1 average
        to 0, not to 180 -- and a talker straight ahead of the case is
        precisely where the naive mean is worst.
        """
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        with self._doa_lock:
            spanned = [deg for t, deg in self._doa_history
                       if start_s <= t <= end_s]
        if not spanned:
            return None
        radians = [math.radians(d) for d in spanned]
        x = sum(math.cos(r) for r in radians)
        y = sum(math.sin(r) for r in radians)
        # Mean resultant length: 1 when every reading agrees, 0 when they
        # cancel. Testing `x == y == 0` instead does not work -- sin(180 deg)
        # is 1.2e-16, not zero, so two opposed bearings sail through it and
        # atan2 confidently returns 90 degrees.
        concentration = math.hypot(x, y) / len(radians)
        if concentration < self.MIN_BEARING_CONCENTRATION:
            # The readings point more apart than together, so their mean is an
            # artefact of arithmetic rather than a direction anyone spoke from.
            return None
        return math.degrees(math.atan2(y, x)) % 360.0

    def send_cue(self, flag: str, direction_deg: float | None,
                 intensity: float, seq: int = 0) -> bool:
        """Fire one haptic actuation. False when no node is attached.

        Never raises: a dead haptic link must degrade to no vibration, never
        interrupt captions.
        """
        conn = self._conn
        if conn is None:
            return False
        try:
            with self._send_lock:
                conn.sendall(
                    netaudio.pack_cue(seq, flag, direction_deg, intensity)
                )
            return True
        except OSError:
            return False

    def blocks(self, stop: threading.Event):
        """Yield captured audio, reconnecting for as long as capture runs.

        A booth demo must survive the node rebooting or WiFi dropping, so a lost
        connection waits for the next one rather than ending the capture.
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(0.5)
        print(f"[live] waiting for hardware node on {self.host}:{self.port}")
        captured = 0
        try:
            while not stop.is_set():
                try:
                    conn, peer = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                print(f"[live] hardware node connected: {peer[0]}")
                self._conn = conn
                # A reconnect is a real capture gap, so the first block after
                # one is marked and the captioner resets its stream.
                yield from self._serve(conn, stop, first_is_gap=captured > 0,
                                       counter=lambda n: None)
                self._conn = None
                conn.close()
                captured += 1
                if not stop.is_set():
                    print("[live] hardware node disconnected — waiting for it")
        finally:
            self._conn = None
            server.close()

    def _serve(self, conn: socket.socket, stop: threading.Event,
               first_is_gap: bool, counter):
        reader = netaudio.FrameReader()
        tracker = netaudio.SequenceTracker()
        conn.settimeout(0.5)
        source_start = 0.0
        pending_gap = first_is_gap
        while not stop.is_set():
            try:
                data = conn.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if not data:
                return
            for frame in reader.feed(data):
                if frame.kind == netaudio.KIND_DOA:
                    body = frame.json()
                    deg = float(body["doa_deg"])
                    with self._doa_lock:
                        self._doa = (deg, time.monotonic())
                        # `source_start` is where the next audio block lands on
                        # the source clock, so it timestamps this bearing on the
                        # same timeline the word spans use.
                        self._doa_history.append((source_start, deg))
                    continue
                if frame.kind == netaudio.KIND_HELLO:
                    body = frame.json()
                    rate = int(body.get("sample_rate", SR))
                    if rate != SR:
                        # Resampling belongs on the node, where it is one cheap
                        # pass; accepting a wrong rate here would silently
                        # detune every timing in the pipeline.
                        print(f"[live] node sample rate {rate} != {SR} — refusing")
                        return
                    print(f"[live] node ready: {body.get('node', 'unknown')}")
                    continue
                if frame.kind != netaudio.KIND_AUDIO:
                    continue

                gap = tracker.observe(frame.seq)
                samples = frame.samples()
                dropped_s = 0.0
                if gap or pending_gap:
                    dropped_s = max(0.0, tracker.dropped) * len(samples) / SR
                    self.dropped_blocks = tracker.dropped
                    print(f"[live] node dropped {tracker.dropped} blocks "
                          f"({dropped_s:.2f}s) — resyncing")
                chunk = AudioChunk(
                    samples=samples,
                    source_start=source_start,
                    discontinuity=gap or pending_gap,
                    dropped_s=dropped_s,
                )
                pending_gap = False
                source_start = chunk.source_end
                yield chunk


def net_blocks(stop: threading.Event, link: "NodeLink"):
    """Adapter so the source selection reads like the other two."""

    yield from link.blocks(stop)


def sample_clip_path(language: str = "en") -> str:
    """The bundled language sample, for testing live mode without a mic.

    English uses the CWI reference dialogue at roughly -36 dBFS, so it also
    exercises input gain. Korean uses a CC BY 4.0 FLEURS utterance.
    `--sample --lang ko` must never test a Korean recognizer with the English
    reference clip.
    """

    root = Path(__file__).resolve().parent.parent
    if language == "ko":
        korean = root / "assets" / "sample-ko.wav"
        if korean.exists():
            return str(korean)
        raise FileNotFoundError(
            "bundled Korean sample not found: assets/sample-ko.wav"
        )
    candidates = [
        # The PR film: reference material AND the clip `--sample` streams. It
        # was checked in twice, byte for byte, until 2026-08-13.
        root / "docs" / "reference" / "PR_Flim.mp4",
        root / "AE PROJECT" / "AE PROJECT" / "(Footage)" / "ASETS" / "Video"
             / "Cena_ref_CI_Template_v02a.mp4",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise SystemExit(
        "no bundled sample clip found — expected one of:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def file_blocks(path: str | Path, realtime: bool = True, loop: bool = False,
                start_s: float = 0.0, _clock=None, _sleep=None):
    """Yield file audio against a source clock without dropping late blocks.

    Deadline pacing is important: sleeping for every block *after* inference
    adds inference time to media time and guarantees steadily growing delay.

    ``loop`` restarts the clip when it ends, so a fixed sample plays as a
    continuous live feed. The source clock keeps advancing across loops and the
    first block of each new pass is marked as a discontinuity, so the captioner
    resets its stream and the browser clears the previous pass instead of
    stacking the same words on top of each other.

    ``start_s`` skips the head of the clip. The source clock still starts at
    zero, so the playhead, the read-ahead floor and every word span stay in the
    same frame of reference — only the audio handed to the recognizer is
    shorter.
    """

    import librosa

    clock = _clock or time.monotonic
    sleep = _sleep or time.sleep
    with warnings.catch_warnings():
        # mp4/aac decodes through librosa's audioread fallback, which warns that
        # PySoundFile "failed" (it just cannot read compressed audio) — harmless.
        warnings.simplefilter("ignore")
        data, _ = librosa.load(str(path), sr=SR, mono=True)
    skip = max(0, int(max(0.0, start_s) * SR))
    if skip < len(data):
        data = data[skip:]
    duration = len(data) / SR
    started = clock()
    base = 0.0
    first_pass = True
    while first_pass or loop:
        i = 0
        while i < len(data):
            source_start = base + i / SR
            if realtime:
                delay = started + source_start - clock()
                if delay > 0:
                    sleep(delay)
            blk = data[i:i + BLOCK].astype(np.float32)
            yield AudioChunk(blk, source_start,
                             discontinuity=(loop and i == 0 and not first_pass))
            i += len(blk)
        base += duration
        first_pass = False


class InputGain:
    """Drive the recognizer's copy of the audio toward a usable level.

    A streaming transducer is trained on speech at conventional levels. Well
    below that it simply stops emitting non-blank tokens, so a quiet talker
    produces no captions at all — not because anything rejected the audio, but
    because there was too little signal to decode. This scales the recognizer
    copy only; ``AudioChunk.samples`` keeps the true captured level so the CWI
    volume -> type-size mapping still distinguishes a whisper from a shout.

    Gain is held (not reduced) during silence so room tone is never amplified
    into hiss, and it moves in bounded steps per second so the encoder never
    sees a level discontinuity mid-word.
    """

    def __init__(self, cfg: dict):
        gain_cfg = dict(cfg.get("live", {}).get("input_gain", {}) or {})
        self.enabled = bool(gain_cfg.get("enabled", True))
        self.target_dbfs = float(gain_cfg.get("target_dbfs", -26.0))
        self.min_gain_db = float(gain_cfg.get("min_gain_db", 0.0))
        self.max_gain_db = float(gain_cfg.get("max_gain_db", 30.0))
        self.floor_margin_db = float(gain_cfg.get("floor_margin_db", 8.0))
        self.absolute_floor_db = float(gain_cfg.get("absolute_floor_db", -72.0))
        self.attack_db_per_s = float(gain_cfg.get("attack_db_per_s", 24.0))
        self.release_db_per_s = float(gain_cfg.get("release_db_per_s", 6.0))
        self.acquire_db_per_s = float(gain_cfg.get("acquire_db_per_s", 60.0))
        self.acquired = False
        self.headroom_dbfs = float(gain_cfg.get("headroom_dbfs", -1.0))
        self.gain_db = float(gain_cfg.get("initial_gain_db", 0.0))
        history_s = float(gain_cfg.get("floor_window_s", 6.0))
        self.levels: deque[float] = deque(maxlen=max(8, int(history_s * SR / BLOCK)))
        self.floor_db = self.absolute_floor_db
        self.rms_db = self.absolute_floor_db
        self.peak_db = self.absolute_floor_db
        self.speech = False
        self.clipping = False
        prosody_cfg = dict(cfg.get("prosody", {}) or {})
        self.pitch_floor_hz = float(prosody_cfg.get("pitch_floor_hz", 75.0))
        self.pitch_ceiling_hz = float(prosody_cfg.get("pitch_ceiling_hz", 500.0))
        self.pitch_hz = 0.0
        self.pitch_confidence = 0.0
        self.spectral_centroid_hz = 0.0
        self._pitch_history: deque[float] = deque(maxlen=3)
        self.delivery_cfg = dict(cfg.get("live", {}).get("delivery", {}) or {})
        self.cfg_live_db_range = list(
            cfg.get("live", {}).get("db_range", [-55.0, -18.0])
        )
        self.delivery_force = 0.0
        self.delivery_attack = 0.0
        self.delivery_contour = 0.0
        self.delivery_flow = 0.0
        self.delivery_texture = 0.0
        self.delivery_confidence = 0.0
        self.delivery_profile = "steady"
        self._delivery_pitch_history: deque[float] = deque(maxlen=5)
        self._previous_rms_db = self.absolute_floor_db
        self._speech_run_s = 0.0
        # Set to a zero-arg callable returning the newest bearing in degrees, or
        # None when nothing is measuring one (no array, or the reading expired).
        # `NodeLink.direction_deg` satisfies this; a local mic leaves it unset,
        # which is why mono still shows `awaiting array`.
        self.direction_source = None

    def _update_floor(self, rms_db: float) -> None:
        self.levels.append(rms_db)
        if len(self.levels) >= 8:
            self.floor_db = float(np.percentile(self.levels, 20))
        else:
            self.floor_db = float(min(self.floor_db, rms_db))

    def process(self, chunk: AudioChunk) -> AudioChunk:
        raw = chunk.samples
        if not len(raw):
            return chunk
        # numpy scalars leak into the SSE payload otherwise: np.float64 is fine
        # because it subclasses float, but np.bool_ does not subclass bool and
        # json.dumps rejects it.
        self.rms_db = float(_rms_db(raw))
        self.peak_db = float(20 * np.log10(max(float(np.max(np.abs(raw))), 1e-8)))
        self.clipping = bool(self.peak_db > -0.5)
        self._update_floor(self.rms_db)
        self.speech = bool(self.rms_db > self.floor_db + self.floor_margin_db and
                           self.rms_db > self.absolute_floor_db)
        if self.speech:
            pitch, confidence, centroid = _realtime_voice_features(
                raw, self.pitch_floor_hz, self.pitch_ceiling_hz
            )
            self.pitch_confidence = float(confidence)
            if pitch > 0:
                self._pitch_history.append(float(pitch))
                self.pitch_hz = float(np.median(self._pitch_history))
            else:
                self.pitch_hz = 0.0
            # A short EMA makes timbre legible without making it lag speech.
            if self.spectral_centroid_hz <= 0:
                self.spectral_centroid_hz = float(centroid)
            else:
                self.spectral_centroid_hz += (
                    float(centroid) - self.spectral_centroid_hz
                ) * 0.28
            self._update_delivery(len(raw) / SR)
        else:
            self.pitch_hz = 0.0
            self.pitch_confidence = 0.0
            self.spectral_centroid_hz = 0.0
            self._pitch_history.clear()
            self._delivery_pitch_history.clear()
            self._speech_run_s = 0.0
            self.delivery_force = 0.0
            self.delivery_attack = 0.0
            self.delivery_contour = 0.0
            self.delivery_flow = 0.0
            self.delivery_texture = 0.0
            self.delivery_confidence = 0.0
            self.delivery_profile = "steady"
        self._previous_rms_db = self.rms_db
        if not self.enabled:
            self.gain_db = 0.0
            return chunk

        # Only speech-like blocks may move the gain. Silence holds it steady.
        if self.speech:
            desired = float(np.clip(self.target_dbfs - self.rms_db,
                                    self.min_gain_db, self.max_gain_db))
            dt = len(raw) / SR
            # Shed gain quickly so a sudden loud passage cannot clip; add it
            # back slowly so the level does not audibly pump between words.
            if desired < self.gain_db:
                rate = self.attack_db_per_s
            elif not self.acquired:
                # Until the gain has converged once, ramp fast: at a slow
                # tracking rate the opening sentence is still under-gained and
                # gets dropped, which is the very failure this exists to fix.
                rate = self.acquire_db_per_s
            else:
                rate = self.release_db_per_s
            step = rate * dt
            self.gain_db += float(np.clip(desired - self.gain_db, -step, step))
            self.gain_db = float(np.clip(self.gain_db, self.min_gain_db, self.max_gain_db))
            if abs(desired - self.gain_db) < 1.0:
                self.acquired = True

        if self.gain_db <= 0.001:
            return chunk
        scaled = raw * (10.0 ** (self.gain_db / 20.0))
        ceiling = 10.0 ** (self.headroom_dbfs / 20.0)
        peak = float(np.max(np.abs(scaled))) if len(scaled) else 0.0
        if peak > ceiling:
            scaled = scaled * (ceiling / peak)
        return replace(chunk, asr_samples=scaled.astype(np.float32, copy=False))

    def _update_delivery(self, dt: float) -> None:
        """Update the real-time voice orb from the current 64 ms observation."""

        self._speech_run_s += dt
        smoothing = float(self.delivery_cfg.get("realtime_smoothing", 0.28))
        smoothing = float(np.clip(smoothing, 0.01, 1.0))
        db_lo, db_hi = map(float, self.cfg_live_db_range)
        raw_force = float(np.clip(
            (self.rms_db - db_lo) / max(1.0, db_hi - db_lo), 0.0, 1.0
        ))
        attack_scale = float(self.delivery_cfg.get("attack_full_scale_db", 12.0))
        raw_attack = float(np.clip(
            (self.rms_db - self._previous_rms_db) / max(1.0, attack_scale),
            0.0,
            1.0,
        ))
        if self.pitch_hz > 0 and self.pitch_confidence >= float(
            self.delivery_cfg.get("pitch_confidence_min", 0.28)
        ):
            self._delivery_pitch_history.append(self.pitch_hz)
        raw_contour = 0.0
        if len(self._delivery_pitch_history) >= 2:
            semitones = 12.0 * math.log2(
                self._delivery_pitch_history[-1]
                / max(self._delivery_pitch_history[0], 1.0)
            )
            raw_contour = float(np.clip(
                semitones / float(self.delivery_cfg.get(
                    "contour_full_scale_semitones", 5.0
                )),
                -1.0,
                1.0,
            ))
        brightness_lo, brightness_hi = map(
            float, self.delivery_cfg.get("texture_brightness_hz", [700, 4200])
        )
        brightness = float(np.clip(
            (self.spectral_centroid_hz - brightness_lo)
            / max(1.0, brightness_hi - brightness_lo),
            0.0,
            1.0,
        ))
        raw_flow = float(np.clip(
            0.75 * self.pitch_confidence
            + 0.25 * (1.0 - min(1.0, abs(raw_contour))),
            0.0,
            1.0,
        ))
        raw_texture = float(np.clip(
            0.70 * (1.0 - self.pitch_confidence) + 0.30 * brightness,
            0.0,
            1.0,
        ))
        raw_confidence = float(np.clip(
            0.45 + 0.55 * self.pitch_confidence, 0.0, 1.0
        ))
        for name, value in (
            ("delivery_force", raw_force),
            ("delivery_attack", raw_attack),
            ("delivery_contour", raw_contour),
            ("delivery_flow", raw_flow),
            ("delivery_texture", raw_texture),
            ("delivery_confidence", raw_confidence),
        ):
            current = float(getattr(self, name))
            setattr(self, name, current + (value - current) * smoothing)
        self.delivery_profile = _delivery_profile(
            force=self.delivery_force,
            attack=self.delivery_attack,
            contour=self.delivery_contour,
            flow=self.delivery_flow,
            texture=self.delivery_texture,
            confidence=self.delivery_confidence,
            duration_s=self._speech_run_s,
            cfg=self.delivery_cfg,
        )

    def status(self) -> str:
        """Coarse state used by both the console and the browser meter."""

        if self.clipping:
            return "clipping"
        if self.rms_db <= self.absolute_floor_db:
            return "no-signal"
        if self.speech:
            headroom = self.max_gain_db - self.gain_db
            if self.rms_db + self.gain_db < self.target_dbfs - 12.0 and headroom < 1.0:
                return "too-quiet"
            return "good"
        return "idle"

    def level_event(self, t: float) -> dict:
        # Direction is OMITTED, never defaulted, when no array is reporting.
        # The compass keys on the field's absence to show `awaiting array`, and
        # a fabricated bearing would be a claim about the room that nothing
        # measured. `NodeLink.direction_deg` already expires a stale reading.
        direction = None
        direction_source = getattr(self, "direction_source", None)
        if direction_source is not None:
            direction = direction_source()
        # Where each speaker slot sits, for the compass to mark. Optional and
        # omitted when empty: the compass shows the CURRENT bearing, and these
        # are the standing positions, which is the part it cannot infer.
        slots = None
        slots_source = getattr(self, "speaker_slots_source", None)
        if slots_source is not None:
            found = slots_source()
            if found:
                slots = [round(float(b), 1) for b in found]
        # Where each DECIDED speaker has been heard from. Carries the speaker
        # id explicitly, unlike `speaker_slots_deg`, whose colour comes from
        # its array INDEX -- the ring marks who is where, so it must name who.
        marks = None
        marks_source = getattr(self, "speaker_bearings_source", None)
        if marks_source is not None:
            found = marks_source()
            if found:
                marks = found
        return {
            "type": "level",
            **({"direction_deg": round(direction, 1)}
               if direction is not None else {}),
            **({"speaker_slots_deg": slots} if slots else {}),
            **({"speaker_bearings": marks} if marks else {}),
            "t": round(t, 3),
            "rms_db": round(self.rms_db, 1),
            "peak_db": round(self.peak_db, 1),
            "floor_db": round(self.floor_db, 1),
            "gain_db": round(self.gain_db, 1),
            "effective_db": round(self.rms_db + self.gain_db, 1),
            "target_db": round(self.target_dbfs, 1),
            "speech": self.speech,
            "status": self.status(),
            "pitch_hz": round(self.pitch_hz, 1),
            "pitch_confidence": round(self.pitch_confidence, 3),
            "spectral_centroid_hz": round(self.spectral_centroid_hz, 1),
            "delivery_force": round(self.delivery_force, 3),
            "delivery_attack": round(self.delivery_attack, 3),
            "delivery_contour": round(self.delivery_contour, 3),
            "delivery_flow": round(self.delivery_flow, 3),
            "delivery_texture": round(self.delivery_texture, 3),
            "delivery_confidence": round(self.delivery_confidence, 3),
            "delivery_profile": self.delivery_profile,
        }


@dataclass(frozen=True)
class HypothesisWord:
    """A display word reconstructed from one or more recognizer tokens.

    ``pieces`` keeps the sub-word tokens the word was assembled from, as
    ``(text, start)`` pairs, for CWI 2.2.4 syllable variation. The transducer
    quantizes timestamps to its encoder frame, so short words emit every piece
    on one frame and carry no usable internal timing; drawn-out words — the
    only ones 2.2.4 applies to — do get distinct onsets.
    """

    text: str
    start: float
    end: float
    conf: float
    conf_available: bool = True
    pieces: tuple[tuple[str, float], ...] = ()

    def syllables(self) -> list[tuple[str, float]]:
        """Merge pieces sharing an encoder frame into distinct-onset groups."""

        groups: list[tuple[str, float]] = []
        for text, start in self.pieces:
            if groups and abs(start - groups[-1][1]) < 1e-6:
                groups[-1] = (groups[-1][0] + text, groups[-1][1])
            else:
                groups.append((text, start))
        return groups


def _word_delivery_features(
    word: HypothesisWord,
    audio: np.ndarray,
    cfg: dict,
) -> dict:
    """Measure one word's force, contour, flow, and acoustic texture.

    The descriptors are deliberately language-independent and interpretable.
    They describe what is present in the waveform; they do not claim to know
    whether the speaker feels angry, happy, sarcastic, etc. Callers freeze this
    result on first paint so incomplete/final ASR revisions cannot cause a
    second motion later.
    """

    delivery_cfg = dict(cfg.get("live", {}).get("delivery", {}) or {})
    neutral = {
        "delivery_force": 0.5,
        "delivery_attack": 0.0,
        "delivery_contour": 0.0,
        "delivery_contour_confidence": 0.0,
        "delivery_flow": 0.0,
        "delivery_texture": 0.0,
        "delivery_confidence": 0.0,
        "delivery_profile": "steady",
    }
    if not delivery_cfg.get("enabled", True) or not len(audio):
        return neutral

    i0 = max(0, int(word.start * SR))
    i1 = min(len(audio), max(i0 + 1, int(word.end * SR)))
    if i1 <= i0:
        return neutral
    duration_s = (i1 - i0) / SR
    context = int(float(delivery_cfg.get("context_s", 0.064)) * SR)
    frame_n = max(256, int(float(delivery_cfg.get("frame_s", 0.064)) * SR))
    hop_n = max(128, int(float(delivery_cfg.get("hop_s", 0.032)) * SR))
    a0, a1 = max(0, i0 - context), min(len(audio), i1 + context)
    analysis = np.asarray(audio[a0:a1], dtype=np.float32)
    if len(analysis) < 256:
        return neutral

    centers = list(range(frame_n // 2, len(analysis), hop_n))
    if not centers:
        centers = [len(analysis) // 2]
    pitch_floor = float(cfg.get("prosody", {}).get("pitch_floor_hz", 75.0))
    pitch_ceiling = float(cfg.get("prosody", {}).get("pitch_ceiling_hz", 500.0))
    confidence_floor = float(delivery_cfg.get("pitch_confidence_min", 0.28))
    brightness_range = delivery_cfg.get("texture_brightness_hz", [700, 4200])
    brightness_lo, brightness_hi = map(float, brightness_range)

    levels: list[float] = []
    pitches: list[float] = []
    periodicities: list[float] = []
    brightnesses: list[float] = []
    relative_centers: list[int] = []
    for center in centers:
        left = max(0, center - frame_n // 2)
        right = min(len(analysis), left + frame_n)
        frame = analysis[left:right]
        if len(frame) < 256:
            continue
        pitch, periodicity, centroid = _realtime_voice_features(
            frame, pitch_floor, pitch_ceiling
        )
        levels.append(float(_rms_db(frame)))
        pitches.append(float(pitch) if periodicity >= confidence_floor else 0.0)
        periodicities.append(float(periodicity))
        brightnesses.append(float(np.clip(
            (centroid - brightness_lo) / max(1.0, brightness_hi - brightness_lo),
            0.0,
            1.0,
        )))
        relative_centers.append(a0 + center)
    if not levels:
        return neutral

    inside = [
        index for index, center in enumerate(relative_centers)
        if i0 <= center <= i1
    ]
    if not inside:
        nearest = min(
            range(len(relative_centers)),
            key=lambda index: abs(relative_centers[index] - (i0 + i1) / 2),
        )
        inside = [nearest]

    word_levels = [levels[index] for index in inside]
    word_periodicities = [periodicities[index] for index in inside]
    word_brightnesses = [brightnesses[index] for index in inside]
    voiced = [
        pitches[index] for index in inside
        if pitches[index] > 0
    ]
    word_db = float(_rms_db(np.asarray(audio[i0:i1], dtype=np.float32)))
    # Absolute force stays anchored to the same true-capture range as live
    # loudness. It does not inherit a late-changing per-speaker percentile.
    db_lo, db_hi = map(float, cfg.get("live", {}).get("db_range", [-55, -18]))
    force = float(np.clip((word_db - db_lo) / max(1.0, db_hi - db_lo), 0.0, 1.0))

    pre = np.asarray(audio[max(0, i0 - context):i0], dtype=np.float32)
    pre_db = float(_rms_db(pre)) if len(pre) >= 128 else min(word_levels)
    early_count = max(1, len(word_levels) // 3)
    early_db = float(np.max(word_levels[:early_count]))
    attack_scale = float(delivery_cfg.get("attack_full_scale_db", 12.0))
    attack = float(np.clip((early_db - pre_db) / max(1.0, attack_scale), 0.0, 1.0))

    # The 64 ms orb estimator is intentionally immediate, but it is too sparse
    # for durable word contour: two adjacent frames frequently octave-jumped
    # and saturated ordinary Korean/English words at +/-1. Use the same robust
    # Praat tracker as durable prosody, discard octave outliers, and require
    # enough voiced evidence before contour is allowed to leave zero.
    contour = 0.0
    contour_confidence = 0.0
    contour_voiced: list[float] = []
    contour_frame_count = 0
    try:
        import parselmouth

        snd = parselmouth.Sound(
            np.asarray(audio[i0:i1], dtype=np.float64),
            sampling_frequency=SR,
        )
        pitch = snd.to_pitch_cc(
            time_step=0.01,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )
        pitch_values = np.asarray(
            pitch.selected_array["frequency"], dtype=np.float64
        )
        contour_frame_count = len(pitch_values)
        raw_voiced = pitch_values[pitch_values > 0]
        if len(raw_voiced):
            median_pitch = float(np.median(raw_voiced))
            octave_ratio = float(delivery_cfg.get(
                "contour_octave_filter_ratio", 1.60
            ))
            contour_voiced = [
                float(value) for value in raw_voiced
                if median_pitch / octave_ratio
                <= value
                <= median_pitch * octave_ratio
            ]
        min_frames = int(delivery_cfg.get("contour_min_voiced_frames", 5))
        voiced_fraction = len(contour_voiced) / max(contour_frame_count, 1)
        min_fraction = float(delivery_cfg.get(
            "contour_min_voiced_fraction", 0.30
        ))
        if (
            len(contour_voiced) >= min_frames
            and voiced_fraction >= min_fraction
        ):
            edge = max(2, len(contour_voiced) // 3)
            first_pitch = float(np.median(contour_voiced[:edge]))
            last_pitch = float(np.median(contour_voiced[-edge:]))
            semitones = 12.0 * math.log2(
                max(last_pitch, 1.0) / max(first_pitch, 1.0)
            )
            contour = float(np.clip(
                semitones / float(delivery_cfg.get(
                    "contour_full_scale_semitones", 7.0
                )),
                -1.0,
                1.0,
            ))
            contour_confidence = float(np.clip(
                min(1.0, len(contour_voiced) / max(2 * min_frames, 1))
                * min(1.0, voiced_fraction / max(min_fraction, 1e-6)),
                0.0,
                1.0,
            ))
    except (parselmouth.PraatError, ValueError):
        pass

    periodicity = float(np.median(word_periodicities))
    flow_voiced = contour_voiced or voiced
    if len(flow_voiced) >= 2:
        log_pitch = np.log2(np.asarray(flow_voiced))
        jitter = float(np.median(np.abs(np.diff(log_pitch))))
        continuity = float(np.clip(1.0 - jitter / 0.12, 0.0, 1.0))
    else:
        continuity = 0.0
    voiced_ratio = (
        len(flow_voiced) / max(1, contour_frame_count)
        if contour_frame_count
        else len(flow_voiced) / max(1, len(inside))
    )
    voiced_ratio = float(np.clip(voiced_ratio, 0.0, 1.0))
    flow = float(np.clip(
        0.55 * voiced_ratio + 0.30 * continuity + 0.15 * periodicity,
        0.0,
        1.0,
    ))
    brightness = float(np.median(word_brightnesses))
    texture = float(np.clip(
        0.70 * (1.0 - periodicity) + 0.30 * brightness,
        0.0,
        1.0,
    ))
    confidence = float(np.clip(
        (duration_s / 0.24) * (0.40 + 0.60 * max(voiced_ratio, periodicity)),
        0.0,
        1.0,
    ))
    profile_name = _delivery_profile(
        force=force,
        attack=attack,
        contour=contour,
        flow=flow,
        texture=texture,
        confidence=confidence,
        duration_s=duration_s,
        cfg=delivery_cfg,
    )
    return {
        "delivery_force": round(force, 4),
        "delivery_attack": round(attack, 4),
        "delivery_contour": round(contour, 4),
        "delivery_contour_confidence": round(contour_confidence, 4),
        "delivery_flow": round(flow, 4),
        "delivery_texture": round(texture, 4),
        "delivery_confidence": round(confidence, 4),
        "delivery_profile": profile_name,
    }


_SCRIPT_RANGES = (
    # Hangul syllables, jamo, compatibility jamo -- the same blocks the
    # frontend's `isWideChar` treats as Korean.
    ("hangul", 0xAC00, 0xD7A3),
    ("hangul", 0x1100, 0x11FF),
    ("hangul", 0x3130, 0x318F),
    ("latin", 0x0041, 0x005A),
    ("latin", 0x0061, 0x007A),
    # Latin-1/Extended letters (café, naïve) are still Latin script.
    ("latin", 0x00C0, 0x024F),
)


def _letter_script(ch: str) -> str | None:
    """'hangul', 'latin', another script name, or None for a non-letter."""

    if not ch.isalpha():
        return None
    o = ord(ch)
    for name, lo, hi in _SCRIPT_RANGES:
        if lo <= o <= hi:
            return name
    return "other"


def word_script_allowed(text: str, allowed: frozenset[str]) -> bool:
    """Is this word's script one the session is allowed to show?

    THE BILINGUAL LANGUAGE ID CAN PICK ANY OF 40 LOCALES, and live resets the
    stream at every endpoint, so every utterance gets its own decision. A wrong
    one decodes Korean speech as Japanese or Chinese and the stage shows Kana
    or Han ideographs -- measured on FLEURS through the `multi` profile, ~1% of
    short segments leak (`おいり 사회의 본질`, `استهلك a dinosau`), always as a
    PREFIX: the ID commits early on little evidence, emits a token or two, then
    self-corrects. Those tokens are noise, not missed content, which is why the
    right response is to drop the word rather than re-decode anything.

    Majority-of-letters, so a real word with one stray glyph survives; a word
    with no letters at all (numbers, punctuation) is script-neutral and always
    passes. With `allowed` empty the gate is off -- that is every profile
    except `multi`, and the pinned profiles cannot leak in the first place.
    """

    if not allowed:
        return True
    scripts = [s for s in map(_letter_script, text) if s is not None]
    if not scripts:
        return True
    outside = sum(1 for s in scripts if s not in allowed)
    return outside * 2 <= len(scripts)


def hypothesis_words(result, audio_duration: float) -> list[HypothesisWord]:
    """Collapse sherpa token pieces into timestamped display words.

    The English and Korean transducers use leading spaces as word-boundary
    markers, e.g. ``[" THE", " YE", "LL", "OW"]`` and
    ``[" 걔는", " 괜찮은", " 척", "하", "려", "구"]``. Some revisions
    contain a standalone space token, so a boundary is kept even when that
    token has no letters.
    """

    tokens = list(getattr(result, "tokens", []) or [])
    stamps = list(getattr(result, "timestamps", []) or [])
    probs = list(getattr(result, "ys_probs", []) or [])
    if not tokens or len(stamps) != len(tokens):
        text = str(getattr(result, "text", "") or "").strip()
        parts = text.split()
        if not parts:
            return []
        span = max(audio_duration, 0.08 * len(parts)) / len(parts)
        return [
            HypothesisWord(p, i * span, max((i + 1) * span, i * span + 0.02), 0.5, False)
            for i, p in enumerate(parts)
        ]

    grouped: list[tuple[str, float, list[float], list[tuple[str, float]]]] = []
    text = ""
    start = 0.0
    word_probs: list[float] = []
    word_pieces: list[tuple[str, float]] = []
    pending_boundary = False

    def flush() -> None:
        nonlocal text, word_probs, word_pieces, pending_boundary
        clean = text.strip()
        if clean:
            grouped.append((clean, start, word_probs, word_pieces))
        text, word_probs, word_pieces, pending_boundary = "", [], [], False

    for i, (token, stamp) in enumerate(zip(tokens, stamps)):
        boundary = bool(token[:1].isspace())
        if boundary and text.strip():
            flush()
        if boundary:
            start = float(stamp)
            pending_boundary = True
        elif not text and not pending_boundary:
            start = float(stamp)
        piece = token.lstrip() if boundary else token
        text += piece
        if piece:
            word_pieces.append((piece, float(stamp)))
        if text:
            pending_boundary = False
        if i < len(probs):
            word_probs.append(float(probs[i]))
    flush()

    words: list[HypothesisWord] = []
    for i, (word, word_start, logps, pieces) in enumerate(grouped):
        if i + 1 < len(grouped):
            word_end = grouped[i + 1][1]
        else:
            # A trailing endpoint contains silence. Limit the last word span so
            # that its loudness is not diluted by that silence.
            word_end = min(audio_duration, word_start + 0.6)
        word_end = max(word_start + 0.02, word_end)
        conf = math.exp(sum(logps) / len(logps)) if logps else 0.5
        words.append(HypothesisWord(word, word_start, word_end,
                                    float(np.clip(conf, 0.0, 1.0)), bool(logps),
                                    tuple(pieces)))
    return words


def common_prefix_len(a: list[HypothesisWord], b: list[HypothesisWord]) -> int:
    """Return the case-insensitive common word prefix of two hypotheses."""

    n = 0
    for left, right in zip(a, b):
        if left.text.casefold() != right.text.casefold():
            break
        n += 1
    return n


def _normalized_token(text: str) -> str:
    # Unicode-aware normalization is essential for Korean verifier alignment.
    # The former Latin-only regex collapsed every Hangul eojeol to "", making
    # SequenceMatcher treat unrelated Korean words as equal and preventing the
    # offline recognizer from correcting the streaming spelling.
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", text).casefold()
        if char.isalnum() or char == "'"
    )


def _trailing_speech_span(
    audio: np.ndarray,
    after_s: float,
) -> tuple[float, float] | None:
    """Find the voiced part of a verifier-only tail.

    Endpoint verification can recover words the streaming decoder omitted.
    Those words have no streaming timestamps.  Squeezing them into synthetic
    20 ms slots at the previous word boundary samples silence for prosody and
    speaker embeddings—the exact failure that made the final reply in the
    bundled sample look motionless and speaker-unknown.

    The verifier has already established that words exist in this tail, so a
    conservative energy search is enough to recover their acoustic region.  It
    is deliberately local to the unmatched tail and does not replace the
    normal streaming word clock.
    """

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration = len(samples) / SR
    start = float(np.clip(after_s, 0.0, duration))
    tail = samples[int(start * SR):]
    frame = round(0.02 * SR)
    if len(tail) < frame:
        return None
    levels = []
    for offset in range(0, len(tail) - frame + 1, frame):
        levels.append(_rms_db(tail[offset:offset + frame]))
    if not levels:
        return None
    levels = np.asarray(levels, dtype=np.float32)
    noise_floor = float(np.percentile(levels, 20))
    threshold = float(np.clip(noise_floor + 10.0, -52.0, -32.0))
    active = np.flatnonzero(levels > threshold)
    if len(active) < 2:
        return None

    # Keep the first substantial activity island. Endpoint audio can contain a
    # later door/music transient; spanning from the first to the last active
    # frame would stretch three words across several seconds and mix speakers.
    max_gap_frames = round(0.28 * SR / frame)
    clusters: list[np.ndarray] = []
    cluster_start = 0
    for index in range(1, len(active)):
        if int(active[index] - active[index - 1]) > max_gap_frames:
            clusters.append(active[cluster_start:index])
            cluster_start = index
    clusters.append(active[cluster_start:])
    substantial = [
        cluster for cluster in clusters
        if len(cluster) >= 5
        and (cluster[-1] - cluster[0] + 1) * frame / SR >= 0.18
    ]
    chosen = substantial[0] if substantial else max(clusters, key=len)

    onset = max(start, start + float(chosen[0]) * frame / SR - 0.04)
    offset = min(
        duration,
        start + float(chosen[-1] + 1) * frame / SR + 0.06,
    )
    return (onset, offset) if offset - onset >= 0.04 else None


def conservative_verified_words(
    streaming: list[HypothesisWord],
    verified_text: str,
    audio: np.ndarray | None = None,
) -> list[HypothesisWord]:
    """Align authoritative endpoint text onto the streaming word clock.

    Equal words and one-for-one corrections retain their original timings.
    Unequal replacement spans are divided across the verified words, pure
    internal insertions use the inter-word gap, trailing insertions use the
    actual active-speech tail when audio is available, and deleted streaming
    words disappear. This makes the durable transcript verifier-accurate while
    retaining real acoustic timing wherever the two recognizers agree. Close
    dialect spelling variants such as British ``dishonoured`` vs US
    ``dishonored`` stay as the streaming speaker produced them.
    """

    verified = verified_text.split()
    if not streaming or not verified:
        return streaming
    left = [_normalized_token(word.text) for word in streaming]
    right = [_normalized_token(word) for word in verified]
    output: list[HypothesisWord] = []
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)

    def corrected(source: HypothesisWord, token: str, token_norm: str) -> HypothesisWord:
        source_norm = _normalized_token(source.text)
        if source_norm != token_norm and not token_norm.startswith(source_norm):
            similarity = difflib.SequenceMatcher(
                a=source_norm, b=token_norm, autojunk=False
            ).ratio()
            if similarity >= 0.88:
                token = source.text
        # Sub-word onsets only describe the spelling they were decoded from, so
        # they survive a verifier pass only when the text is unchanged.
        pieces = source.pieces if token == source.text else ()
        return HypothesisWord(
            token, source.start, source.end, source.conf,
            source.conf_available, pieces,
        )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        source_count = i2 - i1
        verified_count = j2 - j1
        if tag == "delete":
            continue
        if source_count == verified_count:
            output.extend(
                corrected(streaming[source_i], verified[verified_i], right[verified_i])
                for source_i, verified_i in zip(range(i1, i2), range(j1, j2))
            )
            continue

        # A non-1:1 verifier correction has no native token timestamps. Anchor
        # it to the corresponding streaming span (or insertion gap) and divide
        # that small interval monotonically. These inferred slots are only
        # created once, at the phrase endpoint.
        if source_count:
            span_start = streaming[i1].start
            span_end = streaming[i2 - 1].end
            conf = float(np.mean([word.conf for word in streaming[i1:i2]]))
            conf_available = all(word.conf_available for word in streaming[i1:i2])
        else:
            previous_end = streaming[i1 - 1].end if i1 else 0.0
            if i1 < len(streaming):
                span_start, span_end = previous_end, streaming[i1].start
            else:
                active_tail = (
                    _trailing_speech_span(audio, previous_end)
                    if audio is not None else None
                )
                span_start, span_end = active_tail or (
                    previous_end,
                    len(audio) / SR if audio is not None else previous_end,
                )
            if span_end - span_start < 0.02 * verified_count:
                span_end = span_start + 0.02 * verified_count
            neighbors = streaming[max(0, i1 - 1):min(len(streaming), i1 + 1)]
            conf = float(np.mean([word.conf for word in neighbors])) if neighbors else 0.5
            conf_available = False
        span_end = max(span_end, span_start + 0.02 * verified_count)
        step = (span_end - span_start) / max(verified_count, 1)
        for offset, verified_i in enumerate(range(j1, j2)):
            output.append(HypothesisWord(
                verified[verified_i],
                span_start + offset * step,
                span_start + (offset + 1) * step,
                conf,
                conf_available,
            ))
    return output


def repair_verified_tail_timing(
    words: list[HypothesisWord],
    audio: np.ndarray,
) -> list[HypothesisWord]:
    """Move a verifier-confirmed trailing sentence onto its real speech.

    A streaming transducer can emit the right trailing words with timestamps
    collapsed onto the preceding silence.  In that case SequenceMatcher sees
    equal text, so insertion-only repair cannot help.  When a sentence after
    terminal punctuation is quiet at its claimed slots but clearly active
    speech continues later, redistribute only that final sentence over the
    detected speech.  This restores the evidence needed by both prosody and
    diarization without retiming earlier words the viewer already saw.
    """

    if len(words) < 2:
        return words
    terminal = re.compile(r"""[.?!]["')\]]*$""")
    suffix_start = None
    for index in range(len(words) - 2, -1, -1):
        if terminal.search(words[index].text):
            suffix_start = index + 1
            break
    if suffix_start is None or suffix_start >= len(words):
        return words

    old_start = float(words[suffix_start].start)
    old_end = float(words[-1].end)
    active = _trailing_speech_span(audio, old_start)
    if active is None or active[1] <= old_end + 0.18:
        return words

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    claimed = samples[
        max(0, int(old_start * SR)):min(len(samples), int(old_end * SR))
    ]
    detected = samples[
        max(0, int(active[0] * SR)):min(len(samples), int(active[1] * SR))
    ]
    if not len(claimed) or not len(detected):
        return words
    # Do not move genuinely quiet but correctly timed speech merely because a
    # later noise exists.  The later region must contain substantially stronger
    # acoustic evidence than the claimed slots.
    if _rms_db(detected) < _rms_db(claimed) + 6.0:
        return words

    repaired = list(words[:suffix_start])
    suffix = words[suffix_start:]
    step = (active[1] - active[0]) / len(suffix)
    for index, word in enumerate(suffix):
        repaired.append(HypothesisWord(
            word.text,
            active[0] + index * step,
            active[0] + (index + 1) * step,
            word.conf,
            word.conf_available,
            (),
        ))
    return repaired


class EndpointVerifier:
    """Fast local whole-phrase recognizer used only for durable text."""

    def __init__(self, recognizer, tail_padding_s: float = 0.0):
        self.recognizer = recognizer
        self.tail_padding_s = max(0.0, float(tail_padding_s))

    def transcribe(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        if self.tail_padding_s:
            audio = np.concatenate((
                audio,
                np.zeros(round(self.tail_padding_s * SR), dtype=np.float32),
            ))
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SR, audio)
        self.recognizer.decode_stream(stream)
        result = stream.result
        text = str(result.text or "").strip()
        # sherpa's generic CJK formatter removes spaces from Korean `text`,
        # even though this model's SentencePiece tokens retain exact eojeol
        # boundaries as leading ASCII spaces. Reconstruct only when the
        # formatted result lost all internal whitespace; English and other
        # normally formatted models keep their native result unchanged.
        tokens = [str(token) for token in (result.tokens or [])]
        token_text = "".join(tokens).strip()
        if text and " " not in text and " " in token_text:
            return token_text
        return text


class AdaptiveSpeechGate:
    """Track speech duration against the same adaptive noise-floor rule."""

    def __init__(self, live_cfg: dict):
        from collections import deque

        self.fixed_db = live_cfg["silence_db"]
        self.adaptive = live_cfg.get("adaptive_silence", True)
        self.levels = deque(maxlen=int(8 * SR / BLOCK))
        self.speech_s = 0.0

    def threshold(self) -> float:
        if not self.adaptive or len(self.levels) < 30:
            return self.fixed_db
        floor = float(np.percentile(self.levels, 20))
        return float(np.clip(floor + 12.0, -55.0, -30.0))

    def accept(self, block: np.ndarray) -> None:
        level = _rms_db(block)
        self.levels.append(level)
        if level > self.threshold():
            self.speech_s += len(block) / SR

    def reset_utterance(self) -> None:
        self.speech_s = 0.0


# ---------------------------------------------------------------------------
# Energy-based utterance segmentation
# ---------------------------------------------------------------------------

def utterances(blocks, live_cfg: dict):
    """Yield (audio, stream_t0) chunks split on pauses.

    The silence threshold adapts to the noise floor (20th percentile of recent
    block levels + 12 dB) so continuous backgrounds — movie scores, fan noise —
    still allow cuts at dialogue dips instead of always hitting the force-cut.
    """
    from collections import deque

    fixed_db = live_cfg["silence_db"]
    adaptive = live_cfg.get("adaptive_silence", True)
    min_silence = live_cfg["min_silence_s"]
    min_speech = live_cfg["min_speech_s"]
    max_utt = live_cfg["max_utterance_s"]
    preroll_blocks = int(0.3 * SR / BLOCK) + 1
    levels: deque[float] = deque(maxlen=int(8 * SR / BLOCK))  # ~8 s of history

    def threshold() -> float:
        if not adaptive or len(levels) < 30:
            return fixed_db
        floor = float(np.percentile(levels, 20))
        return float(np.clip(floor + 12.0, -55.0, -30.0))

    preroll: list[np.ndarray] = []
    buf: list[np.ndarray] = []
    in_speech = False
    silence_s = 0.0
    t_stream = 0.0
    utt_t0 = 0.0

    def flush():
        nonlocal buf, in_speech, silence_s
        audio = np.concatenate(buf) if buf else np.zeros(0, dtype=np.float32)
        speech_s = len(audio) / SR - silence_s
        buf, in_speech, silence_s = [], False, 0.0
        if speech_s >= min_speech:
            return audio, utt_t0
        return None

    def energy_blocks():
        for item in blocks:
            if isinstance(item, AudioChunk) and len(item.samples) > BLOCK:
                for offset in range(0, len(item.samples), BLOCK):
                    yield AudioChunk(
                        item.samples[offset:offset + BLOCK],
                        item.source_start + offset / SR,
                        item.discontinuity and offset == 0,
                        item.dropped_s if offset == 0 else 0.0,
                    )
            else:
                yield item

    for item in energy_blocks():
        if isinstance(item, AudioChunk):
            if item.discontinuity:
                # Never decode across a real capture-device discontinuity.
                preroll, buf, in_speech, silence_s = [], [], False, 0.0
                t_stream = item.source_start
            else:
                t_stream = max(t_stream, item.source_start)
            blk = item.samples
        else:
            blk = item
        level = _rms_db(blk)
        levels.append(level)
        silence_db = threshold()
        block_s = len(blk) / SR
        t_stream += block_s
        if not in_speech:
            preroll.append(blk)
            if len(preroll) > preroll_blocks:
                preroll.pop(0)
            if level > silence_db:
                in_speech = True
                buf = list(preroll)
                preroll = []
                utt_t0 = t_stream - len(buf) * BLOCK / SR
                silence_s = 0.0
        else:
            buf.append(blk)
            silence_s = 0.0 if level > silence_db else silence_s + block_s
            dur = len(buf) * BLOCK / SR
            if silence_s >= min_silence or dur >= max_utt:
                out = flush()
                if out:
                    yield out
    if in_speech:
        out = flush()
        if out:
            yield out


# ---------------------------------------------------------------------------
# Utterance -> word events (ASR + prosody)
# ---------------------------------------------------------------------------

def word_events(utts, model, lang: str, cfg: dict, speaker: str = "S1"):
    from collections import deque

    prosody_cfg = cfg["prosody"]
    cfg_lo, cfg_hi = cfg["live"]["db_range"]
    # Self-calibrating loudness: the running median of recent word levels is
    # the CWI baseline (5% type size); +15 dB reads as a shout (12%), -5 dB as
    # a whisper (3%). This adapts to any mic gain without configuration.
    hist: deque[float] = deque(maxlen=120)

    import parselmouth

    for audio, t0 in utts:
        t_infer = time.monotonic()
        segments, _ = model.transcribe(
            audio, language=lang, word_timestamps=True, beam_size=1,
            temperature=0.0, condition_on_previous_text=False,
            vad_filter=True,  # drop music/noise-only stretches inside the chunk
        )
        # Skip hallucination-prone segments (music beds make whisper invent
        # phrases like "thanks for watching" with high no-speech probability).
        words = [
            w
            for seg in segments
            if seg.no_speech_prob < 0.6 or seg.avg_logprob > -1.0
            for w in (seg.words or [])
            if w.word.strip() and w.probability >= 0.15
        ]
        if not words:
            continue

        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=SR)
        try:
            pitch = snd.to_pitch_cc(pitch_floor=prosody_cfg["pitch_floor_hz"],
                                    pitch_ceiling=prosody_cfg["pitch_ceiling_hz"])
            f0_t, f0_v = pitch.xs(), pitch.selected_array["frequency"]
        except parselmouth.PraatError:  # utterance too short for the analysis window
            f0_t, f0_v = np.zeros(0), np.zeros(0)

        latency = time.monotonic() - t_infer
        texts = []
        for w in words:
            i0, i1 = int(w.start * SR), min(int(w.end * SR), len(audio))
            span = audio[i0:i1]
            db = _span_db(span) if len(span) else -80.0
            hist.append(db)
            if len(hist) >= 6:
                med = float(np.median(hist))
                lo_db, hi_db = med - 5.0, med + 15.0
            else:
                lo_db, hi_db = cfg_lo, cfg_hi
            mask = (f0_t >= w.start) & (f0_t <= w.end)
            voiced = f0_v[mask][f0_v[mask] > 0]
            pitch_hz = float(np.median(voiced)) if len(voiced) else 0.0
            voiced_frac = float(len(voiced) / max(mask.sum(), 1))
            yield {
                "text": w.word.strip(),
                "t": round(t0 + w.start, 3),          # absolute stream onset
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "speaker": speaker,
                "loudness": round(loudness, 4),
                "pitch": 0.5,                          # renderer uses raw Hz (domain_hz)
                "loudness_db": round(db, 2),
                "pitch_hz": round(pitch_hz, 2),
                "voiced_frac": round(voiced_frac, 3),
                "conf": round(min(max(w.probability, 0.0), 1.0), 3),
            }
            texts.append(w.word.strip())
        print(f"[live] +{latency:.1f}s  {' '.join(texts)}")


# ---------------------------------------------------------------------------
# True-streaming transducer -> revisable hypotheses + committed word events
# ---------------------------------------------------------------------------

SpeakerStatus = Literal["unknown", "provisional", "stable", "corrected"]


class DirectionSpeakerMap:
    """Speakers from azimuth alone, the SpeechCompass way.

    Voice evidence is slow — text at a median 0.62 s, durable identity at
    4.48 s — so a new speaker renders grey for seconds. A bearing is available
    on the word itself, and the endpoint pass still corrects it, so the claim
    stays revisable.

    **This trades a different error, it does not remove one.** Two people at
    one bearing collapse into one slot; one person crossing the tolerance
    splits into two. The tolerance is far wider than the array's own 11-22 deg
    error because it must absorb that without merging people genuinely apart.

    A `None` bearing keeps the CURRENT slot rather than inventing an angle:
    absent direction is "no new evidence", and turn continuity is the ordinary
    diarization prior. Half of all reads carry no bearing.
"""

    def __init__(self, tolerance_deg: float = 35.0) -> None:
        self.tolerance_deg = float(tolerance_deg)
        self._slots: list[float] = []          # circular-mean centroid per slot
        self._counts: list[int] = []
        self._current: int | None = None

    @staticmethod
    def _separation(a: float, b: float) -> float:
        """Smallest angle between two bearings, 0..180."""
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def assign(self, bearing: float | None) -> int | None:
        """The slot index for this bearing, creating one if it is new."""
        if bearing is None:
            return self._current
        bearing %= 360.0
        if self._slots:
            nearest = min(range(len(self._slots)),
                          key=lambda i: self._separation(bearing, self._slots[i]))
            if self._separation(bearing, self._slots[nearest]) <= self.tolerance_deg:
                # Circular running mean: a plain average would drag a centroid
                # across 0/360 to the opposite side of the room.
                n = self._counts[nearest]
                centroid = self._slots[nearest]
                delta = (bearing - centroid + 180.0) % 360.0 - 180.0
                self._slots[nearest] = (centroid + delta / (n + 1)) % 360.0
                self._counts[nearest] = n + 1
                self._current = nearest
                return nearest
        self._slots.append(bearing)
        self._counts.append(1)
        self._current = len(self._slots) - 1
        return self._current

    @property
    def slots(self) -> list[float]:
        return list(self._slots)


class SpeakerBearingMap:
    """Where each DECIDED speaker has been heard from, for the compass ring.

    ``DirectionSpeakerMap`` cannot answer this: it stops being fed the moment
    attribution settles, which is almost immediately. Measured over 22 s, the
    voice lane labelled 343 words while the slot map held ONE centroid — a ring
    fed from it marks one position and goes deaf.

    So this is separate and purely descriptive. It never returns an identity,
    and direction-derived guesses are excluded by the caller: feeding a
    geometric identity back into a geometric position is circular.

    Recent-window for the MEAN, lifetime for the MARK. The position comes from
    the window so a speaker who moves is marked where they are now, but a
    speaker never LEAVES the ring once placed — too scattered to average
    re-emits the last good mark as ``stale`` instead. A viewer cannot tell
    "this speaker left" from "their bearings got noisy for a moment".
"""

    # A speaker's bearings must point more together than apart before their
    # mean is a place anyone stood. Same threshold and same reasoning as
    # ``NodeLink.MIN_BEARING_CONCENTRATION``: below it, the average is an
    # artefact of arithmetic, and the ring would mark somewhere nobody was.
    MIN_CONCENTRATION = 0.5

    # Enough to smooth the array's 11-22 deg localization error without
    # pinning a mark to where somebody was sitting several turns ago.
    WINDOW = 64

    def __init__(self, window: int = WINDOW) -> None:
        self.window = int(window)
        self._bearings: dict[str, deque[float]] = {}
        # The last mark each speaker had that was concentrated enough to mean
        # anything. This is what keeps them on the ring for the session.
        self._placed: dict[str, dict] = {}
        # A word is re-emitted under its own id on every revision, and a
        # correction can move it to another speaker. Counting each (word,
        # speaker) once keeps a single loud word from dragging a centroid.
        self._seen: set[tuple[str, str]] = set()

    def observe(self, word_id: str | None, speaker_id: str,
                bearing: float | None) -> None:
        if bearing is None or word_id is None:
            return
        key = (word_id, speaker_id)
        if key in self._seen:
            return
        self._seen.add(key)
        history = self._bearings.setdefault(
            speaker_id, deque(maxlen=self.window))
        history.append(float(bearing) % 360.0)

    @property
    def marks(self) -> list[dict]:
        """``[{speaker, deg, spread, observations}]``, quietest speaker last.

        Circular mean per speaker: bearings are angles, so 359 and 1 average
        to 0 rather than to 180 -- and a talker straight ahead of the case is
        exactly where the naive mean is worst.
        """
        marks = []
        for speaker_id, history in self._bearings.items():
            if not history:
                continue
            radians = [math.radians(d) for d in history]
            x = sum(math.cos(r) for r in radians)
            y = sum(math.sin(r) for r in radians)
            concentration = math.hypot(x, y) / len(radians)
            if concentration < self.MIN_CONCENTRATION:
                # Scattered: the mean is an artefact of arithmetic and must not
                # be published as a position. But the speaker stays on the ring
                # wearing the last place they were actually seen, flagged so
                # the renderer can show it as remembered rather than live.
                remembered = self._placed.get(speaker_id)
                if remembered is not None:
                    marks.append({**remembered, "stale": True})
                continue
            mark = {
                "speaker": speaker_id,
                # Round BEFORE the wrap. Two bearings either side of north
                # average to -2e-13 deg, which `% 360` turns into 359.9999 and
                # rounding then reports as 360 -- a mark behind the array for
                # a speaker standing in front of it.
                "deg": round(math.degrees(math.atan2(y, x)), 1) % 360.0,
                # HOW WIDELY THIS SPEAKER'S BEARINGS ARE SCATTERED, as the
                # circular standard deviation in degrees: sqrt(-2 ln R), the
                # textbook conversion from the concentration already computed
                # above. It was being thrown away, and it is the honest width
                # for the compass arc -- a speaker seen once from one angle and
                # one seen thirty times from the same angle are not equally
                # located, and a fixed-size mark claims they are.
                # Only ever a spread, never a confidence score: the renderer
                # decides how many sigma to draw.
                # `abs` is for the SIGN, not the size: a perfectly
                # concentrated speaker gives -2 * log(1) = -0.0, which
                # serialises as `-0.0` and reads as a negative angle.
                "spread": abs(round(
                    math.degrees(math.sqrt(abs(-2.0 * math.log(
                        min(1.0, max(1e-6, concentration)))))), 1)),
                "observations": len(history),
            }
            self._placed[speaker_id] = mark
            marks.append(mark)
        marks.sort(key=lambda m: m["speaker"])
        return marks

    def match(self, bearing: float | None, *, sigma: float = 2.0,
              min_observations: int = 4, floor_deg: float = 12.0
              ) -> tuple[str, float] | None:
        """Which PLACED speaker best explains this bearing, and how well.

        Returns ``(speaker_id, z)`` where ``z`` is the separation in units of
        that speaker's OWN spread, so a tightly-pinned talker is matched
        strictly. ``None`` when nothing is close enough must stay a real
        outcome: a bearing from where nobody has spoken is evidence of a NEW
        speaker, not of the nearest old one.

        This is direction as evidence rather than as a fallback.
        ``DirectionSpeakerMap`` can only say "the 2nd distinct direction"; this
        names a speaker the VOICE lane identified.

        **The circularity guard upstream is what makes it sound.**
        ``_record_speaker_bearing`` refuses to record a bearing under a
        direction-derived identity, so this never confirms its own guesses.
        Relax that guard and it becomes a self-reinforcing feedback loop.

        ``floor_deg`` stops a speaker seen from one exact angle having a
        zero-width band nothing can match.
        """
        if bearing is None:
            return None
        bearing %= 360.0
        best: tuple[str, float] | None = None
        for mark in self.marks:
            if int(mark.get("observations", 0)) < min_observations:
                continue
            width = max(float(mark.get("spread", 0.0)), floor_deg)
            z = self._separation(bearing, float(mark["deg"])) / width
            if z <= sigma and (best is None or z < best[1]):
                best = (str(mark["speaker"]), z)
        return best

    @staticmethod
    def _separation(a: float, b: float) -> float:
        """Smallest angle between two bearings, 0..180."""
        return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class SpeakerAttribution:
    """One revision-capable speaker decision returned by ``SpeakerTracker``."""

    speaker_id: str | None
    status: SpeakerStatus
    confidence: float
    speaker_change_probability: float | None
    revision_id: int
    reason: str | None = None
    observation_duration_s: float = 0.0
    best_similarity: float | None = None
    second_best_similarity: float | None = None
    confidence_margin: float | None = None
    centroid_updated: bool = False
    switch_decision: str | None = None

    def event_fields(self, include_debug: bool = False) -> dict:
        fields = {
            "speaker": self.speaker_id,
            "speaker_status": self.status,
            "speaker_confidence": round(float(np.clip(self.confidence, 0.0, 1.0)), 4),
            "speaker_change_probability": (
                None if self.speaker_change_probability is None
                else round(float(np.clip(self.speaker_change_probability, 0.0, 1.0)), 4)
            ),
            "speaker_revision_id": self.revision_id,
        }
        if include_debug:
            fields["speaker_debug"] = {
                "observation_duration_s": round(self.observation_duration_s, 4),
                "best_candidate_similarity": self.best_similarity,
                "second_best_similarity": self.second_best_similarity,
                "confidence_margin": self.confidence_margin,
                "centroid_updated": self.centroid_updated,
                "reason": self.reason,
                "switch_decision": self.switch_decision,
            }
        return fields


class PyannoteSpeakerActivity:
    """Fast local speaker-activity labels from pyannote segmentation 3.0.

    The model identifies up to three locally consistent speaker streams (and
    their overlaps) inside one endpoint phrase.  Those local stream numbers are
    deliberately *not* exposed as durable speaker identities: ``SpeakerTracker``
    uses them only to cut clean acoustic turns, then its embedding profiles map
    each turn onto stable S1/S2/... identities across endpoints.

    Feeding the endpoint phrase directly avoids sherpa's full-file diarization
    wrapper, which recomputes many overlapping speaker embeddings.  The int8
    segmentation pass is about 40 ms for ten seconds on the target Mac.
    """

    _POWERSET = (
        (),
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
    )

    def __init__(self, model: str | Path, num_threads: int = 2):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = max(1, int(num_threads))
        self.session = ort.InferenceSession(
            str(model),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.sample_rate = int(metadata.get("sample_rate", SR))
        self.frame_shift = int(metadata.get("receptive_field_shift", 270))
        self.receptive_field = int(metadata.get("receptive_field_size", 991))
        if self.sample_rate != SR:
            raise ValueError(
                f"speaker segmentation expects {self.sample_rate} Hz, not {SR} Hz"
            )
        classes = int(metadata.get("num_classes", len(self._POWERSET)))
        if classes != len(self._POWERSET):
            raise ValueError(
                f"unsupported speaker segmentation powerset ({classes} classes)"
            )

    def __call__(
        self,
        samples: np.ndarray,
        word_spans: list[tuple[float, float]],
    ) -> list[int | None]:
        if not word_spans or len(samples) < self.receptive_field:
            return [None] * len(word_spans)
        waveform = np.asarray(samples, dtype=np.float32).reshape(1, 1, -1)
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: waveform})[0][0]
        classes = np.argmax(logits, axis=-1)
        frame_times = (
            np.arange(len(classes), dtype=np.float32) * self.frame_shift
            + self.receptive_field * 0.5
        ) / self.sample_rate

        labels: list[int | None] = []
        for start, end in word_spans:
            # A small acoustic skirt makes very short ASR words less sensitive
            # to timestamp quantization while staying well inside a normal turn.
            lo = max(0.0, float(start) - 0.06)
            hi = min(len(samples) / SR, float(end) + 0.06)
            indices = np.flatnonzero((frame_times >= lo) & (frame_times <= hi))
            if not len(indices):
                midpoint = (float(start) + float(end)) * 0.5
                indices = np.array(
                    [int(np.argmin(np.abs(frame_times - midpoint)))],
                    dtype=np.int64,
                )

            votes = np.zeros(3, dtype=np.float32)
            for class_index in classes[indices]:
                for speaker in self._POWERSET[int(class_index)]:
                    votes[speaker] += 1.0
            order = np.argsort(-votes)
            best = int(order[0])
            second = float(votes[order[1]])
            # Ambiguous overlap frames should not invent a boundary. The next
            # clear singleton word supplies the change point instead.
            labels.append(
                best
                if votes[best] > 0 and votes[best] >= second * 1.15
                else None
            )

        return labels


class SpeakerTracker:
    """Confidence-aware online speaker attribution and profile enrollment.

    The tracker is the single owner of live speaker profiles and lifecycle
    state. ``observe`` accepts a timestamped embedding plus optional overlap,
    signal-quality, and direction metadata. Direction is deliberately only a
    weak prior; the current mono capture path does not manufacture one.

    ``classify_span`` is the non-learning mid-stream path. ``label_words`` is
    the endpoint segment-then-observe path and is the only audio entry point
    allowed to enroll or update profiles.
    """

    def __init__(
        self,
        embed,
        similarity: float | None = None,
        max_speakers: int = 6,
        window_s: float = 1.0,
        hop_s: float = 0.25,
        min_span_s: float | None = None,
        change_below: float = 0.3,
        merge_at: float = 0.5,
        *,
        min_enrollment_duration_s: float = 0.8,
        min_assignment_duration_s: float = 0.25,
        stable_after_observations: int = 2,
        stable_after_duration_s: float = 1.1,
        immediate_speaker_limit: int = 2,
        assignment_threshold: float | None = None,
        provisional_threshold: float | None = None,
        new_speaker_threshold: float | None = None,
        centroid_ema_alpha: float = 0.15,
        switch_hysteresis_s: float = 0.35,
        short_turn_max_duration_s: float = 0.4,
        retain_threshold: float = 0.64,
        switch_threshold: float | None = None,
        min_confidence_margin: float = 0.08,
        speaker_min_run_words: int = 2,
        short_stable_threshold: float | None = None,
        short_stable_min_margin: float = 0.12,
        short_stable_max_duration_s: float = 1.3,
        min_signal_quality: float = 0.25,
        direction_prior_weight: float = 0.05,
        speaker_activity=None,
        debug: bool = False,
    ):
        self.embed = embed
        self.speaker_activity = speaker_activity
        self.max_speakers = max(1, int(max_speakers))
        self.window_s = float(window_s)
        self.hop_s = float(hop_s)
        self.min_assignment_duration_s = float(
            min_assignment_duration_s if min_span_s is None else min_span_s
        )
        self.min_enrollment_duration_s = float(min_enrollment_duration_s)
        self.stable_after_observations = max(1, int(stable_after_observations))
        # ...or one observation this long, which is what lets a character with
        # a single line still get a colour. See `_profile_is_stable`.
        self.stable_after_duration_s = float(stable_after_duration_s)
        # S1/S2 stay genuinely live. A third-or-later identity is still
        # supported, but one isolated native slot or embedding must not create
        # a durable new speaker colour in an ordinary two-person conversation.
        self.immediate_speaker_limit = max(
            1,
            min(self.max_speakers, int(immediate_speaker_limit)),
        )

        # ``similarity`` is the pre-lifecycle raw-cosine cutoff. Keep direct
        # construction backward compatible while the new config supplies
        # calibrated 0..1 confidence thresholds explicitly.
        legacy_threshold = None if similarity is None else float(similarity)
        self.assignment_threshold = float(
            assignment_threshold if assignment_threshold is not None
            else (legacy_threshold if legacy_threshold is not None else 0.72)
        )
        self.provisional_threshold = float(
            provisional_threshold if provisional_threshold is not None
            else max(0.0, self.assignment_threshold - 0.14)
        )
        self.new_speaker_threshold = float(
            new_speaker_threshold if new_speaker_threshold is not None
            else (legacy_threshold if legacy_threshold is not None else 0.42)
        )
        self.switch_threshold = float(
            switch_threshold if switch_threshold is not None
            else self.assignment_threshold
        )
        self.retain_threshold = float(retain_threshold)
        self.centroid_ema_alpha = float(np.clip(centroid_ema_alpha, 0.0, 1.0))
        self.switch_hysteresis_s = max(0.0, float(switch_hysteresis_s))
        self.short_turn_max_duration_s = max(0.0, float(short_turn_max_duration_s))
        self.min_confidence_margin = max(0.0, float(min_confidence_margin))
        self.speaker_min_run_words = max(1, int(speaker_min_run_words))
        self.short_stable_threshold = float(
            short_stable_threshold
            if short_stable_threshold is not None
            else self.assignment_threshold
        )
        self.short_stable_min_margin = max(
            self.min_confidence_margin,
            float(short_stable_min_margin),
        )
        self.short_stable_max_duration_s = max(
            self.min_assignment_duration_s,
            float(short_stable_max_duration_s),
        )
        self.min_signal_quality = float(np.clip(min_signal_quality, 0.0, 1.0))
        # THE RATIO BETWEEN THE VOICE ALGORITHM AND THE DIRECTION SENSOR.
        # In `_score`, confidence = (1 - w) * embedding + w * direction, so w is
        # literally the direction sensor's SHARE of a match decision. It was
        # capped at 0.25 as "a weak prior only"; the cap is 0.90 now so a booth
        # in a hard room, where the embedding flickers on short spontaneous
        # turns but the array's bearing is steady, can dial the sensor up until
        # it dominates. Only bites when a bearing is actually present (the array
        # is attached); on a local mic `direction_estimate` is None and the
        # blend is skipped, so raising this changes nothing without the array.
        self.direction_prior_weight = float(np.clip(direction_prior_weight, 0.0, 0.90))
        self.change_below = float(change_below)
        self.merge_at = float(merge_at)  # retained config/API compatibility
        self.debug = bool(debug)

        self.centroids: list[np.ndarray] = []
        self.counts: list[int] = []
        self.enrolled_durations: list[float] = []
        self.profile_observation_groups: list[set[int]] = []
        # Longest SINGLE observation behind each profile. Total enrolled
        # duration cannot be used for the one-turn rule below: two clean
        # fragments of one endpoint sum to the same number as one real
        # turn, and they are not independent evidence.
        self.profile_longest_observation: list[float] = []
        self.profile_stable: list[bool] = []
        self.profile_directions: list[float | None] = []
        self.alias: dict[int, int] = {}

        self.last_confidently_active_speaker: str | None = None
        self.last_speaker_change_timestamp: float | None = None
        self.current_confidence = 0.0
        self.assignment_status: SpeakerStatus = "unknown"
        self.revision_id = 0
        self.revision_history: dict[str, SpeakerAttribution] = {}
        self.recent_direction_estimate: float | None = None

        self._pending_switch: int | None = None
        self._pending_switch_since: float | None = None
        self._queued_revisions: list[tuple[str, SpeakerAttribution]] = []
        self._pending_candidate_word_keys: dict[int, set[str]] = {}
        self._observation_sequence = 0
        self._observation_group_sequence = 0
        self._speaker_switches = 0
        self._corrections = 0
        self._unknown_assignments = 0

    def _canon(self, index: int) -> int:
        while index in self.alias:
            index = self.alias[index]
        return index

    def _active(self) -> list[int]:
        return [i for i in range(len(self.centroids)) if i not in self.alias]

    @staticmethod
    def _speaker_id(index: int) -> str:
        return f"S{index + 1}"

    @staticmethod
    def _normalized_embedding(embedding) -> np.ndarray | None:
        if embedding is None:
            return None
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if not len(vector) or not np.all(np.isfinite(vector)):
            return None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else None

    @staticmethod
    def _direction_similarity(a: float, b: float) -> float:
        distance = abs((a - b + 180.0) % 360.0 - 180.0)
        return 1.0 - distance / 180.0

    def _scores(self, embedding: np.ndarray,
                direction_estimate: float | None) -> list[tuple[int, float]]:
        scored = []
        for index in self._active():
            raw = float(np.dot(self.centroids[index], embedding))
            confidence = float(np.clip(raw, 0.0, 1.0))
            known_direction = self.profile_directions[index]
            if direction_estimate is not None and known_direction is not None:
                direction_score = self._direction_similarity(
                    float(direction_estimate), float(known_direction)
                )
                weight = self.direction_prior_weight
                confidence = (1.0 - weight) * confidence + weight * direction_score
            scored.append((index, confidence))
        return sorted(scored, key=lambda item: (-item[1], item[0]))

    def _profile_is_stable(self, index: int) -> bool:
        required_duration = (
            self.min_enrollment_duration_s * self.stable_after_observations
        )
        repeated = self.counts[index] >= self.stable_after_observations
        if index >= self.immediate_speaker_limit:
            # Several punctuation/acoustic segments from one long noisy turn
            # are not independent evidence. Higher identities need distinct
            # endpoint passes, not merely several fragments in one pass...
            if (
                len(self.profile_observation_groups[index])
                >= self.stable_after_observations
            ):
                return True
            # ...OR one observation long enough to stand on its own. A
            # CHARACTER WHO SPEAKS ONCE IS STILL A CHARACTER, and CWI 2.1 is
            # about telling them apart. Requiring repeats meant everyone after
            # the second speaker rendered permanently neutral: on the PR film,
            # which has eleven speakers and where most of them have a single
            # line, that was 79 words of "Attribution pending" and the drill
            # sergeant sharing the narrator's colour. The phantom this gate
            # guards against is a SHORT noisy embedding, so gate on length.
            return (
                self.profile_longest_observation[index]
                >= self.stable_after_duration_s
            )
        return repeated or self.enrolled_durations[index] >= required_duration

    @staticmethod
    def _speaker_index(speaker_id: str | None) -> int | None:
        if not speaker_id or not speaker_id.startswith("S"):
            return None
        try:
            return int(speaker_id[1:]) - 1
        except ValueError:
            return None

    def _public_attribution(
        self,
        result: SpeakerAttribution,
        observation_key: str | None,
    ) -> SpeakerAttribution:
        """Hide an additional identity until endpoint evidence repeats.

        The candidate profile remains available internally for the next clean
        match. The audience sees neutral attribution instead of a transient
        S3/S4/S5 colour, and the stored word is revised once the profile is
        confirmed.
        """

        index = self._speaker_index(result.speaker_id)
        if (
            index is None
            or index < self.immediate_speaker_limit
            or index >= len(self.profile_stable)
            or self.profile_stable[index]
        ):
            return result
        if observation_key is not None and not observation_key.startswith(
            "@observation:"
        ):
            self._pending_candidate_word_keys.setdefault(index, set()).add(
                observation_key
            )
        return replace(
            result,
            speaker_id=None,
            status="unknown",
            reason=(
                "additional speaker candidate awaiting repeated endpoint "
                "confirmation"
            ),
            centroid_updated=False,
            switch_decision="pending-additional-speaker-confirmation",
        )

    def _create_profile(
        self,
        embedding: np.ndarray,
        duration: float,
        direction_estimate: float | None,
        observation_group: int,
    ) -> int:
        index = len(self.centroids)
        self.centroids.append(embedding.copy())
        self.counts.append(1)
        self.enrolled_durations.append(duration)
        self.profile_observation_groups.append({observation_group})
        self.profile_longest_observation.append(float(duration))
        self.profile_stable.append(False)
        self.profile_directions.append(direction_estimate)
        self.profile_stable[index] = self._profile_is_stable(index)
        return index

    def _update_profile(
        self,
        index: int,
        embedding: np.ndarray,
        duration: float,
        direction_estimate: float | None,
        observation_group: int,
    ) -> bool:
        alpha = self.centroid_ema_alpha
        updated = (1.0 - alpha) * self.centroids[index] + alpha * embedding
        norm = float(np.linalg.norm(updated))
        if norm <= 1e-9:
            return False
        self.centroids[index] = (updated / norm).astype(np.float32)
        self.counts[index] += 1
        self.enrolled_durations[index] += duration
        self.profile_observation_groups[index].add(observation_group)
        self.profile_longest_observation[index] = max(
            self.profile_longest_observation[index], float(duration)
        )
        if direction_estimate is not None:
            previous = self.profile_directions[index]
            self.profile_directions[index] = (
                float(direction_estimate) if previous is None
                else (1.0 - alpha) * float(previous) + alpha * float(direction_estimate)
            )
        self.profile_stable[index] = self._profile_is_stable(index)
        return True

    def _base_result(
        self,
        speaker_id: str | None,
        status: SpeakerStatus,
        confidence: float,
        change_probability: float | None,
        reason: str,
        duration: float,
        best: float | None,
        second: float | None,
        margin: float | None,
        centroid_updated: bool,
        switch_decision: str | None,
    ) -> SpeakerAttribution:
        return SpeakerAttribution(
            speaker_id=speaker_id,
            status=status,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            speaker_change_probability=change_probability,
            revision_id=0,
            reason=reason,
            observation_duration_s=duration,
            best_similarity=None if best is None else round(best, 4),
            second_best_similarity=None if second is None else round(second, 4),
            confidence_margin=None if margin is None else round(margin, 4),
            centroid_updated=centroid_updated,
            switch_decision=switch_decision,
        )

    def _record(
        self,
        result: SpeakerAttribution,
        observation_key: str | None,
    ) -> SpeakerAttribution:
        key = observation_key
        if key is None:
            self._observation_sequence += 1
            key = f"@observation:{self._observation_sequence}"
        previous = self.revision_history.get(key)
        status = result.status
        if (
            previous is not None
            and result.status == "stable"
            and previous.speaker_id is not None
            and result.speaker_id != previous.speaker_id
        ):
            status = "corrected"
            self._corrections += 1
        changed = (
            previous is None
            or previous.speaker_id != result.speaker_id
            or previous.status != status
            or abs(previous.confidence - result.confidence) > 1e-6
        )
        if changed:
            self.revision_id += 1
        recorded = replace(
            result,
            status=status,
            revision_id=self.revision_id if changed else previous.revision_id,
        )
        self.revision_history[key] = recorded
        self.current_confidence = recorded.confidence
        self.assignment_status = recorded.status
        if recorded.status == "unknown":
            self._unknown_assignments += 1
        if self.debug:
            print("[speaker] " + json.dumps({
                "observation": key,
                **recorded.event_fields(include_debug=True),
            }, ensure_ascii=False))
        return recorded

    def _queue_stabilized_profile(self, speaker_id: str,
                                  except_key: str | None = None) -> None:
        for key, previous in list(self.revision_history.items()):
            if key == except_key or key.startswith("@observation:"):
                continue
            if previous.speaker_id != speaker_id or previous.status != "provisional":
                continue
            self.revision_id += 1
            revised = replace(
                previous,
                status="stable",
                revision_id=self.revision_id,
                reason="profile reached stable enrollment",
                centroid_updated=False,
            )
            self.revision_history[key] = revised
            self._queued_revisions.append((key, revised))
        index = self._speaker_index(speaker_id)
        if index is None:
            return
        for key in sorted(self._pending_candidate_word_keys.pop(index, set())):
            if key == except_key:
                continue
            previous = self.revision_history.get(key)
            if previous is None or previous.status != "unknown":
                continue
            self.revision_id += 1
            revised = replace(
                previous,
                speaker_id=speaker_id,
                status="stable",
                revision_id=self.revision_id,
                reason="additional speaker reached repeated endpoint confirmation",
                centroid_updated=False,
                switch_decision="confirmed-additional-speaker",
            )
            self.revision_history[key] = revised
            self._queued_revisions.append((key, revised))

    def drain_revisions(self) -> list[tuple[str, SpeakerAttribution]]:
        revisions = self._queued_revisions
        self._queued_revisions = []
        return revisions

    def metrics(self) -> dict[str, int]:
        return {
            "speaker_id_switches": self._speaker_switches,
            "corrections": self._corrections,
            "unknown_assignments": self._unknown_assignments,
            "profiles": len(self._active()),
        }

    def observe(
        self,
        embedding,
        start_s: float,
        end_s: float,
        *,
        update: bool = True,
        overlap: bool = False,
        signal_quality: float | None = None,
        direction_estimate: float | None = None,
        observation_key: str | None = None,
        observation_group: int | None = None,
    ) -> SpeakerAttribution:
        """Attribute one timestamped embedding and optionally update profiles."""

        duration = max(0.0, float(end_s) - float(start_s))
        if observation_group is None:
            self._observation_group_sequence += 1
            observation_group = self._observation_group_sequence
        if direction_estimate is not None and math.isfinite(float(direction_estimate)):
            self.recent_direction_estimate = float(direction_estimate) % 360.0
        else:
            direction_estimate = None

        unsuitable_reason = None
        if overlap:
            unsuitable_reason = "overlap-marked observation"
        elif signal_quality is not None and signal_quality < self.min_signal_quality:
            unsuitable_reason = "signal quality below update threshold"
        vector = self._normalized_embedding(embedding)
        if vector is None:
            return self._record(self._base_result(
                None, "unknown", 0.0, None, "invalid or missing embedding",
                duration, None, None, None, False, None,
            ), observation_key)

        scored = self._scores(vector, direction_estimate)
        best_index = scored[0][0] if scored else None
        best = scored[0][1] if scored else None
        second = scored[1][1] if len(scored) > 1 else None
        margin = None if best is None else best - (second if second is not None else 0.0)
        # AMBIGUOUS BETWEEN WHOM? The margin guard exists so a word is not
        # handed to the wrong enrolled speaker when two profiles score alike.
        # It is meaningless when NOBODY is a plausible match: measured on the
        # PR film, four separate new voices arrived with best/second of
        # 0.016/0.000, 0.084/0.065, 0.120/0.048 and 0.167/0.161 -- the
        # strongest possible evidence for someone new -- and every one of them
        # was called ambiguous, which cleared `can_update` and blocked
        # enrollment, so a film with eleven speakers produced ONE speaker
        # switch. Below the provisional threshold there is nothing to be
        # ambiguous between.
        ambiguous = (
            second is not None
            and margin < self.min_confidence_margin
            and best is not None
            and best >= self.provisional_threshold
        )
        current_index = None
        if self.last_confidently_active_speaker is not None:
            try:
                current_index = int(self.last_confidently_active_speaker[1:]) - 1
                current_index = self._canon(current_index)
            except (ValueError, IndexError):
                current_index = None
        current_score = next(
            (score for index, score in scored if index == current_index), 0.0
        )

        if duration < self.min_assignment_duration_s:
            if (
                current_index is not None
                and duration <= self.short_turn_max_duration_s
            ):
                confidence = max(current_score, self.provisional_threshold)
                return self._record(self._base_result(
                    self._speaker_id(current_index), "provisional", confidence,
                    0.0, "short turn retained recent speaker by continuity",
                    duration, best, second, margin, False, "retained-short-turn",
                ), observation_key)
            return self._record(self._base_result(
                None, "unknown", best or 0.0, None,
                "observation shorter than minimum assignment duration",
                duration, best, second, margin, False, None,
            ), observation_key)

        can_update = (
            update
            and unsuitable_reason is None
            and duration + 1e-9 >= self.min_enrollment_duration_s
            and not ambiguous
        )

        if not scored:
            if not can_update:
                reason = unsuitable_reason or (
                    "observation too short to enroll the first speaker"
                )
                return self._record(self._base_result(
                    None, "unknown", 0.0, None, reason, duration,
                    None, None, None, False, None,
                ), observation_key)
            index = self._create_profile(
                vector,
                duration,
                direction_estimate,
                observation_group,
            )
            status: SpeakerStatus = (
                "stable" if self.profile_stable[index] else "provisional"
            )
            if status == "stable":
                self.last_confidently_active_speaker = self._speaker_id(index)
                self.last_speaker_change_timestamp = float(end_s)
            result = self._record(self._base_result(
                self._speaker_id(index), status, 1.0, 0.0,
                "enrolled first speaker profile", duration,
                1.0, None, None, True, "initial-enrollment",
            ), observation_key)
            if status == "stable":
                self._queue_stabilized_profile(self._speaker_id(index), observation_key)
            return result

        if (
            best is not None
            and best < self.new_speaker_threshold
            and can_update
            and len(self._active()) < self.max_speakers
        ):
            index = self._create_profile(
                vector,
                duration,
                direction_estimate,
                observation_group,
            )
            status = "stable" if self.profile_stable[index] else "provisional"
            speaker_id = self._speaker_id(index)
            switch_decision = "new-profile-pending"
            if current_index is None and status == "stable":
                self.last_confidently_active_speaker = speaker_id
                self.last_speaker_change_timestamp = float(end_s)
                switch_decision = "new-profile-active"
            elif current_index is not None:
                if (
                    status == "stable"
                    and duration >= self.switch_hysteresis_s
                ):
                    previous_speaker = self.last_confidently_active_speaker
                    self.last_confidently_active_speaker = speaker_id
                    self.last_speaker_change_timestamp = float(end_s)
                    if previous_speaker != speaker_id:
                        self._speaker_switches += 1
                    switch_decision = "accepted-persistent-new-speaker"
                else:
                    status = "provisional"
                    self._pending_switch = index
                    self._pending_switch_since = float(start_s)
            result = self._record(self._base_result(
                speaker_id, status, 1.0, 1.0 if current_index is not None else 0.0,
                "enrolled clearly separated speaker profile", duration,
                best, second, margin, True, switch_decision,
            ), observation_key)
            if status == "stable":
                self._queue_stabilized_profile(speaker_id, observation_key)
            return result

        # Clean short turns naturally produce lower absolute cosine scores.
        # Accept one before the generic low-score rejection only when an
        # already-stable profile wins by a strong margin. Enrollment and
        # ambiguous turns still use the stricter global thresholds.
        short_best_index = (
            self._canon(best_index) if best_index is not None else None
        )
        short_stable_match = (
            short_best_index is not None
            and self.profile_stable[short_best_index]
            and duration <= self.short_stable_max_duration_s
            and best is not None
            and best >= self.short_stable_threshold
            and margin is not None
            and margin >= self.short_stable_min_margin
        )
        if short_stable_match:
            speaker_id = self._speaker_id(short_best_index)
            previous_speaker = self.last_confidently_active_speaker
            self.last_confidently_active_speaker = speaker_id
            self.last_speaker_change_timestamp = float(end_s)
            self._pending_switch = None
            self._pending_switch_since = None
            changed_speaker = (
                previous_speaker is not None
                and previous_speaker != speaker_id
            )
            if changed_speaker:
                self._speaker_switches += 1
            return self._record(self._base_result(
                speaker_id,
                "stable",
                best,
                best if changed_speaker else 0.0,
                "short turn matched stable profile with strong margin",
                duration,
                best,
                second,
                margin,
                False,
                "accepted-short-stable",
            ), observation_key)

        if best is None or best < self.provisional_threshold:
            if (
                current_index is not None
                and duration <= self.short_turn_max_duration_s
            ):
                return self._record(self._base_result(
                    self._speaker_id(current_index), "provisional",
                    max(current_score, self.provisional_threshold), 0.0,
                    "weak short turn retained recent speaker by continuity",
                    duration, best, second, margin, False, "retained-short-turn",
                ), observation_key)
            reason = unsuitable_reason or (
                "best candidate below provisional threshold"
                if len(self._active()) < self.max_speakers
                else "best candidate below threshold and speaker limit reached"
            )
            return self._record(self._base_result(
                None, "unknown", best or 0.0, None, reason, duration,
                best, second, margin, False, "rejected-low-confidence",
            ), observation_key)

        assert best_index is not None
        best_index = self._canon(best_index)
        if ambiguous:
            return self._record(self._base_result(
                None, "unknown", best, None,
                "top speaker candidates are ambiguous", duration,
                best, second, margin, False, "rejected-ambiguous",
            ), observation_key)

        if best < self.assignment_threshold:
            chosen = current_index if (
                current_index is not None and current_score >= self.retain_threshold
            ) else best_index
            return self._record(self._base_result(
                self._speaker_id(chosen), "provisional",
                current_score if chosen == current_index else best,
                0.0 if chosen == current_index else best,
                "candidate met provisional but not stable threshold",
                duration, best, second, margin, False, "provisional",
            ), observation_key)

        was_stable = self.profile_stable[best_index]
        centroid_updated = False
        if can_update:
            centroid_updated = self._update_profile(
                best_index,
                vector,
                duration,
                direction_estimate,
                observation_group,
            )
        became_stable = not was_stable and self.profile_stable[best_index]
        speaker_id = self._speaker_id(best_index)

        if current_index is None:
            status = "stable" if self.profile_stable[best_index] else "provisional"
            if status == "stable":
                self.last_confidently_active_speaker = speaker_id
                self.last_speaker_change_timestamp = float(end_s)
            result = self._record(self._base_result(
                speaker_id, status, best, 0.0,
                "matched enrolled speaker profile", duration,
                best, second, margin, centroid_updated, "initial-active",
            ), observation_key)
            if became_stable:
                self._queue_stabilized_profile(speaker_id, observation_key)
            return result

        if best_index == current_index:
            self._pending_switch = None
            self._pending_switch_since = None
            status = "stable" if self.profile_stable[best_index] else "provisional"
            result = self._record(self._base_result(
                speaker_id, status, best, 0.0,
                "retained current speaker", duration,
                best, second, margin, centroid_updated, "retained-current",
            ), observation_key)
            if became_stable:
                self._queue_stabilized_profile(speaker_id, observation_key)
            return result

        if (
            current_score >= self.retain_threshold
            and best - current_score < self.min_confidence_margin
        ):
            self._pending_switch = None
            self._pending_switch_since = None
            return self._record(self._base_result(
                self._speaker_id(current_index), "provisional", current_score,
                best, "challenger did not clear retention margin",
                duration, best, second, margin, False, "rejected-retention",
            ), observation_key)

        if best < self.switch_threshold:
            return self._record(self._base_result(
                self._speaker_id(current_index), "provisional", current_score,
                best, "challenger did not clear switch threshold",
                duration, best, second, margin, False, "rejected-threshold",
            ), observation_key)

        if self._pending_switch != best_index:
            self._pending_switch = best_index
            self._pending_switch_since = float(start_s)
        persistence = float(end_s) - float(self._pending_switch_since or start_s)
        if persistence < self.switch_hysteresis_s:
            return self._record(self._base_result(
                self._speaker_id(current_index), "provisional", current_score,
                best, "speaker switch awaiting persistence",
                duration, best, second, margin, False, "rejected-hysteresis",
            ), observation_key)

        if not self.profile_stable[best_index]:
            return self._record(self._base_result(
                speaker_id, "provisional", best, best,
                "challenger profile is not stably enrolled",
                duration, best, second, margin, centroid_updated,
                "pending-profile-stability",
            ), observation_key)

        previous_speaker = self.last_confidently_active_speaker
        self.last_confidently_active_speaker = speaker_id
        self.last_speaker_change_timestamp = float(end_s)
        self._pending_switch = None
        self._pending_switch_since = None
        if previous_speaker != speaker_id:
            self._speaker_switches += 1
        result = self._record(self._base_result(
            speaker_id, "stable", best, best,
            "accepted persistent speaker switch", duration,
            best, second, margin, centroid_updated, "accepted-switch",
        ), observation_key)
        if became_stable:
            self._queue_stabilized_profile(speaker_id, observation_key)
        return result

    def provisional_span(
        self,
        start_s: float,
        end_s: float,
        *,
        timestamp_offset: float = 0.0,
    ) -> SpeakerAttribution | None:
        """A FREE guess for a read-ahead word, or None. Never the embedding.

        This is the hypothesis lane: it runs for every uncommitted word on
        every hypothesis emission, so it must cost approximately nothing and
        must have no side effects at all -- no `_record`, no centroid, no
        revision counter. The embedding classifier is ~66 ms on English and
        would be paid several times per word before the word is even
        committed, so the pure-embedding tracker has no cheap signal to offer
        and says so by returning None.

        Returning None is a real answer, not a failure: it publishes a word
        with no speaker, which the renderer draws in neutral read-ahead ink.
        That is what CWI's `unknown` state is for, and it is strictly better
        than asserting a speaker nobody measured.
        """

        return None

    def classify_span(
        self,
        audio: np.ndarray,
        start_s: float,
        end_s: float,
        *,
        observation_key: str | None = None,
        timestamp_offset: float = 0.0,
        overlap: bool = False,
        signal_quality: float | None = None,
        direction_estimate: float | None = None,
    ) -> SpeakerAttribution:
        """Classify without learning; the endpoint remains authoritative."""

        if not self._active():
            # Before the first endpoint there is nothing to classify against;
            # skip the embedding entirely rather than pay it for an unknown.
            return self._record(self._base_result(
                None, "unknown", 0.0, None,
                "no enrolled speaker profiles", end_s - start_s,
                None, None, None, False, None,
            ), observation_key)
        span = audio[max(0, int((end_s - self.window_s) * SR)):int(end_s * SR)]
        if len(span) < int(self.min_assignment_duration_s * SR):
            embedding = None
        else:
            embedding = self.embed(span)
        result = self.observe(
            embedding,
            timestamp_offset + start_s,
            timestamp_offset + end_s,
            update=False,
            overlap=overlap,
            signal_quality=signal_quality,
            direction_estimate=direction_estimate,
            observation_key=observation_key,
        )
        public_result = self._public_attribution(result, observation_key)
        if public_result is not result:
            result = self._record(
                replace(public_result, revision_id=0),
                observation_key,
            )
        # Even a high-confidence mid-stream match remains revisable until the
        # endpoint segmentation sees the whole turn.
        if result.status in {"stable", "corrected"}:
            result = replace(
                result,
                status="provisional",
                reason="classify-only match awaiting endpoint",
            )
            if observation_key is not None:
                self.revision_history[observation_key] = result
        return result

    def label_words(
        self,
        audio: np.ndarray,
        words,
        *,
        observation_keys: list[str] | None = None,
        timestamp_offset: float = 0.0,
        overlap: bool = False,
        signal_quality: float | None = None,
        direction_estimate: float | None = None,
    ) -> list[SpeakerAttribution]:
        """Endpoint segment-then-observe attribution projected onto words."""

        if not words:
            return []
        if observation_keys is not None and len(observation_keys) != len(words):
            raise ValueError("observation_keys must align with words")
        self._observation_group_sequence += 1
        endpoint_observation_group = self._observation_group_sequence
        t0, t1 = words[0].start, words[-1].end
        activity_bounds: list[float] = []
        activity_labels = None
        if self.speaker_activity is not None:
            phrase = audio[int(t0 * SR):int(t1 * SR)]
            spans = [(word.start - t0, word.end - t0) for word in words]
            try:
                activity_labels = self.speaker_activity(phrase, spans)
            except Exception as exc:
                # Speaker colour is an enhancement; a failed segmentation pass
                # must not stop accurate text from reaching the live page.
                if self.debug:
                    print(f"[speaker] segmentation failed: {exc}")
                activity_labels = None
            if activity_labels is not None and len(activity_labels) != len(words):
                activity_labels = None

        if activity_labels is not None:
            # Only clear local-speaker changes become boundaries. Ambiguous
            # overlap/nonspeech labels are skipped until the next clear word.
            previous_label = None
            previous_word = None
            for word, label in zip(words, activity_labels):
                if label is None:
                    continue
                if previous_label is not None and label != previous_label:
                    activity_bounds.append((previous_word.end + word.start) / 2)
                previous_label = label
                previous_word = word

        # Speaker-embedding changes remain necessary for sequential, non-
        # overlapping dialogue: a powerset segmentation stream may legally be
        # reused by a different voice after the first one stops. Non-overlapping
        # windows avoid the previous 75%-shared-audio smear at real turns.
        windows: list[tuple[float, np.ndarray]] = []
        t = t0
        while t + self.window_s <= t1 + 1e-9:
            embedding = self._normalized_embedding(
                self.embed(audio[int(t * SR):int((t + self.window_s) * SR)])
            )
            if embedding is not None:
                windows.append((t + self.window_s / 2, embedding))
            t += self.hop_s
        acoustic_bounds = []
        for (m1, e1), (m2, e2) in zip(windows, windows[1:]):
            if float(np.dot(e1, e2)) < self.change_below:
                acoustic_bounds.append((m1 + m2) / 2)
        # Endpoint punctuation gives a second, acoustically independent
        # opportunity to compare voices. Overlapping 1 s embedding windows can
        # smooth a real turn so strongly that their adjacent cosine never crosses
        # ``change_below`` (the bundled sample merged two speakers into one
        # 3.8 s profile this way). A sentence boundary is not assumed to be a
        # speaker change: it merely creates clean spans which are still matched
        # against the same centroids and will reuse the same id for one speaker.
        terminal = re.compile(r"""[.?!]["')\]]*$""")
        punctuation_candidates: list[tuple[int, float]] = []
        for index, (left, right) in enumerate(zip(words, words[1:])):
            if terminal.search(left.text):
                punctuation_candidates.append((
                    index,
                    (left.end + right.start) / 2,
                ))
        punctuation_bounds = []
        previous_index = -1
        for index, boundary in punctuation_candidates:
            sentence_word_count = index - previous_index
            next_word = words[index + 1]
            # A one-word "sentence" with no timestamped gap is commonly the
            # first word of the following speaker's continuing turn ("Tab? I
            # can't..."). Keeping it with the following span prevents silence-
            # padded one-word embeddings from enrolling a phantom speaker.
            if (
                sentence_word_count == 1
                and next_word.start - words[index].end < 0.08
            ):
                continue
            punctuation_bounds.append(boundary)
            previous_index = index
        # A coarse one-second embedding boundary near an exact verifier
        # punctuation boundary can otherwise carve a mixed sliver out of both
        # turns and enroll a phantom third speaker. Prefer the word-aligned
        # boundary within a 750 ms neighbourhood.
        punctuation_reference_bounds = [
            boundary for _, boundary in punctuation_candidates
        ]
        acoustic_bounds = [
            boundary for boundary in acoustic_bounds
            if not any(
                abs(boundary - punctuation) < min(0.75, self.window_s * 0.75)
                for punctuation in punctuation_reference_bounds
            )
        ]
        bounds = [
            t0,
            *activity_bounds,
            *acoustic_bounds,
            *punctuation_bounds,
            t1,
        ]
        bounds = sorted(set(float(np.clip(boundary, t0, t1))
                            for boundary in bounds))
        cleaned = [bounds[0]]
        for boundary in bounds[1:]:
            if boundary - cleaned[-1] >= self.min_assignment_duration_s:
                cleaned.append(boundary)
        if cleaned[-1] < t1:
            cleaned.append(t1)

        segments: list[tuple[float, float, SpeakerAttribution]] = []
        for start, end in zip(cleaned, cleaned[1:]):
            embedding = self.embed(audio[int(start * SR):int(end * SR)])
            result = self.observe(
                embedding,
                timestamp_offset + start,
                timestamp_offset + end,
                update=True,
                overlap=overlap,
                signal_quality=signal_quality,
                direction_estimate=direction_estimate,
                observation_group=endpoint_observation_group,
            )
            segments.append((start, end, result))

        labels = []
        for index, word in enumerate(words):
            midpoint = (word.start + word.end) / 2
            result = segments[-1][2]
            for start, end, segment_result in segments:
                if start - 1e-9 <= midpoint <= end + 1e-9:
                    result = segment_result
                    break
            key = observation_keys[index] if observation_keys is not None else None
            public_result = self._public_attribution(result, key)
            labels.append(self._record(
                replace(public_result, revision_id=0),
                key,
            ))
        return self._smooth_isolated_switches(
            labels, observation_keys=observation_keys,
        )

    def _smooth_isolated_switches(
        self,
        labels: list[SpeakerAttribution],
        observation_keys: list[str] | None = None,
    ) -> list[SpeakerAttribution]:
        """A LONE WORD MAY NOT CHANGE SPEAKER IN THE MIDDLE OF AN UTTERANCE.

        Identity is decided per SEGMENT, and a segment may be one word long.
        Three paths hand a single word a saturated `stable` label with no
        persistence check, and `switch_hysteresis_s` guards none of them —
        it measures observation TIME, not word runs.

        Measured: of 11 colour runs across 104 settled words, four were exactly
        one word, including three consecutive single-word flips inside one
        narrator's sentence. CWI 2.1 makes colour the speaker signal, so a
        false flip is worse than neutral — it lies about who is talking.

        The rule is boundary-aware: real turn-taking happens at the EDGES of an
        utterance, so the first and last word stay instant. Inside, a switch
        needs `speaker_min_run_words` neighbours to corroborate it. Runs after
        attribution and mutates no centroid, so enrollment is untouched.
        """
        run = max(1, int(self.speaker_min_run_words))
        if run <= 1 or len(labels) < 3:
            return labels
        ids = [label.speaker_id for label in labels]
        smoothed = list(labels)
        for index in range(1, len(ids) - 1):
            current = ids[index]
            if current is None:
                continue
            # How many neighbours on each side agree with this word?
            left = index - 1
            while left >= 0 and ids[left] == current:
                left -= 1
            right = index + 1
            while right < len(ids) and ids[right] == current:
                right += 1
            if (right - left - 1) >= run:
                continue
            # An isolated run. Only override when BOTH sides agree on someone
            # else -- otherwise this is the edge of a genuine turn, not a blip.
            before = ids[left] if left >= 0 else None
            after = ids[right] if right < len(ids) else None
            if before is None or before != after:
                continue
            donor = smoothed[left]
            smoothed[index] = replace(
                labels[index],
                speaker_id=before,
                status=donor.status,
                confidence=min(labels[index].confidence, donor.confidence),
                switch_decision="smoothed-isolated-word",
            )
            # ...AND MAKE IT STICK. `_record` (called before this, in
            # `label_words`) already wrote the PRE-smoothing id into
            # `revision_history`, and `_queue_stabilized_profile` later walks
            # that same dict and re-emits what it finds through
            # `drain_revisions` -- so a word fixed here could be un-fixed by a
            # later correction event carrying the id this rule just removed.
            # Overwrite the stored entry rather than calling `_record` again:
            # recording would see a differing previous id, relabel the word
            # `corrected` and inflate `metrics()["corrections"]`, which is a
            # behavioural change no test asked for. The smoothed value still
            # reaches the page through the returned `labels`.
            if observation_keys is not None and index < len(observation_keys):
                key = observation_keys[index]
                if key is not None and key in self.revision_history:
                    self.revision_history[key] = replace(
                        self.revision_history[key],
                        speaker_id=before,
                        switch_decision="smoothed-isolated-word",
                    )
            # Say so. This rule rewrites a speaker AFTER attribution and
            # deliberately does not `_record`, so until now it was the one
            # decision in the system that left no trace in `debug: true`
            # output -- invisible while being the first thing anyone blames
            # for a wrong colour.
            if self.debug:
                print("[speaker] " + json.dumps({
                    "observation": (
                        observation_keys[index]
                        if observation_keys is not None
                        and index < len(observation_keys)
                        else None
                    ),
                    "smoothed_from": labels[index].speaker_id,
                    "smoothed_to": before,
                    "run_words": right - left - 1,
                    "min_run_words": run,
                }, ensure_ascii=False))
        return smoothed


class SortformerHybridSpeakerTracker:
    """Streaming Sortformer decisions with the proven embedding fallback.

    Sortformer owns low-latency arrival-ordered speaker slots. The existing
    segmentation/embedding tracker still runs at endpoints so quiet speech
    missed by Sortformer is never forced into the wrong slot.
    """

    def __init__(
        self,
        bridge,
        fallback: SpeakerTracker,
        *,
        min_word_coverage: float = 0.24,
        endpoint_wait_ms: float = 90.0,
        debug: bool = False,
    ):
        self.bridge = bridge
        self.fallback = fallback
        self.min_word_coverage = float(
            np.clip(min_word_coverage, 0.0, 1.0)
        )
        self.endpoint_wait_ms = max(0.0, float(endpoint_wait_ms))
        self.debug = bool(debug)
        self.embed = fallback.embed
        self.speaker_activity = fallback.speaker_activity
        self._sortformer_decisions = 0
        self._embedding_fallbacks = 0
        # Sortformer speaker slots are arrival-ordered within the native
        # session. Endpoint embeddings attach those slots to the durable S1…
        # identity namespace, and can merge a transient phantom slot back to
        # a known speaker without rewriting future provisional words.
        self._slot_speakers: dict[int, str] = {}

    def __getattr__(self, name):
        return getattr(self.fallback, name)

    @staticmethod
    def _speaker_id(index: int) -> str:
        return f"S{int(index) + 1}"

    def feed(
        self,
        samples: np.ndarray,
        *,
        source_start: float,
        discontinuity: bool = False,
    ) -> None:
        try:
            self.bridge.feed(
                samples,
                source_start=source_start,
                discontinuity=discontinuity,
            )
        except Exception as exc:
            if self.debug:
                print(f"[speaker] Sortformer feed failed: {exc}")

    def finish(self) -> None:
        try:
            self.bridge.finish()
        except Exception as exc:
            if self.debug:
                print(f"[speaker] Sortformer finalize failed: {exc}")

    def close(self) -> None:
        self.bridge.close()

    def _from_sortformer(
        self,
        decision,
        start_s: float,
        end_s: float,
        *,
        observation_key: str | None,
        endpoint: bool,
    ) -> SpeakerAttribution | None:
        if decision is None or decision.coverage < self.min_word_coverage:
            return None
        self._sortformer_decisions += 1
        confidence = float(np.clip(
            0.55 + 0.35 * decision.coverage + 0.10 * decision.activity,
            0.0,
            1.0,
        ))
        speaker_id = self._slot_speakers.get(decision.speaker_index)
        if speaker_id is None:
            if decision.speaker_index >= self.fallback.immediate_speaker_limit:
                # Native 4-speaker models occasionally flash a faint extra
                # track. Let the non-learning embedding classifier retain a
                # known voice (or return unknown) until an endpoint verifies
                # that this really is an additional person.
                return None
            speaker_id = self._speaker_id(decision.speaker_index)
        result = self.fallback._base_result(
            speaker_id,
            "stable" if endpoint else "provisional",
            confidence,
            None,
            (
                # The native slot is in the reason on purpose: it is the only
                # way to tell "the model never separated them" from "the model
                # separated them and the mapping lost it", and those need
                # opposite fixes.
                f"streaming Sortformer {'endpoint' if endpoint else 'tentative'}"
                f" assignment (native slot {decision.speaker_index})"
            ),
            max(0.0, end_s - start_s),
            None,
            None,
            None,
            False,
            "sortformer",
        )
        return self.fallback._record(result, observation_key)

    def provisional_span(
        self,
        start_s: float,
        end_s: float,
        *,
        timestamp_offset: float = 0.0,
    ) -> SpeakerAttribution | None:
        """The native timeline's answer for a read-ahead word, or None.

        WHY THE HYPOTHESIS LANE NEEDS ITS OWN METHOD. `classify_span` falls
        through to the embedding tracker when Sortformer has nothing, which is
        right at commit time and wrong here: this runs for every uncommitted
        word on every hypothesis emission. So this asks the native timeline
        only -- an in-memory scan of recent segments under a lock, costing
        microseconds -- and returns None rather than reaching for the model.

        NOTHING IS RECORDED. `_from_sortformer` routes through
        `_record`, which bumps the global revision counter, overwrites
        `revision_history[word_id]` and drives the endpoint's own
        stable->corrected logic. Calling that several times per word before the
        word is committed would let a read-ahead guess masquerade as the
        previous durable answer an endpoint is about to correct. The result
        here is display-only and deliberately carries `revision_id=0`, which
        the store still accepts because `newestRevision` falls through to the
        SSE id on a tie.
        """

        try:
            decision = self.bridge.decision(
                timestamp_offset + start_s,
                timestamp_offset + end_s,
            )
        except Exception:
            # A dead helper must degrade to neutral captions, never abort one.
            return None
        if decision is None or decision.coverage < self.min_word_coverage:
            return None
        speaker_id = self._slot_speakers.get(decision.speaker_index)
        if speaker_id is None:
            if decision.speaker_index >= self.fallback.immediate_speaker_limit:
                # An unverified higher slot has no durable name yet, and
                # inventing one here would publish a speaker the endpoint has
                # never seen. Neutral until it does.
                return None
            speaker_id = self._speaker_id(decision.speaker_index)
        confidence = float(np.clip(
            0.55 + 0.35 * decision.coverage + 0.10 * decision.activity,
            0.0,
            1.0,
        ))
        return SpeakerAttribution(
            speaker_id=speaker_id,
            status="provisional",
            confidence=confidence,
            speaker_change_probability=None,
            revision_id=0,
            reason=(
                "streaming Sortformer read-ahead"
                f" (native slot {decision.speaker_index})"
            ),
            observation_duration_s=max(0.0, end_s - start_s),
        )

    def classify_span(
        self,
        audio: np.ndarray,
        start_s: float,
        end_s: float,
        *,
        observation_key: str | None = None,
        timestamp_offset: float = 0.0,
        overlap: bool = False,
        signal_quality: float | None = None,
        direction_estimate: float | None = None,
    ) -> SpeakerAttribution:
        absolute_start = timestamp_offset + start_s
        absolute_end = timestamp_offset + end_s
        decision = self.bridge.decision(absolute_start, absolute_end)
        result = self._from_sortformer(
            decision,
            absolute_start,
            absolute_end,
            observation_key=observation_key,
            endpoint=False,
        )
        if result is not None:
            return result
        self._embedding_fallbacks += 1
        return self.fallback.classify_span(
            audio,
            start_s,
            end_s,
            observation_key=observation_key,
            timestamp_offset=timestamp_offset,
            overlap=overlap,
            signal_quality=signal_quality,
            direction_estimate=direction_estimate,
        )

    def label_words(
        self,
        audio: np.ndarray,
        words,
        *,
        observation_keys: list[str] | None = None,
        timestamp_offset: float = 0.0,
        overlap: bool = False,
        signal_quality: float | None = None,
        direction_estimate: float | None = None,
    ) -> list[SpeakerAttribution]:
        # Always update the embedding profiles. They recover quiet/distant
        # turns and keep the existing >4-speaker path alive.
        fallback = self.fallback.label_words(
            audio,
            words,
            observation_keys=observation_keys,
            timestamp_offset=timestamp_offset,
            overlap=overlap,
            signal_quality=signal_quality,
            direction_estimate=direction_estimate,
        )
        labels = []
        for index, (word, fallback_result) in enumerate(zip(words, fallback)):
            key = observation_keys[index] if observation_keys is not None else None
            absolute_start = timestamp_offset + word.start
            absolute_end = timestamp_offset + word.end
            decision = self.bridge.decision(
                absolute_start,
                absolute_end,
                wait_ms=self.endpoint_wait_ms,
            )
            # Endpoint embeddings are the identity verifier. Sortformer is
            # intentionally trusted for *when* a speaker changed, but a clean
            # full-turn embedding remains more reliable for *who* the slot is.
            # This also prevents a brief Sortformer phantom speaker from
            # splitting a paragraph after verification.
            if (
                decision is not None
                and decision.coverage >= self.min_word_coverage
                and fallback_result.speaker_id is not None
            ):
                self._sortformer_decisions += 1
                resolved_speaker = self._slot_speakers.get(
                    decision.speaker_index,
                    self._speaker_id(decision.speaker_index),
                )
                if fallback_result.status in {"stable", "corrected"}:
                    # REWRITE, DO NOT `setdefault`. A native slot is
                    # arrival-ordered WITHIN THE NATIVE SESSION and the model
                    # has four of them, so on any real programme it reuses a
                    # slot for a different person. Pinning a slot to whoever
                    # was verified in it FIRST meant every later voice on that
                    # slot inherited the first one's colour for the whole
                    # session: on the PR film the drill sergeant arrived on a
                    # slot the narrator already owned and rendered in her
                    # colour until the endpoint corrected him, a second late,
                    # every time he spoke.
                    self._slot_speakers[decision.speaker_index] = (
                        fallback_result.speaker_id
                    )
                    labels.append(fallback_result)
                    continue
                if fallback_result.speaker_id != resolved_speaker:
                    # A weak endpoint embedding that agrees with an already
                    # known speaker is safer than inventing a transient third
                    # speaker from a short Sortformer track.
                    labels.append(fallback_result)
                    continue
            result = self._from_sortformer(
                decision,
                absolute_start,
                absolute_end,
                observation_key=key,
                endpoint=True,
            )
            if result is None:
                self._embedding_fallbacks += 1
                result = fallback_result
            labels.append(result)
        # SMOOTH AGAIN, BECAUSE THE LOOP ABOVE THREW THE SMOOTHING AWAY.
        # `self.fallback.label_words` ends by running
        # `_smooth_isolated_switches`, but every word that took a native slot
        # decision here replaced that smoothed answer with a raw per-word
        # argmax -- `select_sortformer_decision` has no temporal state at all,
        # so consecutive words are free to land on different slots. The bypass
        # is the `fallback_result.speaker_id is not None` guard above: whenever
        # the embedding tracker returned neutral, control reaches
        # `_from_sortformer(endpoint=True)` and stamps a lone `stable` label
        # with no run-length check of any kind.
        # MEASURED on the PR film, settled runs came out
        # [38, 2, 7, 7, 13, 7, 15, 1, 14, 3, 2, 9, 15, 4, 5, 2, 10, 7, 4, 6]:
        # a colour change every 8.6 words with 25% of runs three words or
        # shorter, inside sentences spoken by one person. CWI 2.1 makes colour
        # THE speaker signal, so each of those asserts a speaker change that
        # did not happen.
        return self.fallback._smooth_isolated_switches(
            labels, observation_keys=observation_keys,
        )

    def drain_revisions(self) -> list[tuple[str, SpeakerAttribution]]:
        return self.fallback.drain_revisions()

    def metrics(self) -> dict[str, int]:
        return {
            **self.fallback.metrics(),
            "sortformer_decisions": self._sortformer_decisions,
            "embedding_fallbacks": self._embedding_fallbacks,
        }


class StreamingCaptioner:
    """Turn online recognizer revisions into stable CaptionSpec-shaped words."""

    def __init__(self, recognizer, cfg: dict, speaker: str = "S1",
                 draft_only: bool = False, verifier: EndpointVerifier | None = None,
                 speaker_tracker: SpeakerTracker | None = None):
        from collections import deque

        self.recognizer = recognizer
        self.cfg = cfg
        self.speaker = speaker
        self.draft_only = draft_only
        self.verifier = verifier
        # Bilingual only: scripts the session may show. Words failing
        # `word_script_allowed` are dropped before anything downstream sees
        # them. Empty = off.
        self.allowed_scripts = frozenset(
            str(item).strip().lower()
            for item in ((cfg.get("live", {}) or {}).get("allowed_scripts") or [])
            if str(item).strip()
        )
        # Bilingual only: run the English endpoint verifier on English
        # utterances and leave Korean ones to the stream. See `_verifier_skips`.
        self.verifier_latin_only = bool(
            (cfg.get("live", {}) or {}).get("verifier_latin_only", False))
        self.speaker_tracker = speaker_tracker
        self.stream = self._new_stream()
        self.stream_base = 0.0
        # Built only when a node attaches, alongside `direction_for_span`, so a
        # local mic keeps the existing grey-until-decided behaviour untouched.
        speaker_cfg = (cfg.get("live", {}) or {}).get("speaker_attribution", {}) or {}
        # Prefer the tracker's map when there is one: the draft and accurate
        # captioners must agree about which slot is which, and the level meter
        # reads the same slots to mark them on the compass.
        shared = getattr(speaker_tracker, "direction_speakers", None)
        self.direction_speakers = shared if shared is not None else (
            DirectionSpeakerMap(
                tolerance_deg=float(
                    speaker_cfg.get("direction_slot_tolerance_deg", 35.0)))
            if speaker_cfg.get("direction_slot_colouring", True) else None
        )
        # The compass's "who is where" marks. Shared from the tracker for the
        # same reason the slot map is: draft and accurate captioners must feed
        # ONE map, or the ring shows two disagreeing sets of marks. Absent
        # without a node, exactly like every other direction consumer.
        self.speaker_bearings = getattr(speaker_tracker, "speaker_bearings", None)
        # HOW MUCH THE GEOMETRY IS ALLOWED TO SAY. See
        # `_colour_from_direction` and `SpeakerBearingMap.match`.
        self.direction_identity = bool(
            speaker_cfg.get("direction_identity", True))
        self.direction_match_sigma = float(
            speaker_cfg.get("direction_match_sigma", 2.0))
        self.direction_min_observations = int(
            speaker_cfg.get("direction_min_observations", 4))
        self.direction_override_below = float(
            speaker_cfg.get("direction_override_below", 0.45))
        self.total_samples = 0
        self.audio_blocks: list[np.ndarray] = []
        self.previous: list[HypothesisWord] = []
        self.committed: list[HypothesisWord] = []
        self.last_partial_key: tuple = ()
        self.utterance = 0
        self.db_history: deque[float] = deque(maxlen=120)
        # Vocal effort runs on its own history and its own percentile window so
        # the level channel keeps its full range: folding effort into `db`
        # before normalization widens that window and MEASURED flattens the
        # narration's own intonation (p90 1.25 -> 1.00) to buy the shout.
        # `(tilt_db, pitch_hz, span_s)` per settled word -- the three cues
        # `_prominence` combines, kept raw so the composite can be re-derived
        # against the speaker's CURRENT baseline rather than a stale one.
        self.effort_history: deque[tuple[float, float, float]] = deque(maxlen=120)
        # ...and the cold-start bootstrap, on exactly the terms `db_bootstrap`
        # uses: `effort_history` takes FINAL words only, and the PR film's
        # opening monologue is one 24 s utterance, so without this the lift
        # cannot fire until the 33rd second. That is why "louder" never moved.
        self.effort_bootstrap: dict[tuple[str, int, int], tuple[float, float, float]] = {}
        self.effort_cache: dict[tuple[str, int, int], float] = {}
        self.prosody_cache: dict[tuple[str, int], tuple[float, float, float]] = {}
        self.delivery_cache: dict[tuple[str, int, int], dict] = {}
        self._last_final_speaker: str | None = None
        # Zero-arg callable returning the newest bearing in degrees, or None
        # when nothing is measuring one. Set by the live runner when a mic
        # array is attached; unset for a local mic, which is why a mono capture
        # produces no `direction_deg` and the compass says `awaiting array`.
        self.direction_source = None
        self._word_slots: list[tuple[float, float, str]] = []
        self._final_word_events: dict[str, dict] = {}
        self._word_revisions: dict[str, dict] = {}
        # Whether read-ahead words carry a provisional speaker at all. A flag
        # rather than a constant so the lane can be A/B'd in one line: it is
        # the newest and least-guarded attribution path, and it publishes a raw
        # per-word Sortformer argmax, so it is the first suspect whenever
        # colour STABILITY regresses even as correctness improves.
        self._read_ahead_attribution = bool(
            cfg.get("live", {})
            .get("speaker_attribution", {})
            .get("read_ahead_attribution", True)
        )
        # STABILITY MODE: never introduce a NEW specific speaker mid-stream.
        # Measured on real Korean dialogue, 18% of words are attributed to one
        # speaker mid-turn and CORRECTED at the endpoint -- and every flip is a
        # confident wrong guess on a short span (the embedding matches the wrong
        # enrolled voice before the full utterance is in). When on, a mid-stream
        # pick that is neither the currently-active speaker nor a `stable`
        # decision is published GREY instead, so the word goes grey -> right
        # colour (not flicker) rather than wrong colour -> right colour. This is
        # the design system's own preference -- "a wrong colour is a false claim
        # about who spoke; neutral is the unknown state" -- traded against a
        # little more first-paint grey. Default OFF: it is UNMEASURED for its
        # cost in first-paint correctness (no labelled data), and the array's
        # `direction_trust` is the better fix where the hardware is present.
        self._readahead_stability = bool(
            cfg.get("live", {})
            .get("speaker_attribution", {})
            .get("readahead_stability", False)
        )

    @property
    def audio(self) -> np.ndarray:
        if not self.audio_blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.audio_blocks)

    def _prosody(self, word: HypothesisWord, audio: np.ndarray) -> tuple[float, float, float]:
        i0 = max(0, int(word.start * SR))
        i1 = min(len(audio), max(i0 + 1, int(word.end * SR)))
        span = audio[i0:i1]
        db = _span_db(span) if len(span) else -80.0

        # Pitch benefits from a small context pad while loudness stays confined
        # to the recognized word itself.
        p0, p1 = max(0, i0 - int(0.04 * SR)), min(len(audio), i1 + int(0.04 * SR))
        pitch_hz = 0.0
        voiced_frac = 0.0
        if p1 - p0 >= int(0.04 * SR):
            import parselmouth

            try:
                snd = parselmouth.Sound(audio[p0:p1].astype(np.float64),
                                        sampling_frequency=SR)
                pitch = snd.to_pitch_cc(
                    pitch_floor=self.cfg["prosody"]["pitch_floor_hz"],
                    pitch_ceiling=self.cfg["prosody"]["pitch_ceiling_hz"],
                )
                values = pitch.selected_array["frequency"]
                voiced = values[values > 0]
                if len(voiced):
                    pitch_hz = float(np.median(voiced))
                voiced_frac = float(len(voiced) / max(len(values), 1))
            except parselmouth.PraatError:
                pass
        return db, pitch_hz, voiced_frac

    @staticmethod
    def _percentile_window(values, percentiles, min_span: float) -> tuple[float, float]:
        """Percentiles of a speaker's own history, widened to `min_span`.

        A monotone passage must not amplify millimetre differences into full
        whisper..shout swings, so the window never closes below `min_span`.
        """
        lo_pct, hi_pct = percentiles
        lo = float(np.percentile(values, lo_pct))
        hi = float(np.percentile(values, hi_pct))
        if hi - lo < min_span:
            mid = (hi + lo) / 2
            lo, hi = mid - min_span / 2, mid + min_span / 2
        return lo, hi

    def _vocal_effort(self, word: HypothesisWord, audio: np.ndarray) -> float | None:
        """Spectral tilt of the word's strongest frames, in dB. Gain-invariant.

        LEVEL CANNOT SEE A SHOUT ON MASTERED OR AUTO-LEVELLED AUDIO, AND CWI
        2.3.5 ASKS FOR VOLUME, NOT LEVEL. Measured on the PR film's drill
        sergeant (`--sample`, t=30-37.5 s): shouting at full effort, and
        **1.6 dB QUIETER** than the calm narration around it, because the clip
        is broadcast-mastered. Ranking those words against the narration by
        level scores **AUC 0.367** -- worse than chance, actively backwards.
        What survives mastering is vocal effort: pressed phonation flattens
        the spectrum, so the drill sergeant's tilt runs -5.7 dB against the
        narration's -13.7 dB (AUC 0.801 raw, 0.892 causally smoothed).

        Energy-weighted on purpose. A word's fricatives carry enormous HF
        energy and its silences carry none, so tilt over the whole span
        measures PHONEMES, not effort -- that version scored a useless AUC and
        a 21 dB per-word spread. Taking the loudest half of the frames
        measures the tilt where the voice actually is.
        """
        i0 = max(0, int(word.start * SR))
        i1 = min(len(audio), max(i0 + 1, int(word.end * SR)))
        span = audio[i0:i1]
        n = int(0.03 * SR)
        if len(span) < n:
            return None
        window = np.hanning(n)
        freqs = np.fft.rfftfreq(n, 1 / SR)
        low_band = (freqs >= 100) & (freqs < 1000)
        high_band = (freqs >= 1000) & (freqs < 5000)
        frames: list[tuple[float, float]] = []
        for i in range(0, len(span) - n + 1, max(1, n // 2)):
            spectrum = np.abs(np.fft.rfft(span[i:i + n] * window)) ** 2
            low = float(spectrum[low_band].sum())
            if low <= 0:
                continue
            high = float(spectrum[high_band].sum())
            frames.append((low + high, 10 * np.log10(high / low + 1e-12)))
        if not frames:
            return None
        frames.sort(key=lambda frame: -frame[0])
        keep = frames[:max(1, len(frames) // 2)]
        return float(np.mean([tilt for _, tilt in keep]))

    @staticmethod
    def _prominence(
        cues: list[tuple[float, float, float]],
        baseline: tuple[float, float],
        cfg: dict,
    ) -> list[float]:
        """Effort + pitch excursion + lengthening, in the tilt's own dB units.

        TILT ALONE CANNOT SEE THE EMPHASIS THE FILM DRAWS BIGGEST. The film
        sets "louder" at 2.08x its settled height, and measured, that word is
        only ~1 dB above its neighbours — level cannot see it. Its tilt, its
        F0 and its per-character duration all can, and all three survive
        mastering.

        THE SMOOTHING IS WHAT ERASED IT. `smoothing_words` exists because
        sustained pressed phonation is a speaking STYLE that a causal mean
        reads far better than one word does. A single stressed word is an
        EVENT, and that same mean deletes it: the calm narration before
        "louder" drags the mean below the median and the lift computes to
        exactly 0.000 however the rest is tuned. `emphasis_blend` is how much
        of the word's OWN tilt survives the mean; pitch is what buys the
        detector's AUC back.

        Lengthening is per CHARACTER, not per word, for the same reason tilt is
        energy-weighted: raw duration mostly measures syllable count. All three
        terms are ratios against the speaker's own running median, in dB, so
        they add. No voiced pitch, or no baseline yet, contributes nothing
        rather than a guess.

        Takes the whole sequence because the blend needs causal neighbours.
        """
        med_pitch, med_span = baseline
        blend = float(np.clip(cfg.get("emphasis_blend", 1.0), 0.0, 1.0))
        window = max(1, int(cfg.get("smoothing_words", 4)))
        pitch_gain = float(cfg.get("pitch_gain", 0.0))
        length_gain = float(cfg.get("length_gain", 0.0))
        scores: list[float] = []
        for index, (tilt, pitch_hz, span_s) in enumerate(cues):
            recent = [c[0] for c in cues[max(0, index - window + 1):index + 1]]
            score = blend * float(tilt) + (1 - blend) * float(np.mean(recent))
            if pitch_gain and pitch_hz > 0 and med_pitch > 0:
                score += pitch_gain * 20.0 * float(np.log10(pitch_hz / med_pitch))
            if length_gain and span_s > 0 and med_span > 0:
                score += length_gain * 20.0 * float(np.log10(span_s / med_span))
            scores.append(score)
        return scores

    def _intonation_envelope(
        self, word: HypothesisWord, audio: np.ndarray, samples: int,
    ) -> dict[str, list[float]] | None:
        """CWI 2.3 INSIDE the word: the contour, not one number for the span.

        The design system's own illustrations are per-CHARACTER, not per-word.
        p.34 sets "Put that coffee dOWn!" directly beneath its waveform with the
        `O` and `W` huge and the `d` and `n!` small; p.38 does the same to weight
        with "neeee**eeeed**" under a pitch curve; p.40 ramps one sentence from
        black to hairline. A single median per word cannot express any of it,
        and `_prosody` above throws the contour away precisely when it computes
        it.

        Returns `samples` evenly spaced readings across the word's own span:
        `loudness` in dB (normalised client-side against the same 2.3.5 pivot as
        the word-level value), `pitch` in Hz, and `texture` as the spectral
        centroid the width axis already uses. Returns None when the span is too
        short to say anything honest about its shape -- the client then falls
        back to the word-level value for every character.
        """
        n = int(samples)
        if n < 2:
            return None          # 0 disables the per-character channel entirely
        i0 = max(0, int(word.start * SR))
        i1 = min(len(audio), max(i0 + 1, int(word.end * SR)))
        span = audio[i0:i1]
        # Below roughly a 20 ms window per sample there is no contour left to
        # measure, only noise; a flat envelope would be a fabricated shape.
        if len(span) < n * int(0.02 * SR):
            return None

        edges = np.linspace(0, len(span), n + 1).astype(int)
        loudness = [
            _rms_db(span[a:b]) if b > a else -80.0
            for a, b in zip(edges[:-1], edges[1:])
        ]

        pitch_track = [0.0] * n
        p0, p1 = max(0, i0 - int(0.04 * SR)), min(len(audio), i1 + int(0.04 * SR))
        if p1 - p0 >= int(0.04 * SR):
            import parselmouth

            try:
                snd = parselmouth.Sound(audio[p0:p1].astype(np.float64),
                                        sampling_frequency=SR)
                pitch = snd.to_pitch_cc(
                    pitch_floor=self.cfg["prosody"]["pitch_floor_hz"],
                    pitch_ceiling=self.cfg["prosody"]["pitch_ceiling_hz"],
                )
                values = pitch.selected_array["frequency"]
                times = pitch.xs() + (p0 - i0) / SR   # relative to the word
                duration = max(1e-6, len(span) / SR)
                for index in range(n):
                    lo = index * duration / n
                    hi = (index + 1) * duration / n
                    window = values[(times >= lo) & (times < hi)]
                    voiced = window[window > 0]
                    # Unvoiced stays 0.0, which `voiceTone` already reads as
                    # neutral rather than as an impossibly low voice.
                    if len(voiced):
                        pitch_track[index] = float(np.median(voiced))
            except parselmouth.PraatError:
                pass

        texture = []
        for a, b in zip(edges[:-1], edges[1:]):
            chunk = span[a:b]
            if len(chunk) < 32:
                texture.append(0.0)
                continue
            spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
            freqs = np.fft.rfftfreq(len(chunk), 1 / SR)
            total = float(spectrum.sum())
            texture.append(
                float((spectrum * freqs).sum() / total) if total > 1e-9 else 0.0
            )

        return {
            "loudness": [round(v, 2) for v in loudness],
            "pitch": [round(v, 1) for v in pitch_track],
            "texture": [round(v, 1) for v in texture],
        }

    def _syllable_stops(self, word: HypothesisWord) -> list[dict] | None:
        """CWI 2.2.4 stops: how far the colour has advanced through the word.

        Each stop is ``{"t": fraction of the word's span, "c": fraction of its
        characters}``. Returns None unless the word is drawn out enough to
        qualify and the recognizer gave it distinct sub-word onsets.
        """

        fill_cfg = self.cfg.get("motion", {}).get("syllable_fill", {}) or {}
        if not fill_cfg.get("enabled", True):
            return None
        duration = word.end - word.start
        if duration < float(fill_cfg.get("min_duration_s", 0.32)):
            return None
        groups = word.syllables()
        if len(groups) < int(fill_cfg.get("min_syllables", 2)):
            return None
        total_chars = sum(len(text) for text, _ in groups)
        if not total_chars:
            return None
        stops: list[dict] = []
        chars = 0
        for text, start in groups:
            stops.append({
                "t": round(float(np.clip((start - word.start) / duration, 0.0, 1.0)), 4),
                "c": round(chars / total_chars, 4),
            })
            chars += len(text)
        stops.append({"t": 1.0, "c": 1.0})
        # A leading stop that already sits at t>0 would hold the word white past
        # its own onset, breaking the 2.2.2 contract that colour starts at onset.
        stops[0]["t"] = 0.0
        return stops

    def _word_id(self, word: HypothesisWord) -> str:
        return self._word_ids_for([word])[0]

    def _word_ids_for(self, words: list[HypothesisWord]) -> list[str]:
        """One-to-one timing alignment onto stable ids within an utterance."""

        slots = getattr(self, "_word_slots", None)
        if slots is None:
            slots = self._word_slots = []
        used: set[str] = set()
        ids = []
        for word in words:
            candidates = [
                (abs(start - word.start) + 0.5 * abs(end - word.end), word_id)
                for start, end, word_id in slots
                if word_id not in used and abs(start - word.start) < 0.22
            ]
            if candidates:
                _, word_id = min(candidates)
            else:
                word_id = f"u{self.utterance}:w{len(slots)}"
                slots.append((word.start, word.end, word_id))
            used.add(word_id)
            ids.append(word_id)
        return ids

    def _read_ahead_speaker(self, word) -> SpeakerAttribution | None:
        """Free provisional identity for a word that has not committed yet.

        Deliberately NOT cached per word id. A hypothesis word is re-published
        as the recognizer extends it, and Sortformer's answer for the same span
        IMPROVES as more audio is processed -- a word at the edge of the buffer
        may have no coverage on its first emission and a confident slot two
        emissions later. Freezing the first answer would keep the worst one.
        Churn before the colour turn is invisible anyway: the word is still in
        read-ahead ink until the playhead reaches it.
        """

        tracker = self.speaker_tracker
        if tracker is None or not self._read_ahead_attribution:
            return None
        probe = getattr(tracker, "provisional_span", None)
        if probe is None:
            return None
        try:
            return probe(
                word.start, word.end, timestamp_offset=self.stream_base,
            )
        except Exception:
            return None

    def _attribution(
        self,
        value: SpeakerAttribution | str | None,
        *,
        final: bool,
    ) -> SpeakerAttribution:
        if isinstance(value, SpeakerAttribution):
            return value
        if isinstance(value, str):
            return SpeakerAttribution(
                value, "stable" if final else "provisional", 1.0, 0.0, 0,
                "explicit speaker label",
            )
        if getattr(self, "speaker_tracker", None) is None:
            # Backward-compatible single-speaker degradation when the optional
            # local embedding model is disabled or absent. THIS IS THE ONLY
            # PLACE `self.speaker` IS A DEFENSIBLE ANSWER: with no tracker
            # there is exactly one speaker by construction, so it is a fact
            # rather than a guess. It now covers non-final words too, because
            # the publication path no longer supplies a blanket fallback --
            # without this a diarization-disabled session would render its
            # whole read-ahead neutral.
            return SpeakerAttribution(
                self.speaker, "stable" if final else "provisional", 1.0, 0.0, 0,
                "single-speaker fallback (tracker unavailable)",
            )
        return SpeakerAttribution(
            None, "unknown", 0.0, None, 0, "speaker not yet attributed",
        )

    def _word_revision_fields(
        self,
        word_id: str,
        *,
        text: str,
        t: float,
        start: float,
        end: float,
    ) -> dict[str, int]:
        """Return monotonic per-channel revisions for one displayed word.

        SSE arrival order is insufficient once EventSource reconnect replay and
        the draft/accurate streams interleave.  The browser compares these
        explicit revisions so an old hypothesis cannot roll verified text or
        timing back.  Speaker attribution retains its independent revision id.
        """

        revisions = getattr(self, "_word_revisions", None)
        if revisions is None:
            revisions = self._word_revisions = {}
        previous = revisions.get(word_id)
        if previous is None:
            current = {
                "text": text,
                "timing": (t, start, end),
                "text_revision_id": 1,
                "timing_revision_id": 1,
            }
        else:
            current = dict(previous)
            if text != previous["text"]:
                current["text"] = text
                current["text_revision_id"] += 1
            timing = (t, start, end)
            if timing != previous["timing"]:
                current["timing"] = timing
                current["timing_revision_id"] += 1
        revisions[word_id] = current
        return {
            "text_revision_id": current["text_revision_id"],
            "timing_revision_id": current["timing_revision_id"],
        }

    def _colour_from_direction(self, result, bearing: float | None):
        """Give an undecided word a colour from its bearing.

        The voice lane leaves a new speaker's first words `unknown` for
        seconds -- text at a median 0.62 s, durable identity at 4.48 s -- and
        those words render grey. Azimuth is available immediately, so this
        fills the gap the SpeechCompass way and the endpoint pass corrects it
        afterwards, which is what keeps the claim revisable.

        Only ever *adds* a colour: a word the voice lane has already decided
        is left exactly as it was, so this cannot overwrite real evidence with
        geometry.
        """
        if self.direction_speakers is None:
            return result
        decided = result is not None and getattr(result, "speaker_id", None) is not None

        # WHERE THIS BEARING SAYS THE TALKER IS, against the positions the
        # VOICE lane established. Preferred over a geometric slot wherever it
        # answers: a slot can only ever say "the 2nd distinct direction", while
        # this names a speaker somebody already identified by voice.
        placed = None
        bearings = getattr(self, "speaker_bearings", None)
        if bearings is not None and self.direction_identity:
            placed = bearings.match(
                bearing,
                sigma=self.direction_match_sigma,
                min_observations=self.direction_min_observations,
            )

        if decided:
            # DIRECTION MAY CONTEST A WEAK VOICE DECISION, and only a weak one.
            # The voice lane's own confidence is the gate: a `final` identity,
            # or any decision at or above `direction_override_below`, is left
            # exactly as it was. Below it the lanes genuinely disagree and the
            # bearing is independent evidence -- a different failure mode from
            # the embedding, which is the whole reason it is worth having.
            # Still `provisional`, so the endpoint pass corrects it afterwards.
            if (placed is None or self.direction_override_below <= 0.0
                    or getattr(result, "status", None) == "final"
                    or getattr(result, "confidence", 1.0)
                    >= self.direction_override_below
                    or placed[0] == getattr(result, "speaker_id", None)):
                return result
            return replace(
                result,
                speaker_id=placed[0],
                status="provisional",
                reason="direction-contested",
            )

        if placed is not None:
            return SpeakerAttribution(
                speaker_id=placed[0],
                status="provisional",
                confidence=0.0,
                speaker_change_probability=None,
                revision_id=0,
                reason="direction-placed",
            )

        slot = self.direction_speakers.assign(bearing)
        if slot is None:
            return result                    # no bearing yet, and no turn to continue
        return SpeakerAttribution(
            speaker_id=f"S{slot + 1}",
            status="provisional",
            confidence=0.0,
            speaker_change_probability=None,
            revision_id=0,
            reason="direction",
        )

    def _record_speaker_bearing(self, word_id: str | None, attribution,
                                word) -> None:
        """Remember the bearing a decided speaker was heard at.

        Descriptive only -- nothing reads this back into attribution, and it
        cannot change a word's colour. It exists so the compass ring can mark
        who is where instead of only marking the one position that happened to
        be current while the voice lane was still undecided.

        **Direction-derived identities are skipped.** `_colour_from_direction`
        names a speaker FROM a bearing; recording that bearing under that name
        would be the map confirming its own guess, and the mark would look like
        evidence while carrying none.

        getattr throughout: the tests build captioners with `__new__` to skip
        loading a recognizer, so this must tolerate an instance that never ran
        `__init__`.
        """
        bearings = getattr(self, "speaker_bearings", None)
        if bearings is None or word_id is None:
            return
        speaker_id = getattr(attribution, "speaker_id", None)
        if speaker_id is None:
            return
        # EVERY direction-derived reason, not just the original one. This is
        # the guard that keeps `SpeakerBearingMap` free of its own guesses, and
        # `_colour_from_direction` now has three of them
        # (`direction`, `direction-placed`, `direction-contested`). A prefix
        # test so a fourth cannot silently escape it.
        if str(getattr(attribution, "reason", "") or "").startswith("direction"):
            return
        bearings.observe(word_id, speaker_id,
                         self._direction_for(word.start, word.end))

    def _direction_for(self, start: float, end: float) -> float | None:
        """The array's bearing while this span was spoken, or None.

        Absent whenever there is no array, no node, or no reading inside the
        span -- attribution then scores on voice evidence alone, exactly as it
        did before direction existed. The lookup is set on the tracker only
        when a node is attached, so a local mic never reaches this.

        Word times are stream-relative; bearings are recorded on the source
        clock, so `stream_base` is what makes the two comparable. Getting that
        wrong would score every word against a bearing from a different moment.
        """
        tracker = getattr(self, "speaker_tracker", None)
        lookup = getattr(tracker, "direction_for_span", None)
        if lookup is None:
            return None
        return lookup(self.stream_base + start, self.stream_base + end)

    def _word_event(self, word: HypothesisWord, audio: np.ndarray,
                    final: bool,
                    speaker: SpeakerAttribution | str | None = None,
                    word_id: str | None = None) -> dict:
        # Hypotheses are revised frequently. Re-running pitch analysis for the
        # same word on every decoder tick wastes enough CPU to create its own
        # backlog, and it also makes the displayed typography wobble.
        #
        # THE INVARIANT: a word that has been shown must stop changing. Every
        # field below is therefore frozen per time SLOT, not per text — the
        # verifier respells words, so a text-keyed cache misses precisely when
        # it matters. Measured before this: 42/59 slots reported a different
        # `loudness` at verification than at commit, and 13/59 a different
        # `pitch_hz`, which the renderer faithfully turned into resizing text.
        slot = round(word.start * 20)
        slot_key = ("§slot", self.utterance, slot)
        frozen = self.prosody_cache.get(slot_key)
        if frozen is None:
            # Endpoint retiming can move a word by a few tens of ms, which
            # would otherwise miss its own frozen entry and let it re-style.
            for delta in (-1, 1, -2, 2):
                frozen = self.prosody_cache.get(("§slot", self.utterance, slot + delta))
                if frozen is not None:
                    break
        delivery_cache = getattr(self, "delivery_cache", None)
        if delivery_cache is None:
            delivery_cache = self.delivery_cache = {}
        delivery = delivery_cache.get(slot_key)
        if delivery is None:
            for delta in (-1, 1, -2, 2):
                delivery = delivery_cache.get(
                    ("§slot", self.utterance, slot + delta)
                )
                if delivery is not None:
                    break
        if delivery is None:
            # Unlike cold-start loudness normalization, these exact-span
            # descriptors freeze at the first event. A later endpoint may have
            # more audio, but allowing it to restyle a visible word recreates
            # the late-motion failure this layer is designed to prevent.
            delivery = _word_delivery_features(word, audio, self.cfg)
            delivery_cache[slot_key] = delivery

        # CWI 2.3's contour, frozen on exactly the same terms as `delivery`:
        # once a word has been drawn, the shape of its letters must not change
        # underneath it because the verifier respelled it.
        envelope_cache = getattr(self, "envelope_cache", None)
        if envelope_cache is None:
            envelope_cache = self.envelope_cache = {}
        span_s = max(0.0, word.end - word.start)
        cached = envelope_cache.get(slot_key)
        if cached is None:
            # The neighbouring-slot search exists so endpoint RETIMING -- the
            # same word moving a few tens of ms -- keeps its frozen shape. It
            # must not let a DIFFERENT word inherit it: a slot is 50 ms, so two
            # short words land in adjacent slots easily. Measured, a 0.02 s
            # "You" borrowed the envelope of the 0.72 s "know" beside it and was
            # drawn with that word's contour. Matching the span rejects that
            # while still absorbing the retiming it is there for.
            for delta in (-1, 1, -2, 2):
                neighbour = envelope_cache.get(
                    ("§slot", self.utterance, slot + delta)
                )
                if neighbour is None:
                    continue
                other = neighbour["span_s"]
                if abs(other - span_s) <= 0.3 * max(other, span_s, 1e-6):
                    cached = neighbour
                    break
        if cached is None:
            cached = {
                "env": self._intonation_envelope(
                    word,
                    audio,
                    self.cfg.get("display", {}).get(
                        "intonation_envelope_samples", 8
                    ),
                ),
                "span_s": span_s,
            }
            envelope_cache[slot_key] = cached
        envelope = cached["env"]
        prosody_key = (word.text.casefold(), round(word.start * 100))
        features = self.prosody_cache.get(prosody_key)
        if frozen is not None:
            features = frozen[:3]
        elif features is None or final:
            # `final` re-measures legitimately: the audio buffer is longer by
            # then, so the word's own span is better covered.
            features = self._prosody(word, audio)
            self.prosody_cache[prosody_key] = features
        db, pitch_hz, voiced_frac = features
        # Vocal effort is frozen per SLOT for the same reason `delivery` is: a
        # word that has been shown must stop changing, and the verifier
        # respells words so a text-keyed cache misses exactly when it matters.
        effort_cfg = dict(self.cfg.get("live", {}).get("vocal_effort", {}) or {})
        effort_cache = getattr(self, "effort_cache", None)
        if effort_cache is None:
            effort_cache = self.effort_cache = {}
        effort_history = getattr(self, "effort_history", None)
        if effort_history is None:
            effort_history = self.effort_history = deque(maxlen=120)
        effort_bootstrap = getattr(self, "effort_bootstrap", None)
        if effort_bootstrap is None:
            effort_bootstrap = self.effort_bootstrap = {}
        effort = effort_cache.get(slot_key)
        if effort is None:
            for delta in (-1, 1, -2, 2):
                effort = effort_cache.get(("§slot", self.utterance, slot + delta))
                if effort is not None:
                    break
        if effort is None and effort_cfg.get("enabled", True):
            effort = self._vocal_effort(word, audio)
            if effort is not None:
                effort_cache[slot_key] = effort
        # The three raw cues this word contributes to the speaker's baseline.
        # Length is seconds per CHARACTER: raw duration ranks by syllable count.
        cue = (
            (
                float(effort),
                float(pitch_hz),
                max(0.0, float(word.end - word.start))
                / max(1, len(word.text.strip(" .,!?"))),
            )
            if effort is not None else None
        )
        syllables = self._syllable_stops(word)
        if final:
            self.db_history.append(db)
            if cue is not None:
                effort_history.append(cue)
        # SAME COLD START AS `db_bootstrap`, AND IT IS WHY "louder" NEVER MOVED.
        # `effort_history` takes FINAL words only; the PR film opens with a
        # single 24 s utterance, so on the whole demo section -- every word the
        # film itself captions, "louder" included -- the lift's own
        # `len(effort_history) >= 6` gate was false and the channel was simply
        # off. Non-final emissions carry the same measured cues, keyed by time
        # slot so a re-emitted hypothesis overwrites itself.
        if len(effort_history) < 6:
            if cue is not None:
                effort_bootstrap[slot_key] = cue
        elif effort_bootstrap:
            effort_bootstrap.clear()
        cfg_lo, cfg_hi = self.cfg["live"]["db_range"]
        med = float(np.median(self.db_history)) if self.db_history else 0.0
        # COLD-START BOOTSTRAP. `db_history` holds FINAL words only, and a
        # long first utterance finalizes all at once at its endpoint — on the
        # PR film's 24 s opening monologue that left ~60 words cueing against
        # the static `db_range` fallback, where the narration saturated at
        # loudness ≈ 1.0 and every ordinary word rendered at the crest clamp.
        # Non-final emissions carry the same measured dB, so they can
        # calibrate the window; keyed by the word's time SLOT so a pending
        # word re-emitted on every hypothesis overwrites its own entry
        # instead of stuffing the window with copies of itself.
        db_bootstrap = getattr(self, "db_bootstrap", None)
        if db_bootstrap is None:
            db_bootstrap = self.db_bootstrap = {}
        if len(self.db_history) < 6:
            db_bootstrap[slot_key] = db
        elif db_bootstrap:
            # Durable calibration has taken over; the hand-off is explicit.
            db_bootstrap.clear()
        percentiles = self.cfg["live"].get("db_percentiles", [15, 95])
        min_span = float(self.cfg["live"].get("db_min_span", 18.0))

        def _percentile_window(values) -> tuple[float, float]:
            return self._percentile_window(values, percentiles, min_span)

        if len(self.db_history) >= 6:
            # Percentiles of this speaker's own words, not a fixed offset from
            # the median. A `median - 5 dB` floor was catastrophically tight:
            # ordinary connected speech runs ~26 dB below its median at the
            # 10th percentile, so 35% of words clipped to loudness 0 and
            # rendered at CWI whisper size (3%) beside normal ones. Clipping
            # happens BEFORE the display smoothing, so no amount of hysteresis
            # could repair it.
            lo_db, hi_db = _percentile_window(self.db_history)
            norm_med = med
        elif len(db_bootstrap) >= 6:
            # Same maths over the bootstrap slots. The hypothesis grows at
            # speech rate, so this branch takes over by roughly the seventh
            # word; the first six cue against the config range but are
            # re-emitted by later hypotheses inside the playhead delay, so
            # their axes correct before their colour turn.
            bootstrap_values = list(db_bootstrap.values())
            lo_db, hi_db = _percentile_window(bootstrap_values)
            norm_med = float(np.median(bootstrap_values))
        else:
            lo_db, hi_db = cfg_lo, cfg_hi
            norm_med = med

        # Haptic salience flags. DHH haptics research (Haptic-Captioning
        # CHI'23; Tactile Emotions CHI'25) found continuous vibration
        # distracting: actuate selectively on speaker changes and strong
        # prosody, with a user-adjustable threshold. Durable words carry the
        # selection so the future haptic module never needs analysis code.
        salience: dict = {}
        attribution = self._attribution(speaker, final=final)
        stable_attribution = attribution.status in {"stable", "corrected"}
        self._record_speaker_bearing(word_id, attribution, word)
        if final and stable_attribution and attribution.speaker_id is not None:
            spoken_by = attribution.speaker_id
            if (self._last_final_speaker is not None
                    and spoken_by != self._last_final_speaker):
                salience["speaker_change"] = True
            self._last_final_speaker = spoken_by
        if final:
            # Emphasis is acoustic salience, independent of whether speaker
            # identity has stabilized. Only the speaker-change flag is gated.
            emphasis_db = self.cfg.get("haptics", {}).get("emphasis_db", 6.0)
            if len(self.db_history) >= 6 and db - med >= emphasis_db:
                salience["emphasis"] = True
        # The bearing this word was spoken from, when a mic array measured one.
        # It rides on the WORD rather than streaming to the motors: driving
        # haptics from raw direction is exactly the continuous vibration
        # Tactile Emotions (CHI '25) measured as distracting, while a bearing
        # attached to a speaker change is one event. Absent with no array --
        # never fabricate direction.
        # getattr, not attribute access: the tests build captioners with
        # `__new__` to skip loading a recognizer, so anything read here has to
        # tolerate an instance that never ran `__init__`.
        if final:
            # A final ASR word commonly arrives seconds after it was spoken.
            # Its cue must retain the DoA observed *during that word's span*,
            # not look up the newest reading now and lose it to the short TTL.
            # The newest reading remains a fallback for array integrations
            # that do not expose span history.
            bearing = self._direction_for(word.start, word.end)
            direction_source = getattr(self, "direction_source", None)
            if bearing is None and direction_source is not None:
                bearing = direction_source()
            if bearing is not None:
                salience["direction_deg"] = round(float(bearing), 1)
        # Pivot the scale on the speaker's median so it lands on the CWI
        # baseline (2.3.5: normal speaking volume = 5% of frame height).
        # A plain lo..hi normalization put the MEDIAN word at mid-scale, which
        # renders at 6.5% — every ordinary word reading as slightly shouted,
        # and the size ratio blowing out to 3x on screen.
        mapping = self.cfg["mapping"]["loudness_to"]
        pivot = ((mapping.get("baseline", 5) - mapping["min"]) /
                 max(1e-6, mapping["max"] - mapping["min"]))
        calibrated = (
            (len(self.db_history) >= 6 or len(db_bootstrap) >= 6)
            and lo_db < norm_med < hi_db
        )

        def _normalize_db(value: float) -> float:
            """dB -> 0..1 on the 2.3.5-pivoted scale the client's axes expect."""
            if calibrated:
                if value <= norm_med:
                    return float(np.clip(
                        pivot * (value - lo_db) / max(1e-6, norm_med - lo_db),
                        0.0, 1.0,
                    ))
                return float(np.clip(
                    pivot
                    + (1 - pivot) * (value - norm_med)
                    / max(1e-6, hi_db - norm_med),
                    0.0, 1.0,
                ))
            # UNMEASURED IS NEUTRAL, NOT LOUD. Before the bootstrap has six
            # words there is no speaker scale to place this word on, and the
            # raw config-range normalization rendered the session's first
            # words at the crest clamp (the PR film's narration measured
            # loudness ≈ 1.0 against it). The CWI answer for an unmeasured
            # channel is the 2.3.5 baseline — the same reasoning that renders
            # an unattributed word neutral — and the endpoint's one allowed
            # correction restores any genuine emphasis.
            return pivot

        loudness = _normalize_db(db)

        # NOTE: the CWI 2.3.5 vocal-effort lift is applied AFTER the frozen
        # restore below, not here. It has its own per-slot freeze, because the
        # level freeze happens on the first CALL -- when the word is still at
        # the edge of the audio buffer and effort cannot be measured -- and
        # `frozen` is also inherited from a neighbouring slot by the retiming
        # lookup. Applying the lift before that restore meant every
        # drill-sergeant word computed a positive excess (0.06..0.46) and then
        # had it thrown away; MEASURED lift on the shout was 0.000 across the
        # whole section, twice, before the order was fixed.

        # CWI 2.3.5 ASKS FOR VOLUME AND `db` ONLY MEASURES LEVEL. On mastered,
        # AGC'd or auto-levelled audio those are different quantities: the PR
        # film's drill sergeant shouts 1.6 dB QUIETER than the calm narration,
        # and ranking his words by level scores AUC 0.367 -- worse than chance.
        # Effort (see `_vocal_effort`) survives that normalization, so it is
        # added to the level channel as a one-sided lift.
        #
        # ONE-SIDED, AND ON ITS OWN PERCENTILE WINDOW. Low effort already
        # renders quiet through `db`, so subtracting would penalize a soft
        # voice twice; and normalizing effort separately is what keeps this
        # from costing the level channel its range -- MEASURED, folding effort
        # into `db` before the percentile window flattened the narration's own
        # intonation from p90 1.25 to 1.00, while this leaves it at 1.25
        # untouched and still lifts the shout from p90 1.06 to 1.41.
        def _effort_lift() -> float:
            """The one-sided 0..1 addition this word earns for vocal effort."""
            gain = float(effort_cfg.get("gain", 0.5))
            settled = list(effort_history)
            baseline_cues = settled or list(effort_bootstrap.values())
            if not (effort_cfg.get("enabled", True) and cue is not None
                    and gain > 0 and len(baseline_cues) >= 6):
                return 0.0
            # AND A WORD THAT IS MEASURABLY QUIET IS NOT A SHOUT, WHATEVER ITS
            # SPECTRUM SAYS (2026-08-03). Effort exists because a mastered or
            # AGC'd shout need not be louder on the meter -- but that argument
            # only ever justified lifting a word AT OR ABOVE the speaker's own
            # level, never one well below it. Tilt is energy-weighted and still
            # tracks phonemes: the loudest frames of "softer" are its /s/, so
            # the film's canonical QUIET word earned a full effort lift and
            # rendered at 1.27x where the reference draws it SMALLER than
            # normal. `_span_db` now separates it from "louder" by 12.2 dB on
            # plain level, so the level channel can be trusted to veto this
            # one. Below the pivot the lift fades out rather than switching, so
            # two adjacent words either side of the median cannot render at
            # obviously different sizes -- the same continuity `voiceScale`'s
            # deadband edge is tested for.
            if loudness < pivot:
                fade = max(0.0, loudness / max(1e-6, pivot))
                if fade <= 0.0:
                    return 0.0
                gain *= fade ** 2
            # The speaker's own pitch and word length, which `_prominence`
            # measures this word's excursion against. Derived from whichever
            # baseline is available, so the composite is the same quantity
            # during the bootstrap and after it.
            baseline = (
                float(np.median([c[1] for c in baseline_cues if c[1] > 0] or [0.0])),
                float(np.median([c[2] for c in baseline_cues if c[2] > 0] or [0.0])),
            )
            # Published as `pitch_register_hz`: 2.3.7's domain is a VOICE, and
            # the renderer needs the speaker's register to tell a light voice
            # apart from a pushed one.
            self._pitch_register_hz = baseline[0]
            # THE SMOOTHING WINDOW CANNOT READ `effort_history`: that holds
            # FINAL words only, and a word is first seen -- and its lift first
            # frozen -- while its own utterance is still open. During the
            # shout the newest finals are therefore still the CALM narration
            # before it, so the mean was dragged back to baseline and the lift
            # computed to exactly 0.0 for every shouted word, then cached.
            # `effort_recent` is keyed by time slot so a re-emitted hypothesis
            # overwrites itself instead of stuffing the window with copies --
            # the same trick `db_bootstrap` uses. It holds RAW cues, because
            # the blend in `_prominence` is what has to see the sequence.
            recent_map = getattr(self, "effort_recent", None)
            if recent_map is None:
                recent_map = self.effort_recent = {}
            here = (self.utterance, slot)
            recent_map[here] = cue
            if len(recent_map) > 512:
                for key in sorted(recent_map)[:256]:
                    recent_map.pop(key, None)
            recent = [recent_map[k] for k in sorted(k for k in recent_map if k <= here)]
            span = max(1, int(effort_cfg.get("smoothing_words", 4)))
            score = self._prominence(recent[-span:], baseline, effort_cfg)[-1]
            # The BASELINE stays on settled history on purpose: a shout should
            # be measured against the calm speech around it, not against
            # itself.
            history = self._prominence(baseline_cues, baseline, effort_cfg)
            lo_e, hi_e = self._percentile_window(
                history,
                effort_cfg.get("percentiles", [10, 95]),
                float(effort_cfg.get("min_span", 12.0)),
            )
            med_e = float(np.median(history))
            if not (lo_e < med_e < hi_e and score > med_e):
                return 0.0
            # CLIPPED AT 1, exactly as `_normalize_db` clips the level channel.
            # Unclipped, a word far above the speaker's p95 produced a `level`
            # of 2 or 3 and a lift big enough to slam `loudness` into its own
            # ceiling: MEASURED, 10% of the film's words pinned at the crest
            # clamp and 19.4% rendered above 1.50x against the film's 2.8%.
            # With the clip the largest possible lift is a config decision
            # (`gain` x 0.514) instead of an artefact of the window.
            level = float(np.clip(
                pivot + (1 - pivot) * (score - med_e) / max(1e-6, hi_e - med_e),
                0.0, 1.0,
            ))
            # A DEADBAND, for the same reason `voice_scale_deadband` has one.
            # Without it every word above the running median lifts a little,
            # and energy weighting does not fully remove phoneme content from
            # tilt -- MEASURED, 34% of ordinary narration words lifted and
            # sibilant-heavy ones ("synchronized", "so") gained +0.30, which is
            # spelling driving type size.
            #
            # RE-SPANNED, NOT SUBTRACTED (2026-08-02). Subtracting the band
            # from the lift makes the band steal from the TOP: widening it to
            # hold ordinary words still also caps the loudest word, so the
            # drill sergeant could not be made stronger without letting the
            # narration move too. That is the same anti-pattern
            # `voice_scale_response_quiet` already hit -- "weakening the
            # response made the whole channel disappear instead of stopping
            # ordinary words from moving". `voiceScale` re-spans for exactly
            # this reason, so do the same here: the band decides WHO moves and
            # `gain` decides how far, independently.
            dead = float(np.clip(effort_cfg.get("deadband", 0.34), 0.0, 0.95))
            excess = (level - pivot) / max(1e-6, 1 - pivot)
            if excess <= dead:
                return 0.0
            return gain * (1 - pivot) * (excess - dead) / (1 - dead)

        # Frozen per slot on the same terms as everything else here: a word
        # that has been shown must stop changing.
        lift_cache = getattr(self, "effort_lift_cache", None)
        if lift_cache is None:
            lift_cache = self.effort_lift_cache = {}
        lift = lift_cache.get(slot_key)
        if lift is None:
            for delta in (-1, 1, -2, 2):
                lift = lift_cache.get(("§slot", self.utterance, slot + delta))
                if lift is not None:
                    break

        bright_lo, bright_hi = self.cfg.get("display", {}).get(
            "intent_circle_brightness_hz", [500, 3500]
        )
        if frozen is not None:
            loudness = frozen[3]
        elif len(self.db_history) >= 6:
            # Freeze only once the scale is calibrated. The first few words of
            # a session normalize against the cold-start `db_range` and are
            # genuinely wrong, so they are allowed exactly one correction.
            self.prosody_cache[slot_key] = (db, pitch_hz, voiced_frac, loudness)

        # THE EFFORT LIFT GOES ON TOP OF THE RESTORED LEVEL, NOT BEFORE IT.
        # `frozen` above is written on the first CALL for a slot -- when the
        # word still sits at the edge of the audio buffer and `_vocal_effort`
        # has under 30 ms to work with -- and it is also inherited from a
        # NEIGHBOURING slot by the retiming lookup. Both mean an effort lift
        # computed before that restore is silently discarded: MEASURED, every
        # drill-sergeant word produced a positive excess (0.06..0.46) and
        # shipped 0.000 of it, twice, until the order was fixed. Freezing the
        # lift separately keeps the "a shown word stops changing" invariant.
        if lift is None:
            lift = _effort_lift()
            # ...but freeze only once the baseline is SETTLED, exactly as the
            # level channel freezes only once `db_history` has six words. A
            # bootstrap baseline is still moving, and a word cued against three
            # slots must be allowed to correct before its colour turn.
            if cue is not None and len(effort_history) >= 6:
                lift_cache[slot_key] = lift
        if lift:
            loudness = float(np.clip(loudness + lift, 0.0, 1.0))
        word_id = word_id or self._word_id(word)
        event_t = round(self.stream_base + word.start, 3)
        event_start = round(word.start, 3)
        event_end = round(word.end, 3)
        revision_fields = self._word_revision_fields(
            word_id,
            text=word.text,
            t=event_t,
            start=event_start,
            end=event_end,
        )
        speaker_fields = attribution.event_fields(
            include_debug=bool(
                self.cfg.get("live", {})
                .get("speaker_attribution", {})
                .get("debug", False)
            )
        )
        event = {
            "type": "word",
            "final": final,
            "utterance": self.utterance,
            "word_id": word_id,
            "text": word.text,
            "t": event_t,
            "start": event_start,
            "end": event_end,
            **revision_fields,
            # AN UNDECIDED WORD PUBLISHES NO SPEAKER. This used to read
            # `or self.speaker`, defaulting every unattributed word to "S1",
            # and the comment justifying it said unknown assignments "are
            # rendered neutral because speaker_status is authoritative". That
            # was true when it was written and stopped being true on
            # 2026-08-02, when `speakerStatus` began promoting a
            # speaker-carrying unknown to `provisional` -- itself a correct
            # fix, for durable words that were stuck grey forever. The two
            # composed into a false claim: MEASURED on the PR film, the
            # default was right 17 times out of 43, which is just how often
            # the narrator happens to be talking (79/172), i.e. it carried no
            # information at all while painting 26 words in the wrong
            # speaker's colour.
            # `speakerColor(null)` is already `--caption-unknown` and the
            # legacy renderer already keys on `speaker_known`, so both
            # renderers draw this correctly; only the id was dishonest.
            "speaker": speaker_fields.pop("speaker"),
            "speaker_known": attribution.speaker_id is not None,
            **speaker_fields,
            "loudness": round(loudness, 4),
            "pitch": 0.5,
            "loudness_db": round(db, 2),
            "pitch_hz": round(pitch_hz, 2),
            # The SPEAKER's running median F0, not this word's. See
            # `voiceTone` -- 2.3.7 describes who is speaking, so the register
            # half of the weight axis must be measured per voice.
            "pitch_register_hz": round(
                float(getattr(self, "_pitch_register_hz", 0.0) or 0.0), 2),
            "voiced_frac": round(voiced_frac, 3),
            **delivery,
            "conf": round(word.conf, 3),
            "conf_available": word.conf_available,
            **({"syllables": syllables} if syllables else {}),
            # CWI 2.3 per CHARACTER (p.34/p.38/p.40). Absent for spans too short
            # to have a measurable shape; the client then uses the word-level
            # values for every character rather than inventing a contour.
            **({
                # Through the SAME pivot as `loudness` above, so the client's
                # 2.3.5 anchor means one thing. Shape is frozen with the word;
                # only the absolute level can still shift during the first few
                # words, exactly as the word-level value does while the speaker's
                # percentiles calibrate.
                "env_loudness": [
                    round(_normalize_db(v), 4) for v in envelope["loudness"]
                ],
                "env_pitch": envelope["pitch"],
                # Spectral centroid -> the same 0..1 brightness the width axis
                # already consumes as `delivery_texture` (0 = warm/low
                # harmonics = wider, 1 = bright/high harmonics = condensed), on
                # the configured brightness range so both agree.
                "env_texture": [
                    round(float(np.clip(
                        (v - bright_lo) / max(1e-6, bright_hi - bright_lo), 0.0, 1.0
                    )), 4)
                    for v in envelope["texture"]
                ],
            } if envelope else {}),
            **salience,
        }
        if final:
            final_events = getattr(self, "_final_word_events", None)
            if final_events is None:
                final_events = self._final_word_events = {}
            final_events[word_id] = dict(event)
        return event

    def _process_result(self, endpoint: bool = False):
        audio = self.audio
        result = self.recognizer.get_result_all(self.stream)
        current = hypothesis_words(result, len(audio) / SR)
        if self.allowed_scripts:
            # ONE choke point, before ids are assigned, so commits, partials
            # and the verifier all see the same filtered list and a dropped
            # word can never have claimed an id first.
            current = [word for word in current
                       if word_script_allowed(word.text, self.allowed_scripts)]
        current_word_ids = self._word_ids_for(current)

        if self.draft_only:
            # The low-latency recognizer is a revisable white read-ahead layer.
            # Only the accurate recognizer is allowed to create durable words.
            commit_to = 0
        elif endpoint:
            # A streaming transducer only produces text when it has acoustic
            # evidence; an extra RMS gate deleted valid quiet/program audio.
            commit_to = len(current)
        else:
            stable = common_prefix_len(self.previous, current)
            # Hold the newest word: without a following word its end time and
            # spelling are the parts most likely to change.
            # The hold-back is deliberate and measured: releasing the trailing
            # word early (tried with a 0.6 s trailing-silence rule) saved no
            # time — it fired at the same moment as the endpoint — and it
            # committed truncated spellings, because a trailing word's pieces
            # are not fully flushed until the endpoint. A lone word reaches
            # the screen through the fast-mode white read-ahead instead.
            commit_to = min(stable, max(0, len(current) - 1))
            commit_to = max(len(self.committed), commit_to)

        committed_now = current[len(self.committed):commit_to]

        def provisional_speaker(word, word_id):
            # Mid-stream classification only: it never invents a speaker or
            # moves a centroid, because a window here can straddle a turn
            # boundary. The endpoint pass owns learning; it corrects these.
            if self.speaker_tracker is None:
                return None
            bearing = self._direction_for(word.start, word.end)
            result = self.speaker_tracker.classify_span(
                audio,
                word.start,
                word.end,
                observation_key=word_id,
                timestamp_offset=self.stream_base,
                direction_estimate=bearing,
            )
            result = self._colour_from_direction(result, bearing)
            if self._readahead_stability and result is not None:
                # Suppress a mid-stream pick that is a NEW specific speaker: keep
                # only a `stable`/`corrected` decision or a continuation of the
                # active speaker; anything else waits grey for the endpoint.
                sid = getattr(result, "speaker_id", None)
                status = getattr(result, "status", None)
                active = getattr(self.speaker_tracker,
                                 "last_confidently_active_speaker", None)
                if (sid is not None
                        and status not in ("stable", "corrected")
                        and sid != active):
                    return None
            return result

        # WHETHER THIS UTTERANCE GETS VERIFIED, DECIDED ONCE.
        # It selects the emission path as well as the verification pass, and
        # those two MUST agree: when a verifier exists, durable words are
        # emitted only by the verification block below, so an utterance that
        # skips verification without also taking the no-verifier path here
        # produces no final words at all. Measured when this was a bare `skip`:
        # Korean on the bilingual profile returned 120/120 clips SILENT.
        verifying = self.verifier is not None and not self._verifier_skips(current)

        if not verifying:
            if endpoint and current and self.speaker_tracker is not None:
                # Korean deliberately has no weaker offline text verifier, but
                # speaker attribution still needs one full-utterance endpoint
                # pass. Re-emit the same word IDs so early live words remain
                # visible and their provisional identities settle in place.
                already_final = set(self._final_word_events)
                speakers = self.speaker_tracker.label_words(
                    audio,
                    current,
                    observation_keys=current_word_ids,
                    timestamp_offset=self.stream_base,
                    direction_estimate=self._direction_for(
                        current[0].start, current[-1].end),
                )
                for index, word in enumerate(current):
                    event = self._word_event(
                        word,
                        audio,
                        final=True,
                        speaker=speakers[index],
                        word_id=current_word_ids[index],
                    )
                    if current_word_ids[index] in already_final:
                        # A historical identity correction updates colour and
                        # paragraph ownership; it must not replay haptics.
                        event.pop("speaker_change", None)
                        event.pop("emphasis", None)
                        event["correction"] = True
                    yield event
            else:
                for offset, word in enumerate(
                    committed_now, start=len(self.committed)
                ):
                    yield self._word_event(word, audio, final=True,
                                           speaker=provisional_speaker(
                                               word, current_word_ids[offset]
                                           ), word_id=current_word_ids[offset])
        elif not endpoint:
            # Stable streaming words may be rendered immediately, but remain
            # provisional until the whole-phrase verifier sees the endpoint.
            for offset, word in enumerate(
                committed_now, start=len(self.committed)
            ):
                event = self._word_event(word, audio, final=False,
                                         speaker=provisional_speaker(
                                             word, current_word_ids[offset]
                                         ), word_id=current_word_ids[offset])
                event.update(type="commit", provisional=True, verified=False)
                yield event
        if commit_to > len(self.committed):
            self.committed = current[:commit_to]

        if verifying and endpoint and current:
            verified_text = self.verifier.transcribe(audio)
            verified = conservative_verified_words(
                current,
                verified_text,
                audio=audio,
            )
            verified = repair_verified_tail_timing(verified, audio)
            word_ids = self._word_ids_for(verified)
            speakers = (
                self.speaker_tracker.label_words(
                    audio,
                    verified,
                    observation_keys=word_ids,
                    timestamp_offset=self.stream_base,
                    direction_estimate=self._direction_for(
                        verified[0].start, verified[-1].end),
                )
                if self.speaker_tracker is not None and verified else None
            )
            final_events = [self._word_event(
                                word, audio, final=True,
                                speaker=speakers[i] if speakers else None,
                                word_id=word_ids[i],
                            )
                            for i, word in enumerate(verified)]
            for event in final_events:
                event.update(
                    verified=True,
                    provisional=False,
                    speaker_known=event.get("speaker_status", "stable") != "unknown",
                )
            yield {
                "type": "verification",
                "utterance": self.utterance,
                "text": verified_text,
                "words": final_events,
            }
            yield from final_events
            # A later clean observation may make an earlier provisional
            # profile stable. Re-emit the complete durable word with the same
            # ``word_id`` so replay, logs, and the page can update in place.
            if self.speaker_tracker is not None:
                current_ids = {event["word_id"] for event in final_events}
                for word_id, attribution in self.speaker_tracker.drain_revisions():
                    if word_id in current_ids:
                        continue
                    original = self._final_word_events.get(word_id)
                    if original is None:
                        continue
                    fields = attribution.event_fields(
                        include_debug=bool(
                            self.cfg.get("live", {})
                            .get("speaker_attribution", {})
                            .get("debug", False)
                        )
                    )
                    fields["speaker"] = fields.get("speaker") or original["speaker"]
                    revised = {
                        **original,
                        **fields,
                        "type": "word",
                        "final": True,
                        "verified": True,
                        "correction": True,
                    }
                    # Historical corrections update state but never replay a
                    # delayed haptic pulse.
                    revised.pop("speaker_change", None)
                    revised.pop("emphasis", None)
                    self._final_word_events[word_id] = revised
                    yield revised

        pending = current[len(self.committed):]
        partial_start = len(self.committed)
        partial_events = [
            self._word_event(
                word, audio, final=False,
                word_id=current_word_ids[partial_start + index],
                # THE READ-AHEAD LANE HAD NO ATTRIBUTION AT ALL, AND IT IS THE
                # FIRST THING THE VIEWER SEES. Every word reaches the stage as
                # a hypothesis first; this call site passed no speaker, so
                # `_attribution` returned unknown and the publication line
                # below defaulted it to `self.speaker` -- "S1". MEASURED on the
                # PR film, 160 of 172 words first painted as S1/unknown and 83
                # of them were somebody else, so 45.9% of words were still the
                # WRONG colour when the playhead reached them, narrator yellow
                # over the drill sergeant's whole exchange.
                # Sortformer already knows: it runs continuously at 1.04 s
                # latency, which is well inside the 1.75 s the playhead trails
                # by, and asking it costs nothing.
                speaker=self._read_ahead_speaker(word),
            )
            for index, word in enumerate(pending)
        ]
        partial_key = tuple((w["text"], w["start"], w["end"]) for w in partial_events)
        if partial_key != self.last_partial_key or committed_now or endpoint:
            yield {
                "type": "hypothesis",
                "utterance": self.utterance,
                "endpoint": endpoint,
                "words": partial_events,
            }
            self.last_partial_key = partial_key
        self.previous = current

    def accept(self, item: np.ndarray | AudioChunk):
        if isinstance(item, AudioChunk):
            if item.discontinuity:
                source_sample = int(round(item.source_start * SR))
                self.total_samples = source_sample
                self._reset_stream(base=item.source_start)
                # Clear stale white hypotheses immediately. Colored committed
                # words remain durable, but no delayed partial survives a jump.
                yield {
                    "type": "hypothesis",
                    "utterance": self.utterance,
                    "endpoint": True,
                    "resync": True,
                    "dropped_s": round(item.dropped_s, 3),
                    "words": [],
                }
            block = item.samples
            asr_block = item.recognizer_samples
        else:
            block = asr_block = item
        block = np.asarray(block, dtype=np.float32)
        asr_block = np.asarray(asr_block, dtype=np.float32)
        # Prosody reads audio_blocks, so it must see the true captured level;
        # only the recognizer gets the gained copy.
        self.audio_blocks.append(block)
        self.total_samples += len(block)
        self.stream.accept_waveform(SR, asr_block)
        decoded = False
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
            decoded = True
        if decoded:
            yield from self._process_result(endpoint=False)
        if self.recognizer.is_endpoint(self.stream):
            yield from self._process_result(endpoint=True)
            self._reset_stream()

    def finish(self):
        # Flush the neural encoder and commit the final word even if the file
        # ends without enough trailing silence for endpoint detection.
        tail = np.zeros(int(0.8 * SR), dtype=np.float32)
        self.audio_blocks.append(tail)
        self.total_samples += len(tail)
        self.stream.accept_waveform(SR, tail)
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        yield from self._process_result(endpoint=True)

    def _new_stream(self):
        """Open a decoder stream, conditioning the language when the model takes one.

        MULTILINGUAL NEMOTRON CARRIES THE LANGUAGE PER STREAM, NOT PER
        RECOGNIZER. Its encoder takes a 6th input (`prompt_index`), and
        sherpa-onnx exposes it as `stream.set_option("language", ...)` --
        which is why `OnlineRecognizer.from_transducer` has no language
        argument and why this has to be applied at EVERY stream creation,
        including the reset after each endpoint. A stream opened without it
        silently decodes under whatever the model defaults to.

        `auto` is the model's own language identification, which is the whole
        point of the bilingual profile. MEASURED by NVIDIA at the 1.12 s chunk:
        Korean CER 7.12% with the language named, 7.30% on auto -- so detection
        costs 0.18 points.

        Unset for the English and Korean profiles, which are single-language
        models with no such input; calling it there would be a no-op at best.
        """
        stream = self.recognizer.create_stream()
        language = str(
            (self.cfg.get("live", {}) or {}).get("streaming_language", "") or ""
        ).strip()
        if language and hasattr(stream, "set_option"):
            stream.set_option("language", language)
        return stream

    def _verifier_skips(self, words) -> bool:
        """Does this utterance belong to a language the verifier cannot read?

        THE ENDPOINT VERIFIER IS AN ENGLISH MODEL. `parakeet-unified-en-offline`
        is what the English profile gains its durable text from, and it is the
        lane the bilingual profile went without -- so `multi` was being compared
        against `en` with a whole correction stage missing, which is not a
        comparison between two sets of model weights at all.

        Giving `multi` that lane back is only safe if it never reads a Korean
        utterance. `conservative_verified_words` reconciles the stream against
        the verifier's text per word, so an English recognizer handed Hangul
        does not degrade the result politely -- it proposes a whole alternative
        utterance in the wrong script.

        The gate is the SCRIPT OF THE HYPOTHESIS, not the configured language:
        in a bilingual session the language is a per-utterance fact, and the
        stream has already decided it by the time an endpoint fires. Any Hangul
        at all disqualifies the utterance, because the verifier replaces the
        whole line rather than the Korean part of it.
        """

        if not self.verifier_latin_only:
            return False
        from .scoring import is_korean

        return any(is_korean(str(getattr(word, "text", "") or ""))
                   for word in words)

    def _reset_stream(self, base: float | None = None) -> None:
        self.stream_base = self.total_samples / SR if base is None else base
        self.stream = self._new_stream()
        self.audio_blocks = []
        self.previous = []
        self.committed = []
        self.last_partial_key = ()
        self.utterance += 1
        self.prosody_cache = {}
        self.delivery_cache = {}
        self._word_slots = []
        # Word ids include the utterance number, so completed per-channel
        # counters cannot be referenced by the next stream and need not grow
        # for the lifetime of a long-running caption session.
        self._word_revisions = {}


class DualStreamingCaptioner:
    """Publish accurate live words, with an optional lower-latency draft."""

    def __init__(self, draft_recognizer, accurate_recognizer, cfg: dict,
                 verifier: EndpointVerifier | None = None,
                 speaker_tracker: "SpeakerTracker | None" = None):
        self.draft = (
            StreamingCaptioner(draft_recognizer, cfg, draft_only=True)
            if draft_recognizer is not None else None
        )
        self.accurate = StreamingCaptioner(
            accurate_recognizer, cfg, verifier=verifier,
            speaker_tracker=speaker_tracker,
        )
        self.draft_words: list[dict] = []
        self.accurate_words: list[dict] = []
        self.cued_slots: set[tuple[int, str]] = set()
        self.last_merged_key: tuple = ()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr")
        self.closed = False

    @staticmethod
    def _absolute_end(word: dict) -> float:
        return word["t"] + word["end"] - word["start"]

    def _merged_hypothesis(self, endpoint: bool = False,
                           resync: bool = False, dropped_s: float = 0.0):
        frontier = self.accurate.stream_base
        if self.accurate.committed:
            frontier = self.accurate.stream_base + self.accurate.committed[-1].end

        # Source tags are display-only: the accurate stream's tail revises far
        # less than the 160 ms draft, so the page's `fast` mode can render
        # accurate partials as read-ahead without the draft's churn.
        merged = [{**word, "src": "accurate"} for word in self.accurate_words]
        for draft in self.draft_words:
            if draft["t"] < frontier - 0.04:
                continue
            # The two profiles endpoint independently. After one resets, the
            # same acoustic word can be timestamped up to one second apart
            # (e.g. accurate "finally" and draft "Finally"). Prefer the
            # accurate spelling instead of displaying a duplicate tail word.
            if any(
                draft["text"].casefold() == word["text"].casefold()
                and abs(draft["t"] - word["t"]) < 1.0
                for word in self.accurate_words
            ):
                continue
            if any(abs(draft["t"] - word["t"]) < 0.22 for word in self.accurate_words):
                continue
            merged.append({**draft, "src": "draft"})
        merged.sort(key=lambda word: (word["t"], word["start"]))
        key = tuple((word["text"], word["t"], word["end"]) for word in merged)
        if key == self.last_merged_key and not endpoint and not resync:
            return None
        self.last_merged_key = key
        event = {
            "type": "hypothesis",
            "utterance": self.accurate.utterance,
            "endpoint": endpoint,
            "words": merged,
        }
        if resync:
            event.update(resync=True, dropped_s=round(dropped_s, 3))
        return event

    def _provisional_cues(self, words: list[dict], utterance: int) -> list[dict]:
        """Emit accurate-profile words once for display-only color/pop timing."""

        cues = []
        for index, word in enumerate(words):
            # A transducer can emit several words on the same encoder frame.
            # A 50 ms timestamp key collapsed all of those into one cue, so
            # only the first word received its live colour update. Stable word
            # identity keeps every word independent; the indexed fallback is
            # only for legacy/custom recognizers that do not expose word_id.
            identity = str(
                word.get("word_id")
                or f"legacy:{index}:{round(word['t'] * 1000)}"
            )
            slot = (utterance, identity)
            if slot in self.cued_slots:
                continue
            self.cued_slots.add(slot)
            cue = dict(word)
            cue.update(type="cue", final=False, provisional=True,
                       utterance=utterance, src="accurate")
            cues.append(cue)
        return cues

    def _handle_draft_events(self, events):
        for event in events:
            if event["type"] != "hypothesis":
                continue
            self.draft_words = event["words"]
            merged = self._merged_hypothesis(
                # The draft endpoint is not authoritative; only the accurate
                # stream may declare a phrase locked.
                endpoint=False,
                resync=event.get("resync", False),
                dropped_s=event.get("dropped_s", 0.0),
            )
            if merged:
                yield merged

    def _handle_accurate_events(self, events):
        changed = False
        endpoint = False
        resync = False
        dropped_s = 0.0
        cues = []
        verifications = []
        durable = []
        for event in events:
            if event["type"] == "hypothesis":
                self.accurate_words = event["words"]
                cues.extend(self._provisional_cues(
                    event["words"], event["utterance"]
                ))
                endpoint = event.get("endpoint", False)
                resync = event.get("resync", False)
                dropped_s = event.get("dropped_s", 0.0)
                changed = True
            elif event["type"] == "verification":
                verified = dict(event)
                verified["words"] = [
                    {**word, "src": word.get("src", "accurate")}
                    for word in event.get("words", [])
                ]
                verifications.append(verified)
                changed = True
            else:
                accurate = dict(event)
                accurate.setdefault("src", "accurate")
                durable.append(accurate)
                changed = True
        # Endpoint verification must reach the browser before the following
        # empty hypothesis clears the last provisional word nodes.
        yield from verifications
        if changed:
            merged = self._merged_hypothesis(endpoint, resync, dropped_s)
            if merged:
                yield merged
        # Publish the display snapshot before its cues so the browser always
        # has a stable word node to animate. Durable words retain ASR order.
        yield from cues
        yield from durable

    def accept(self, item):
        # The ONNX calls release the GIL. When explicit readahead is enabled,
        # run both profiles concurrently and publish whichever completes first.
        futures = {
            self.executor.submit(lambda: list(self.accurate.accept(item))): "accurate",
        }
        if self.draft is not None:
            futures[
                self.executor.submit(lambda: list(self.draft.accept(item)))
            ] = "draft"
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda f: futures[f] != "accurate"):
                events = future.result()
                if futures[future] == "accurate":
                    yield from self._handle_accurate_events(events)
                else:
                    yield from self._handle_draft_events(events)

    def finish(self):
        draft_future = (
            self.executor.submit(lambda: list(self.draft.finish()))
            if self.draft is not None else None
        )
        accurate_future = self.executor.submit(lambda: list(self.accurate.finish()))
        if draft_future is not None:
            yield from self._handle_draft_events(draft_future.result())
        changed = False
        for event in accurate_future.result():
            if event["type"] == "hypothesis":
                self.accurate_words = event["words"]
                changed = True
            else:
                yield event
                changed = True
        if changed:
            merged = self._merged_hypothesis(endpoint=True)
            if merged:
                yield merged
        self.close()

    def close(self):
        if not self.closed:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.closed = True


def streaming_events(blocks, recognizer, cfg: dict, draft_recognizer=None,
                     verifier: EndpointVerifier | None = None,
                     gain: "InputGain | None" = None,
                     speaker_tracker: "SpeakerTracker | None" = None,
                     sound_detector=None, onset_detector=None):
    captioner = (
        DualStreamingCaptioner(
            draft_recognizer, recognizer, cfg, verifier=verifier,
            speaker_tracker=speaker_tracker,
        )
        if draft_recognizer is not None or verifier is not None
        else StreamingCaptioner(
            recognizer, cfg, verifier=verifier,
            speaker_tracker=speaker_tracker,
        )
    )
    gain = gain if gain is not None else InputGain(cfg)
    # One direction source feeds both lanes: the level meter (so the compass
    # tracks continuously) and the durable word (so a haptic cue knows where).
    # `DualStreamingCaptioner` composes rather than inherits, so the durable
    # lane is its inner `accurate` captioner -- the draft never emits finals.
    if getattr(gain, "direction_source", None) is not None:
        target = getattr(captioner, "accurate", captioner)
        target.direction_source = gain.direction_source
    # Sound tagging is useful context, but its AudioSet inference must not sit
    # in front of the speech recognizer. A dedicated serial worker preserves
    # detector state/order while the caption-critical path continues.
    sound_pool = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="sound")
        if sound_detector is not None else None
    )
    sound_futures: deque = deque()
    onset_pool = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="onset")
        if onset_detector is not None else None
    )
    onset_futures: deque = deque()

    def drain_sound(wait_for_all: bool = False):
        while sound_futures and (wait_for_all or sound_futures[0].done()):
            yield from sound_futures.popleft().result()

    def drain_onset(wait_for_all: bool = False):
        while onset_futures and (wait_for_all or onset_futures[0].done()):
            yield from onset_futures.popleft().result()

    def current_utterance() -> int:
        if isinstance(captioner, DualStreamingCaptioner):
            return captioner.accurate.utterance
        return captioner.utterance

    level_period_s = float(
        cfg.get("live", {}).get("input_gain", {}).get("level_event_period_s", 0.12)
    )
    next_level_at = 0.0
    try:
        for block in blocks:
            if isinstance(block, AudioChunk):
                block = gain.process(block)
                if (
                    speaker_tracker is not None
                    and hasattr(speaker_tracker, "feed")
                ):
                    # Sortformer runs in its own native Core ML process. Feed
                    # it before ASR so its 1.04 s look-ahead and Nemotron's
                    # 1.12 s acoustic context advance together.
                    speaker_tracker.feed(
                        block.samples,
                        source_start=block.source_start,
                        discontinuity=block.discontinuity,
                    )
                if block.source_start >= next_level_at:
                    next_level_at = block.source_start + level_period_s
                    yield gain.level_event(block.source_start)
                if sound_pool is not None:
                    # TRUE captured level (block.samples), never the gained ASR
                    # copy — the acoustic scene must not be speech-normalized.
                    samples = block.samples
                    source_end = block.source_end
                    sound_futures.append(sound_pool.submit(
                        lambda audio=samples, t=source_end: list(
                            sound_detector.feed(audio, t)
                        )
                    ))
                    yield from drain_sound()
                if onset_pool is not None:
                    # The onset model receives the same gained copy as ASR,
                    # never the true-level prosody signal. It runs in its own
                    # serial worker so a 40 ms phone probe cannot sit in front
                    # of either Nemotron stream.
                    onset_futures.append(onset_pool.submit(
                        onset_detector.feed,
                        block.recognizer_samples,
                        block.source_start,
                        current_utterance(),
                        discontinuity=block.discontinuity,
                    ))
                    yield from drain_onset()
            yield from captioner.accept(block)
            yield from drain_sound()
            yield from drain_onset()
        if (
            speaker_tracker is not None
            and hasattr(speaker_tracker, "finish")
        ):
            # Flush the native right-context preview before the ASR's own
            # end-of-file flush projects final speaker labels onto words.
            speaker_tracker.finish()
        yield from captioner.finish()
        if sound_detector is not None:
            yield from drain_sound(wait_for_all=True)
            yield from sound_detector.finish()
        if onset_detector is not None:
            yield from drain_onset(wait_for_all=True)
    finally:
        if isinstance(captioner, DualStreamingCaptioner):
            captioner.close()
        if sound_pool is not None:
            sound_pool.shutdown(wait=True, cancel_futures=True)
        if onset_pool is not None:
            onset_pool.shutdown(wait=True, cancel_futures=True)
        if (
            speaker_tracker is not None
            and hasattr(speaker_tracker, "close")
        ):
            speaker_tracker.close()
        if speaker_tracker is not None and speaker_tracker.debug:
            print("[speaker] summary " + json.dumps(
                speaker_tracker.metrics(), ensure_ascii=False
            ))


def load_streaming_recognizer(cfg: dict, model_dir: str | Path | None = None):
    """Load the configured local streaming model with no network fallback."""

    import sherpa_onnx

    model_dir = Path(model_dir or cfg["live"]["streaming_model_dir"])
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent.parent / model_dir
    file_cfg = cfg["live"].get("streaming_files", {})
    files = {
        "tokens": model_dir / file_cfg.get("tokens", "tokens.txt"),
        "encoder": model_dir / file_cfg.get("encoder", "encoder.int8.onnx"),
        "decoder": model_dir / file_cfg.get("decoder", "decoder.int8.onnx"),
        "joiner": model_dir / file_cfg.get("joiner", "joiner.int8.onnx"),
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        fetch_flag = (
            " --korean-only"
            if cfg.get("live", {}).get("lang") == "ko"
            else ""
        )
        raise SystemExit(
            "streaming model is missing — run: "
            f".venv/bin/python scripts/fetch_streaming_model.py{fetch_flag}\n  "
            + "\n  ".join(missing)
        )
    live_cfg = cfg["live"]
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        **{name: str(path) for name, path in files.items()},
        num_threads=live_cfg.get("num_threads", 2),
        provider="cpu",
        enable_endpoint_detection=True,
        # Rule 1 applies before any speech has been decoded. Keep it generous
        # so the recognizer is not reset on the very block where speech begins.
        rule1_min_trailing_silence=max(2.4, live_cfg.get("endpoint_silence_s", 0.8)),
        rule2_min_trailing_silence=live_cfg.get("endpoint_silence_s", 0.8),
        rule3_min_utterance_length=live_cfg.get("endpoint_max_s", 12.0),
        decoding_method=live_cfg.get("decoding_method", "greedy_search"),
        max_active_paths=live_cfg.get("streaming_max_active_paths", 4),
        # RECALL KNOB. Subtracted from the blank logit at every frame, so a
        # positive value makes the transducer less willing to emit nothing.
        # This buys COVERAGE with insertions, which is the trade this project
        # wants on the stage: a wrong word is a caption, a dropped word is a
        # hole the viewer cannot even see. Default 0.0 = stock behaviour.
        blank_penalty=float(live_cfg.get("blank_penalty", 0.0)),
    )


def load_speaker_tracker(cfg: dict) -> SpeakerTracker | None:
    """Build the live diarization tracker, or None if disabled/missing.

    Absence degrades to single-speaker S1 rather than failing: attribution is
    an enhancement, and live mode must still run before the one-time model
    download has happened. On Apple Silicon, ``auto`` prefers the native
    Streaming Sortformer helper and retains the embedding tracker as a quiet-
    speech/>4-speaker fallback.
    """

    dia = dict(cfg.get("live", {}).get("diarization", {}) or {})
    backend = str(dia.get("backend", "auto")).casefold()
    if not dia.get("enabled", False) or backend == "off":
        return None
    model = Path(dia.get(
        "model",
        "assets/speaker-embedding-en/"
        "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
    ))
    if not model.is_absolute():
        model = Path(__file__).resolve().parent.parent / model
    if not model.is_file():
        # Keep an existing install working: fall back to whatever speaker model
        # is already downloaded rather than silently dropping to one speaker
        # until the user re-fetches.
        alt = sorted(model.parent.glob("*.onnx")) if model.parent.is_dir() else []
        if alt:
            print(f"[live] {model.name} not found — using {alt[0].name}; run "
                  "scripts/fetch_streaming_model.py for the stronger default")
            model = alt[0]
        else:
            print("[live] speaker model missing — run scripts/fetch_streaming_model.py"
                  " (continuing single-speaker)")
            return None

    import sherpa_onnx

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model), num_threads=dia.get("num_threads", 2)))

    segmentation = Path(dia.get(
        "segmentation_model",
        "assets/speaker-segmentation-en/model.int8.onnx",
    ))
    if not segmentation.is_absolute():
        segmentation = Path(__file__).resolve().parent.parent / segmentation
    speaker_activity = None
    if segmentation.is_file():
        try:
            speaker_activity = PyannoteSpeakerActivity(
                segmentation,
                num_threads=dia.get("segmentation_num_threads", 2),
            )
        except Exception as exc:
            print(
                f"[live] speaker segmentation failed to load ({exc}) — "
                "using embedding change points"
            )
    else:
        print(
            "[live] speaker segmentation model missing — run "
            "scripts/fetch_streaming_model.py --speaker-only "
            "(using embedding change points)"
        )

    def embed(samples: np.ndarray) -> np.ndarray | None:
        if len(samples) < int(0.25 * SR):
            return None
        stream = extractor.create_stream()
        stream.accept_waveform(SR, samples)
        stream.input_finished()
        vec = np.asarray(extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else None

    policy = dict(cfg.get("live", {}).get("speaker_attribution", {}) or {})
    fallback = SpeakerTracker(
        embed,
        max_speakers=dia.get("max_speakers", 6),
        window_s=dia.get("window_s", 1.0),
        hop_s=dia.get("hop_s", 0.25),
        change_below=dia.get("change_below", 0.3),
        merge_at=dia.get("merge_at", 0.5),
        min_enrollment_duration_s=policy.get("min_enrollment_duration_s", 0.8),
        min_assignment_duration_s=policy.get("min_assignment_duration_s", 0.25),
        stable_after_observations=policy.get("stable_after_observations", 2),
        stable_after_duration_s=policy.get("stable_after_duration_s", 1.1),
        immediate_speaker_limit=policy.get("immediate_speaker_limit", 2),
        assignment_threshold=policy.get("assignment_threshold", 0.72),
        provisional_threshold=policy.get("provisional_threshold", 0.58),
        new_speaker_threshold=policy.get("new_speaker_threshold", 0.42),
        centroid_ema_alpha=policy.get("centroid_ema_alpha", 0.15),
        switch_hysteresis_s=policy.get("switch_hysteresis_s", 0.35),
        short_turn_max_duration_s=policy.get("short_turn_max_duration_s", 0.4),
        retain_threshold=policy.get("retain_threshold", 0.64),
        switch_threshold=policy.get("switch_threshold", 0.72),
        min_confidence_margin=policy.get("min_confidence_margin", 0.08),
        speaker_min_run_words=policy.get("speaker_min_run_words", 2),
        short_stable_threshold=policy.get("short_stable_threshold"),
        short_stable_min_margin=policy.get("short_stable_min_margin", 0.12),
        short_stable_max_duration_s=policy.get(
            "short_stable_max_duration_s", 1.3
        ),
        min_signal_quality=policy.get("min_signal_quality", 0.25),
        # `direction_trust` is the current name for the voice/direction ratio;
        # `direction_prior_weight` is kept as a fallback so old configs and the
        # `--gain`-style CLI override both still resolve.
        direction_prior_weight=policy.get(
            "direction_trust",
            policy.get("direction_prior_weight", 0.05)),
        speaker_activity=speaker_activity,
        debug=policy.get("debug", False),
    )
    if backend not in {"auto", "sortformer"}:
        print(f"[live] speaker diarizer: embedding ({model.name})")
        return fallback

    sortformer = dict(dia.get("sortformer", {}) or {})
    executable = Path(sortformer.get(
        "executable",
        "native/sortformer/.build/release/autocwi-sortformer",
    ))
    cache_dir = Path(sortformer.get(
        "cache_dir",
        "assets/sortformer-coreml",
    ))
    if not executable.is_absolute():
        executable = REPO_ROOT / executable
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    native_supported = (
        platform.system() == "Darwin"
        and platform.machine() in {"arm64", "aarch64"}
    )
    model_prepared = any(
        path.is_dir()
        for path in cache_dir.glob(
            "sortformer/**/Sortformer_v2.1.mlmodelc"
        )
    )
    if (
        not native_supported
        or not executable.is_file()
        or not model_prepared
    ):
        reason = (
            "requires Apple Silicon"
            if not native_supported
            else (
                "helper is not built"
                if not executable.is_file()
                else "model cache is not prepared"
            )
        )
        print(
            f"[live] Sortformer unavailable ({reason}) — using "
            f"embedding fallback ({model.name}); run "
            "scripts/fetch_streaming_model.py --sortformer-only"
        )
        return fallback

    try:
        from .sortformer import SortformerBridge

        bridge = SortformerBridge(
            executable,
            cache_dir,
            startup_timeout_s=sortformer.get("startup_timeout_s", 120.0),
            debug=policy.get("debug", False),
            preset=str(sortformer.get("preset", "fastV2_1")),
            fp16=bool(sortformer.get("fp16", False)),
        )
    except Exception as exc:
        print(
            f"[live] Sortformer failed to start ({exc}) — using "
            f"embedding fallback ({model.name})"
        )
        return fallback
    print(
        "[live] speaker diarizer: Streaming Sortformer "
        f"({bridge.latency_s:.2f}s) + {model.name} fallback"
    )
    return SortformerHybridSpeakerTracker(
        bridge,
        fallback,
        min_word_coverage=sortformer.get("min_word_coverage", 0.24),
        endpoint_wait_ms=sortformer.get("endpoint_wait_ms", 90.0),
        debug=policy.get("debug", False),
    )


def load_sound_event_detector(cfg: dict):
    """Build the non-speech sound tagger, or None if disabled/missing.

    Like speaker attribution, absence is non-fatal: the non-speech lane is an
    enhancement, and live mode must still run before the one-time audio-tagging
    model download has happened.
    """

    from .soundevents import SoundEventDetector

    se = dict(cfg.get("live", {}).get("sound_events", {}) or {})
    if not se.get("enabled", False):
        return None
    root = Path(__file__).resolve().parent.parent
    model = Path(se.get("model", "assets/audio-tagging-en/model.int8.onnx"))
    labels = Path(se.get("labels", "assets/audio-tagging-en/class_labels_indices.csv"))
    model = model if model.is_absolute() else root / model
    labels = labels if labels.is_absolute() else root / labels
    if not model.is_file() or not labels.is_file():
        print("[live] audio-tagging model missing — run scripts/fetch_streaming_model.py"
              " (continuing without the non-speech lane)")
        return None

    import sherpa_onnx

    tagger = sherpa_onnx.AudioTagging(
        sherpa_onnx.AudioTaggingConfig(
            model=sherpa_onnx.AudioTaggingModelConfig(
                zipformer=sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(
                    model=str(model)),
                num_threads=se.get("num_threads", 2),
                provider="cpu",
            ),
            labels=str(labels),
            top_k=int(se.get("top_k", 5)),
        )
    )

    def classify(samples: np.ndarray) -> list[tuple[str, float]]:
        stream = tagger.create_stream()
        stream.accept_waveform(SR, samples)
        return [(r.name, float(r.prob)) for r in tagger.compute(stream)]

    return SoundEventDetector(
        classify,
        categories=se.get("categories", {}) or {},
        suppress=se.get("suppress", []) or [],
        window_s=se.get("window_s", 2.0),
        hop_s=se.get("hop_s", 0.5),
        min_conf=se.get("min_conf", 0.35),
        end_conf=se.get("end_conf", 0.20),
        hold_s=se.get("hold_s", 0.6),
        min_gap_s=se.get("min_gap_s", 0.8),
    )


def _is_durable_record(ev: dict) -> bool:
    """Which events are written to live_events.jsonl (durable text + haptics).

    Durable speech words, and each finalized non-speech sound (its `end` event,
    which carries the settled duration). Provisional/hypothesis/level/cue and a
    sound's transient `start` are display-only.
    """

    kind = ev.get("type", "word")
    if kind == "word":
        return ev.get("final", True)
    if kind == "sound":
        return ev.get("state") == "end"
    return False


def reconstruct_durable_words(events: Iterable[dict]) -> list[dict]:
    """Reconstruct the latest attributed transcript from a durable event log.

    Speaker correction records are complete word events with the same
    ``word_id`` as the word they supersede. Legacy logs without ``word_id``
    retain append-only behavior.
    """

    words: list[dict] = []
    positions: dict[str, int] = {}
    for event in events:
        if event.get("type", "word") != "word" or not event.get("final", True):
            continue
        word_id = event.get("word_id")
        if word_id is not None and word_id in positions:
            words[positions[word_id]] = dict(event)
        else:
            if word_id is not None:
                positions[word_id] = len(words)
            words.append(dict(event))
    return words


class WhisperEndpointVerifier:
    """Whole-phrase verifier backed by faster-whisper. Text only, never timing.

    WHY THIS EXISTS. Korean shipped with `verifier_enabled: false`, so its
    streaming Zipformer owned durable text alone. MEASURED, Korean's largest
    error category is the FIRST WORD of an utterance -- 5 of 8 clips in the old
    slice had an error in their opening word, with whole words dropped -- which
    is exactly where a CAUSAL streaming model has the least context and a
    whole-utterance pass has all of it.

    The accurate open Korean models are all offline and none of them return
    usable word timings while streaming (Qwen3-ASR states outright that
    "streaming inference does not support ... returning timestamps"). That is
    the same disqualifier the OpenAI lane carries, and the same answer applies:
    a model with no timings may supply **text only**, at the endpoint, with the
    streaming recognizer keeping ownership of every `start`/`end`.

    faster-whisper rather than a better-scoring model because it is ALREADY
    pinned in `requirements.txt` (with ctranslate2), so this adds no dependency
    and no new runtime. Qwen3-ASR-0.6B scores better on OpenKoASR (KsponSpeech
    clean CER 0.139 against large-v3-turbo's 0.155) and is Apache-2.0, but needs
    MLX or a transformers bump; see `docs/KOREAN-ASR.md`.
    """

    def __init__(self, model, language: str, beam_size: int = 5,
                 tail_padding_s: float = 0.0):
        self.model = model
        self.language = language
        self.beam_size = int(beam_size)
        self.tail_padding_s = max(0.0, float(tail_padding_s))

    def transcribe(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        if self.tail_padding_s:
            audio = np.concatenate((
                audio,
                np.zeros(round(self.tail_padding_s * SR), dtype=np.float32),
            ))
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            # The verifier sees ONE endpoint-bounded phrase, so Whisper's own
            # segmentation and its previous-text conditioning have nothing to
            # add and are the usual source of invented text on short audio.
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
        )
        return " ".join(str(s.text).strip() for s in segments).strip()


def load_whisper_verifier(cfg: dict):
    """Build the faster-whisper verifier from `live.verifier_whisper`."""

    from faster_whisper import WhisperModel

    live_cfg = cfg["live"]
    whisper_cfg = dict(live_cfg.get("verifier_whisper", {}) or {})
    name = whisper_cfg.get("model", "large-v3-turbo")
    # CPU int8 is the documented Apple Silicon path: CTranslate2 has no MPS
    # backend, and MPS here is pyannote/torch only.
    model = WhisperModel(
        name,
        device=whisper_cfg.get("device", "cpu"),
        compute_type=whisper_cfg.get("compute_type", "int8"),
        cpu_threads=int(whisper_cfg.get("cpu_threads", 4)),
    )
    print(f"[live] endpoint verifier: faster-whisper {name} "
          f"({whisper_cfg.get('compute_type', 'int8')}, "
          f"lang={live_cfg.get('lang', 'en')})")
    return WhisperEndpointVerifier(
        model,
        language=str(live_cfg.get("lang", "en")),
        beam_size=int(whisper_cfg.get("beam_size", 5)),
        tail_padding_s=live_cfg.get("verifier_tail_padding_s", 0.0),
    )


def load_endpoint_verifier(cfg: dict) -> EndpointVerifier:
    """Load the configured phrase verifier.

    Defaults to the fully offline local model. `live.verifier_backend: openai`
    wraps it in the opt-in cloud verifier, which keeps this local recognizer as
    its mandatory fallback -- see `autocwi/cloud_verifier.py` for why the cloud
    lane may only ever touch durable TEXT and never word timing.
    """

    live_cfg = cfg["live"]
    # `whisper` routes to faster-whisper; anything else keeps the sherpa
    # transducer this has always loaded.
    if str(live_cfg.get("verifier_engine", "sherpa")).casefold() == "whisper":
        return apply_verifier_backend(load_whisper_verifier(cfg), cfg)

    import sherpa_onnx

    model_dir = Path(live_cfg["verifier_model_dir"])
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent.parent / model_dir
    configured_files = live_cfg.get("verifier_files", {}) or {}
    files = {
        "tokens": model_dir / configured_files.get("tokens", "tokens.txt"),
        "encoder": model_dir / configured_files.get(
            "encoder", "encoder.int8.onnx"
        ),
        "decoder": model_dir / configured_files.get(
            "decoder", "decoder.int8.onnx"
        ),
        "joiner": model_dir / configured_files.get(
            "joiner", "joiner.int8.onnx"
        ),
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        fetch_flag = (
            " --korean-only"
            if live_cfg.get("lang") == "ko"
            else ""
        )
        raise SystemExit(
            "endpoint verifier is missing — run: "
            f".venv/bin/python scripts/fetch_streaming_model.py{fetch_flag}\n  "
            + "\n  ".join(missing)
        )
    recognizer_kwargs = dict(
        **{name: str(path) for name, path in files.items()},
        num_threads=live_cfg.get("verifier_num_threads", 4),
        provider="cpu",
        decoding_method=live_cfg.get(
            "verifier_decoding_method", "modified_beam_search"
        ),
        max_active_paths=live_cfg.get("verifier_max_active_paths", 4),
    )
    model_type = live_cfg.get("verifier_model_type", "nemo_transducer")
    if model_type:
        recognizer_kwargs["model_type"] = model_type
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        **recognizer_kwargs
    )
    verifier = EndpointVerifier(
        recognizer,
        tail_padding_s=live_cfg.get("verifier_tail_padding_s", 0.0),
    )

    return apply_verifier_backend(verifier, cfg)


def apply_verifier_backend(verifier: EndpointVerifier, cfg: dict):
    """Wrap the offline verifier according to `live.verifier_backend`.

    Split out from the loader so the backend decision is testable without the
    sherpa model files the offline test suite deliberately does not carry.
    """

    live_cfg = cfg.get("live", {}) or {}
    backend = str(live_cfg.get("verifier_backend", "local")).casefold()
    if backend in ("", "local", "offline"):
        return verifier
    if backend != "openai":
        raise SystemExit(
            f"unknown live.verifier_backend: {backend!r} (expected 'local' or "
            "'openai')"
        )

    from .cloud_verifier import CloudEndpointVerifier, privacy_notice

    cloud = CloudEndpointVerifier(verifier, cfg)
    print(privacy_notice(cloud.model))
    return cloud


# ---------------------------------------------------------------------------
# SSE broadcast server
# ---------------------------------------------------------------------------

class LiveLanguageSession:
    """Thread-safe language choice shared by the startup thread and local UI."""

    def __init__(
        self,
        languages: list[dict],
        language: str | None = None,
    ):
        self.languages = tuple(dict(item) for item in languages)
        self._supported = {str(item["id"]) for item in self.languages}
        if language is not None and language not in self._supported:
            raise ValueError(f"unsupported live language: {language}")
        self._language = language
        self._stage = "loading" if language else "selecting"
        self._lock = threading.Lock()
        self._selected = threading.Event()
        if language:
            self._selected.set()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self._stage,
                "language": self._language,
                "languages": [dict(item) for item in self.languages],
            }

    def select_language(self, language: str) -> dict:
        with self._lock:
            if language not in self._supported:
                raise ValueError(f"unsupported live language: {language}")
            if self._language is not None and self._language != language:
                raise RuntimeError(
                    "language is locked for this capture; restart live mode to change it"
                )
            self._language = language
            if self._stage == "selecting":
                self._stage = "loading"
            self._selected.set()
            return {
                "state": self._stage,
                "language": self._language,
                "languages": [dict(item) for item in self.languages],
            }

    def wait_for_language(self) -> str:
        self._selected.wait()
        with self._lock:
            assert self._language is not None
            return self._language

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._stage = stage


class Broadcaster:
    """Bounded SSE fan-out with durable replay on browser reconnect.

    A stalled tab is disconnected instead of accumulating an unbounded queue.
    EventSource reconnects with ``Last-Event-ID`` and receives the retained
    committed/verified transcript plus the most recent hypothesis snapshot.
    """

    def __init__(self, max_queue: int = 256, history_limit: int = 1024):
        self._clients: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._max_queue = max_queue
        self._next_id = 1
        self._history: deque[tuple[int, bytes]] = deque(maxlen=history_limit)
        self._latest_hypothesis: tuple[int, bytes] | None = None
        self._has_presented_to_client = False

    @staticmethod
    def _replay_chunk(
        event_id: int,
        chunk: bytes,
        *,
        first_presentation: bool = False,
    ) -> bytes:
        """Mark a retained SSE record without changing its stable event id."""

        try:
            data_line = next(
                line for line in chunk.decode().splitlines()
                if line.startswith("data: ")
            )
            event = json.loads(data_line[6:])
            event["_replay"] = True
            event["_first_presentation"] = first_presentation
            return (
                f"id: {event_id}\n"
                f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            ).encode()
        except (StopIteration, UnicodeDecodeError, json.JSONDecodeError):
            return chunk

    def register(self, last_event_id: int | None = None) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._max_queue)
        after = max(0, last_event_id or 0)
        with self._lock:
            first_presentation = not self._has_presented_to_client
            self._has_presented_to_client = True
            replay = [(event_id, chunk) for event_id, chunk in self._history
                      if event_id > after]
            if (self._latest_hypothesis is not None and
                    self._latest_hypothesis[0] > after):
                replay.append(self._latest_hypothesis)
            replay = sorted(dict(replay).items())[-self._max_queue:]
            # Tell the renderer which retained records are history. Their
            # state must be reconstructed, but their already-finished CWI
            # motion must not replay after reconnect. The copied payload gets
            # an ephemeral marker while preserving its original stable SSE id.
            # The first audience connection is different: model loading can
            # delay the browser bundle until opening words are already retained,
            # even though that audience has never seen them. Mark that one
            # startup backlog as first presentation so it keeps first-paint
            # motion; later connections remain ordinary settled replay.
            for event_id, chunk in replay:
                q.put_nowait(self._replay_chunk(
                    event_id,
                    chunk,
                    first_presentation=first_presentation,
                ))
            self._clients.add(q)
        return q

    def unregister(self, q: queue.Queue) -> None:
        with self._lock:
            self._clients.discard(q)

    def publish(self, obj: dict) -> None:
        with self._lock:
            event_id = self._next_id
            self._next_id += 1
            data = (f"id: {event_id}\n"
                    f"data: {json.dumps(obj, ensure_ascii=False)}\n\n").encode()
            event_type = obj.get("type", "word")
            durable = event_type in {"commit", "verification"} or (
                event_type == "word" and obj.get("final", True)
            ) or (event_type == "sound" and obj.get("state") == "end")
            if durable:
                self._history.append((event_id, data))
            elif event_type == "hypothesis":
                self._latest_hypothesis = (event_id, data)
            clients = list(self._clients)
            for q in clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    # Force a clean EventSource reconnect. The browser's last
                    # acknowledged id lets the retained transcript fill the gap.
                    self._clients.discard(q)
                    try:
                        while True:
                            q.get_nowait()
                    except queue.Empty:
                        pass
                    q.put_nowait(None)


def make_handler(
    page_path: Path,
    broadcaster: Broadcaster,
    *,
    static_root: Path | None = None,
    # A Path, or a callable that renders one on demand -- see
    # `_legacy_page_resolver`.
    legacy_page_path: Path | Callable[[], Path] | None = None,
    font_path: Path | None = None,
    korean_font_path: Path | None = None,
    runtime_config: dict | None = None,
    language_session: LiveLanguageSession | None = None,
):
    static_root = static_root.resolve() if static_root is not None else None

    def runtime_body() -> bytes:
        """Serialized on request, NOT once at startup.

        The studio opens before the language picker is answered, and the
        picked profile can override `display:` keys -- `read_ahead_delay_s`
        does, because the recognizers differ in latency. A body frozen at
        construction always describes the pre-language config, which is how
        Korean served a 1750 ms read-ahead budget it cannot meet. `run_live`
        updates `runtime_config` in place once the language is resolved, and
        the client re-fetches when the session language changes.
        """
        return json.dumps(
            runtime_config or {}, ensure_ascii=False
        ).encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                # Browsers close long-lived EventSource sockets during reload
                # and shutdown; this is a normal client lifecycle, not a
                # production server error worth a traceback.
                pass

        def log_message(self, *a):  # keep the console for caption output
            pass

        def _send_bytes(
            self,
            body: bytes,
            content_type: str,
            *,
            cache_control: str = "no-cache",
            status: int = 200,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, value: dict, *, status: int = 200) -> None:
            self._send_bytes(
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status=status,
            )

        def _send_file(
            self,
            path: Path,
            *,
            content_type: str | None = None,
            immutable: bool = False,
        ) -> None:
            mime = content_type or mimetypes.guess_type(path.name)[0]
            self._send_bytes(
                path.read_bytes(),
                mime or "application/octet-stream",
                cache_control=(
                    "public, max-age=31536000, immutable"
                    if immutable else "no-cache"
                ),
            )

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/", "/index.html", "/live.html"):
                self._send_file(
                    page_path, content_type="text/html; charset=utf-8"
                )
            elif path in ("/legacy", "/legacy/") and legacy_page_path is not None:
                target = (
                    legacy_page_path()
                    if callable(legacy_page_path)
                    else legacy_page_path
                )
                self._send_file(
                    target, content_type="text/html; charset=utf-8"
                )
            elif path == "/runtime-config.json":
                self._send_bytes(
                    runtime_body(), "application/json; charset=utf-8"
                )
            elif path == "/session" and language_session is not None:
                self._send_json(language_session.snapshot())
            elif path == "/RobotoFlex.ttf" and font_path is not None:
                self._send_file(
                    font_path,
                    content_type="font/ttf",
                    immutable=True,
                )
            elif path == "/NotoSansKR.ttf" and korean_font_path is not None:
                self._send_file(
                    korean_font_path,
                    content_type="font/ttf",
                    immutable=True,
                )
            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    last_event_id = int(self.headers.get("Last-Event-ID", "0"))
                except ValueError:
                    last_event_id = 0
                q = broadcaster.register(last_event_id)
                try:
                    self.wfile.write(b"retry: 500\n: connected\n\n")
                    self.wfile.flush()
                    while True:
                        try:
                            chunk = q.get(timeout=15)
                        except queue.Empty:
                            chunk = b": keepalive\n\n"
                        if chunk is None:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    broadcaster.unregister(q)
            elif static_root is not None:
                candidate = (static_root / path.lstrip("/")).resolve()
                if (
                    candidate.is_relative_to(static_root)
                    and candidate.is_file()
                ):
                    self._send_file(
                        candidate,
                        immutable=path.startswith("/_next/static/"),
                    )
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path != "/session/language" or language_session is None:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json({"error": "invalid content length"}, status=400)
                return
            if length <= 0 or length > 1024:
                self._send_json({"error": "invalid request body"}, status=400)
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                language = str(payload.get("language", ""))
                snapshot = language_session.select_language(language)
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                self._send_json({"error": "invalid JSON body"}, status=400)
                return
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
                return
            self._send_json(snapshot, status=202)

        def do_OPTIONS(self):
            path = urlsplit(self.path).path
            if path != "/session/language" or language_session is None:
                self.send_error(404)
                return
            # The exported app is same-origin. This preflight exists only for
            # the documented localhost:3000 Next development workflow.
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_live_stack(cfg: dict):
    """Load the active live models and warm them before capture starts.

    Measured: the pool saves little (~0.7 s of 8.3 s — the ONNX session
    constructors hold the GIL, so loads mostly serialize), and that is fine.
    The point is what happens around it: the first-inference initialization
    (~1.3 s) is paid here on silence instead of on the user's first spoken
    words, and the server/page come up BEFORE this runs, with a boot status,
    so "listening" on screen marks the moment speech actually starts being
    captured.
    """

    live_cfg = cfg["live"]
    from .onset import load_phoneme_onset_detector

    # Fast mode renders only the accurate stream. Loading and running the draft
    # anyway used roughly half the streaming-ASR compute while contributing no
    # visible words. Keep it exclusively for explicit raw readahead.
    needs_draft = (
        cfg.get("display", {}).get("mode") == "readahead"
        and live_cfg.get("draft_enabled", True)
    )
    needs_verifier = live_cfg.get("verifier_enabled", True)
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="load") as pool:
        accurate = pool.submit(load_streaming_recognizer, cfg)
        draft = (
            pool.submit(
                load_streaming_recognizer, cfg,
                live_cfg.get("draft_model_dir", live_cfg["streaming_model_dir"]),
            )
            if needs_draft else None
        )
        verifier = pool.submit(load_endpoint_verifier, cfg) if needs_verifier else None
        tracker = pool.submit(load_speaker_tracker, cfg)
        detector = pool.submit(load_sound_event_detector, cfg)
        onset = pool.submit(load_phoneme_onset_detector, cfg)
        accurate = accurate.result()
        draft = draft.result() if draft is not None else None
        verifier = verifier.result() if verifier is not None else None
        tracker = tracker.result()
        detector = detector.result()
        onset = onset.result()
    warm = np.zeros(SR // 2, dtype=np.float32)
    for recognizer in (item for item in (accurate, draft) if item is not None):
        stream = recognizer.create_stream()
        stream.accept_waveform(SR, warm)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
    if verifier is not None:
        verifier.transcribe(warm)
    if tracker is not None:
        tracker.embed(np.zeros(SR, dtype=np.float32))
    if detector is not None:
        # Pay first-inference init here on silence; then drop the warm buffer so
        # the real stream is not classified against 2 s of leading zeros.
        detector.feed(np.zeros(2 * SR, dtype=np.float32), 0.0)
        detector.reset()
    if onset is not None:
        onset.warm()
    return accurate, draft, verifier, tracker, detector, onset


def _start_server(
    page: Path,
    port: int,
    open_browser: bool,
    *,
    static_root: Path | None = None,
    legacy_page: Path | Callable[[], Path] | None = None,
    font_path: Path | None = None,
    korean_font_path: Path | None = None,
    runtime_config: dict | None = None,
    language_session: LiveLanguageSession | None = None,
    host: str = "127.0.0.1",
):
    broadcaster = Broadcaster()
    handler = make_handler(
        page,
        broadcaster,
        static_root=static_root,
        legacy_page_path=legacy_page,
        font_path=font_path,
        korean_font_path=korean_font_path,
        runtime_config=runtime_config,
        language_session=language_session,
    )
    server = None
    for p in range(port, port + 10):
        try:
            server = http.server.ThreadingHTTPServer((host, p), handler)
            port = p
            break
        except OSError:
            print(f"[live] port {p} busy, trying {p + 1}")
    if server is None:
        raise SystemExit(f"no free port in {port}..{port + 9} — pass --port")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # The browser always opens on loopback even when the bind is wider: the
    # studio runs on this machine, and only the hardware node needs the LAN
    # address. `Local and offline by default` -- a wider bind is opt-in, says
    # so, and still reaches nothing beyond the LAN.
    url = f"http://127.0.0.1:{port}/"
    print(f"[live] serving {url}")
    if host not in ("127.0.0.1", "localhost"):
        print(f"[live] also reachable on the LAN at {host}:{port} "
              f"(--host {host}) — no telemetry, no internet")
    if open_browser:
        webbrowser.open(url)
    return server, broadcaster


def _live_language_options(cfg: dict) -> list[dict]:
    configured = cfg.get("live", {}).get("languages", {}) or {}
    options = []
    for language, values in configured.items():
        values = values or {}
        options.append({
            "id": language,
            "label": values.get("label", language),
            "nativeLabel": values.get("native_label", values.get("label", language)),
            "description": values.get("description", ""),
        })
    if not options:
        options.append({
            "id": "en",
            "label": "English",
            "nativeLabel": "English",
            "description": "",
        })
    return options


def _configure_live_language(cfg: dict, language: str) -> dict:
    live_cfg = cfg.get("live", {}) or {}
    languages = live_cfg.get("languages", {}) or {}
    if language not in languages:
        supported = ", ".join(languages) or "en"
        raise SystemExit(
            f"unsupported live language {language!r}; choose one of: {supported}"
        )
    override = dict(languages.get(language, {}) or {})
    # A PROFILE MAY ALSO OVERRIDE `display:` KEYS, and the read-ahead budget is
    # why. `display.read_ahead_delay_s` has to exceed the recognizer's OWN
    # latency or CWI 2.2.1 delivers nothing, and the profiles do not share a
    # recognizer: MEASURED 2026-09-04 at the shipped 1.75 s, English delivers
    # 488 ms of read-ahead and Korean delivers **0 ms**, because the 1120 ms
    # multilingual model plus its endpoint hold-back spends the whole budget.
    # Raising the GLOBAL would push English 1.45 s further behind the speaker
    # to fix a profile it is already fine for, so the budget belongs to the
    # profile. Popped before the `live:` merge so it cannot also land there.
    display_override = dict(override.pop("display", {}) or {})
    merged = {**live_cfg, **override, "lang": language}
    for nested in (
        "onset_prefix",
        "diarization",
        "speaker_attribution",
    ):
        if nested in override:
            merged[nested] = {
                **(live_cfg.get(nested, {}) or {}),
                **(override.get(nested, {}) or {}),
            }
    resolved = {**cfg, "live": merged}
    if display_override:
        resolved["display"] = {
            **(cfg.get("display", {}) or {}),
            **display_override,
        }
    return resolved


def _studio_runtime_config(
    cfg: dict,
    *,
    selected_language: str | None = None,
    language_selection_required: bool = False,
) -> dict:
    display = cfg.get("display", {}) or {}
    read_ahead = cfg.get("read_ahead", {}) or {}
    live_sync = cfg.get("motion", {}).get("live_sync", {}) or {}
    # The Next studio's motion is its own: `studio:` overrides any shared key,
    # so the legacy diagnostics renderer keeps the values it was tuned against.
    # See the block comment on `studio:` in config.yaml.
    live_sync = {**live_sync, **(live_sync.get("studio") or {})}
    return {
        "palette": list(cfg.get("palette", []))
        + list(cfg.get("palette_support", [])),
        # Same speakers, same hues, darkened for the studio's light stage.
        "paletteLight": list(cfg.get("palette_light", []))
        + list(cfg.get("palette_support_light", [])),
        # CWI 2.1.1 mains and 2.1.2 supporting colours arrive concatenated; the
        # client has to know where the boundary is to exhaust the mains first.
        "paletteSupportCount": len(cfg.get("palette_support", [])),
        "displayMode": display.get("mode", "fast"),
        "maxWords": display.get("max_words", 8),
        "paragraphWordLimit": display.get(
            "studio_paragraph_word_limit", 0
        ),
        "stageParagraphHistory": display.get(
            "studio_stage_paragraph_history", 6
        ),
        "stageWordsPerBlock": display.get(
            "studio_stack_words_per_block", 6
        ),
        "stageWordsMin": display.get("studio_stack_words_min", 3),
        "stageMinRows": display.get("studio_stack_min_rows", 16),
        # CWI 2.2.1. How far the caption playhead trails the acoustic clock,
        # and therefore how much recognized-but-uncoloured text the viewer can
        # read ahead. See the long note on `read_ahead_delay_s` in config.yaml.
        "readAheadDelayMs": round(
            float(display.get("read_ahead_delay_s", 2.5)) * 1000
        ),
        # A per-WORD floor on the read-ahead. `read_ahead_delay_s` sets the
        # MEAN lead and says nothing about its spread; the recognizer releases
        # a batch at each endpoint, so measured at 1.75s the median lead is a
        # healthy 700ms while 11% of frames still show ZERO words ahead of the
        # playhead. This guarantees every word is readable before it turns.
        "minReadAheadMs": round(
            float(display.get("min_read_ahead_ms", 420))
        ),
        # STAGGER FOR CAUGHT-UP WORDS. The recognizer releases a batch of words
        # at each endpoint; when they are all behind the playhead they clamp to
        # the same per-word floor and pop as one wall, which reads as "the
        # queued words popped at once". This spaces successive floor-clamped
        # words so a release ripples instead. Only the studio uses it; the
        # legacy diagnostics page reads the raw `word_reveal_*` keys.
        "wordRevealCatchupGapMs": round(
            float(display.get("word_reveal_catchup_gap_s", 0.06)) * 1000
        ),
        "readAheadColor": read_ahead.get("color", "#FFFFFF"),
        # Same read-ahead, legible on the boxless light stage. See config.yaml.
        "readAheadColorLight": read_ahead.get("color_light", "#6E6E73"),
        "readAheadOpacity": float(read_ahead.get("opacity", 0.9)),
        "colorTurnMs": round(
            float(cfg.get("motion", {}).get("color_turn_ms", 90))
        ),
        "wordMotionBaseMs": round(
            float(display.get("word_motion_duration_s", 0.42)) * 1000
        ),
        "wordMotionMaxMs": round(
            float(display.get("word_motion_max_duration_s", 0.85)) * 1000
        ),
        "wordMotionSpanStretch": display.get(
            "word_motion_span_stretch", 0.42
        ),
        "wordMotionPopMaxMs": round(
            float(display.get("word_motion_pop_max_duration_s", 0.70)) * 1000
        ),
        "wordMotionMinMs": round(
            float(display.get("word_motion_min_duration_s", 0.32)) * 1000
        ),
        "wordMotionFollowsSpeech": bool(
            display.get("word_motion_follows_speech", True)),
        "wordMotionSpeechScale": float(
            display.get("word_motion_speech_scale", 2.4)),
        "wordMotionSpeechFloorMs": round(
            float(display.get("word_motion_speech_floor_s", 0.16)) * 1000
        ),
        "syncPop": live_sync.get("sync_pop", 0.15),
        "syncPopEnhanced": live_sync.get("sync_pop_enhanced", 0.55),
        # What fraction of the pop an UNEMPHASISED word takes. The fallback
        # here matches config.yaml deliberately: the other fallbacks on this
        # object do NOT, and a word built from them before /runtime-config.json
        # resolves freezes at that amplitude for the rest of the session.
        "syncPopFloorEnhanced": live_sync.get("sync_pop_floor_enhanced", 0.5),
        "wordLiftEmEnhanced": live_sync.get("word_lift_em_enhanced", 0.045),
        "wordMotionEnhancedMs": live_sync.get("word_motion_enhanced_ms", 500),
        "wordMotionEnhancedEmphasisMs": live_sync.get(
            "word_motion_enhanced_emphasis_ms", 0),
        "crestLagMs": live_sync.get("crest_lag_ms", 40),
        "voiceScaleDeadbandEnhanced": live_sync.get(
            "voice_scale_deadband_enhanced", 0.10),
        "voiceScaleResponseEnhanced": live_sync.get(
            "voice_scale_response_enhanced", 0.50),
        "voiceScalePivotEnhanced": live_sync.get(
            "voice_scale_pivot_enhanced", 0.10),
        "voiceScaleCurveEnhanced": live_sync.get(
            "voice_scale_curve_enhanced", 2.0),
        "voiceScalePointsEnhanced": live_sync.get(
            "voice_scale_points_enhanced") or None,
        "characterWaveFalloff": live_sync.get("character_wave_falloff", 1.0),
        "characterWaveFloor": live_sync.get("character_wave_floor", 0.0),
        "holdLiftEm": live_sync.get("hold_lift_em", 0.525),
        "holdFullS": live_sync.get("hold_full_s", 0.88),
        "holdMaxS": live_sync.get("hold_max_s", 1.06),
        "holdMinS": live_sync.get("hold_min_s", 0.78),
        "holdPreMs": live_sync.get("hold_pre_ms", 260),
        "holdHoldMs": live_sync.get("hold_hold_ms", 420),
        "holdLandMs": live_sync.get("hold_land_ms", 290),
        # CWI 2.4.4/2.4.5: sound labels yield to speech (the PR film never
        # shows one beside an active dialogue caption).
        "soundLingerMs": round(
            float(display.get("sound_label_speech_linger_s", 0.8)) * 1000
        ),
        "voiceScaleRange": [
            float(v) for v in live_sync.get(
                "voice_scale_range", [0.90, 1.20]
            )
        ],
        "voiceScaleResponse": float(
            live_sync.get("voice_scale_response", 0.25)
        ),
        "voiceScaleResponseQuiet": float(
            live_sync.get("voice_scale_response_quiet", 0.55)
        ),
        "voiceScaleDeadband": float(
            live_sync.get("voice_scale_deadband", 0.34)
        ),
        "widthRange": [
            float(v) for v in live_sync.get("width_range", [82, 124])
        ],
        "deliveryMotionEnabled": live_sync.get("delivery_enabled", True),
        "deliveryFlowDurationMs": live_sync.get(
            "delivery_flow_duration_ms", 90
        ),
        "weightRange": [
            float(v) for v in live_sync.get("weight_range", [340, 760])
        ],
        "weightEmphasis": float(live_sync.get("weight_emphasis", 0.55)),
        "deliveryMinConfidence": live_sync.get(
            "delivery_min_confidence", 0.38
        ),
        "languages": _live_language_options(cfg),
        "selectedLanguage": selected_language,
        "languageSelectionRequired": language_selection_required,
    }


def _legacy_page_resolver(cfg: dict, out: Path) -> Callable[[], Path]:
    """Render `out/live.html` on FIRST REQUEST, not on every run.

    The diagnostics page is ~2.4 MB because the variable font is base64-embedded
    in it, and with the studio built nothing reads it unless someone opens
    `/legacy`. Writing it every run left a stale multi-megabyte artefact in
    `out/` after a run that never used it.
    """
    cache: dict[str, Path] = {}

    def resolve() -> Path:
        if "path" not in cache:
            from .livepage import render_live
            cache["path"] = Path(render_live(cfg, out))
        return cache["path"]

    return resolve


def _select_live_frontend(
    cfg: dict,
    legacy_page: Callable[[], Path],
) -> tuple[Path, Path | None, Callable[[], Path] | None, Path, Path | None, dict]:
    display = cfg.get("display", {}) or {}
    studio_root = REPO_ROOT / "web" / "out"
    studio_page = studio_root / "index.html"
    requested = display.get("frontend", "next")
    static_root = None
    # Only the fallback renders it eagerly, because only the fallback IS the
    # frontend; behind a built studio it stays a callable.
    page = studio_page if requested == "next" and studio_page.is_file() else legacy_page()
    legacy_route = None
    if requested == "next" and studio_page.is_file():
        static_root = studio_root
        legacy_route = legacy_page
        print("[live] frontend: Next.js studio (legacy diagnostics at /legacy)")
    elif requested == "next":
        print(
            "[live] Next.js studio not built; using legacy frontend "
            "(run: cd web && npm install && npm run build)"
        )

    font_path = Path(cfg["render"]["font_path"])
    if not font_path.is_absolute():
        font_path = REPO_ROOT / font_path
    korean_font_value = cfg["render"].get(
        "korean_font_path", "assets/NotoSansKR.ttf"
    )
    korean_font_path = Path(korean_font_value)
    if not korean_font_path.is_absolute():
        korean_font_path = REPO_ROOT / korean_font_path
    if not korean_font_path.is_file():
        print(
            "[live] Korean variable font missing; using a system fallback "
            "(run: python scripts/fetch_font.py)"
        )
        korean_font_path = None
    return (
        page,
        static_root,
        legacy_route,
        font_path,
        korean_font_path,
        _studio_runtime_config(cfg),
    )


def run_live(args, cfg: dict, device: str) -> None:
    live_cfg = cfg["live"]
    if getattr(args, "list_devices", False):
        print("[live] input devices:\n" + list_input_devices())
        return
    requested_lang = getattr(args, "lang", None)
    port = getattr(args, "port", None) or live_cfg["port"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # CLI overrides sit on top of config.yaml so a booth operator can pin the
    # gain without editing the tracked configuration.
    gain_cfg = dict(live_cfg.get("input_gain", {}) or {})
    if getattr(args, "no_gain", False):
        gain_cfg["enabled"] = False
    if getattr(args, "gain", None) is not None:
        gain_cfg.update(enabled=True, initial_gain_db=args.gain,
                        min_gain_db=args.gain, max_gain_db=args.gain)
    cfg = {**cfg, "live": {**live_cfg, "input_gain": gain_cfg}}
    live_cfg = cfg["live"]

    # Same pattern for the voice/direction ratio: pin it from the booth without
    # editing the config. Applied AFTER the language overlay so it overrides the
    # active profile's speaker_attribution, not the root default.
    if getattr(args, "direction_trust", None) is not None:
        trust = float(np.clip(args.direction_trust, 0.0, 0.9))
        sa = dict(live_cfg.get("speaker_attribution", {}) or {})
        sa["direction_trust"] = trust
        cfg = {**cfg, "live": {**live_cfg, "speaker_attribution": sa}}
        live_cfg = cfg["live"]
        print(f"[live] direction trust pinned to {trust:.2f} "
              f"(voice {1 - trust:.2f} / direction {trust:.2f})")

    legacy_page = _legacy_page_resolver(cfg, out)
    (
        page,
        static_root,
        legacy_route,
        font_path,
        korean_font_path,
        runtime_config,
    ) = (
        _select_live_frontend(cfg, legacy_page)
    )

    stop = threading.Event()
    headless = bool(getattr(args, "once", False))
    whisper_model = getattr(args, "whisper", None)
    language_options = _live_language_options(cfg)
    supported_languages = {item["id"] for item in language_options}
    if requested_lang is not None and requested_lang not in supported_languages:
        raise SystemExit(
            f"unsupported live language {requested_lang!r}; choose one of: "
            + ", ".join(sorted(supported_languages))
        )
    # The static Next.js studio is the language picker. An explicit --lang
    # remains the deterministic/headless override, while the diagnostics
    # fallback keeps the configured default if the studio has not been built.
    selection_required = (
        requested_lang is None
        and not headless
        and static_root is not None
    )
    initial_lang = (
        None if selection_required
        else (requested_lang or live_cfg.get("lang", "en"))
    )
    language_session = LiveLanguageSession(language_options, initial_lang)
    runtime_config.update({
        "languages": language_options,
        "selectedLanguage": initial_lang,
        "languageSelectionRequired": selection_required,
    })

    realtime = not os.environ.get("AUTOCWI_FAST")
    mic_device = getattr(args, "device", None)
    if mic_device is not None and str(mic_device).lstrip("-").isdigit():
        mic_device = int(mic_device)

    # The hardware node owns capture when it is selected, so it binds before
    # the models load and waits for the Pi rather than opening a local mic.
    node_link = None
    if getattr(args, "node", False):
        host = getattr(args, "host", "127.0.0.1")
        if host == "127.0.0.1":
            print("[live] --node on 127.0.0.1: only a node on THIS machine can "
                  "connect. Pass --host 0.0.0.0 to accept one over the LAN.")
        node_link = NodeLink(host=host, port=getattr(args, "node_port", 7338))
    server = broadcaster = None
    if not headless:
        # Page first, language second, models third. Capture has not started
        # while the local UI is selecting or warming its language-specific ASR.
        server, broadcaster = _start_server(
            page,
            port,
            not getattr(args, "no_open", False),
            static_root=static_root,
            legacy_page=legacy_route,
            font_path=font_path,
            korean_font_path=korean_font_path,
            runtime_config=runtime_config,
            language_session=language_session,
            host=getattr(args, "host", "127.0.0.1"),
        )
        if selection_required:
            broadcaster.publish({"type": "boot", "stage": "choose language"})
            print("[live] choose English or 한국어 in the browser")
            try:
                lang = language_session.wait_for_language()
            except KeyboardInterrupt:
                stop.set()
                server.shutdown()
                print("\n[live] stopped before capture")
                return
        else:
            lang = initial_lang
        language_session.set_stage("loading")
        broadcaster.publish({
            "type": "boot",
            "stage": "loading models",
            "language": lang,
        })
    else:
        lang = initial_lang

    assert lang is not None
    cfg = _configure_live_language(cfg, lang)
    live_cfg = cfg["live"]
    # The profile may override `display:` keys, and the runtime config handed
    # to the studio was built before the language was known. Update it IN
    # PLACE: the server closed over this dict, and `/runtime-config.json` is
    # serialized per request precisely so this lands.
    if runtime_config is not None:
        runtime_config.clear()
        runtime_config.update(_studio_runtime_config(cfg))
    diarizer_override = getattr(args, "diarizer", None)
    if diarizer_override is not None:
        cfg = {
            **cfg,
            "live": {
                **live_cfg,
                "diarization": {
                    **(live_cfg.get("diarization", {}) or {}),
                    "backend": diarizer_override,
                    "enabled": diarizer_override != "off",
                },
            },
        }
        live_cfg = cfg["live"]

    # Resolve the bundled clip only after language selection. Previously the
    # picker could select Korean while `--sample` had already bound the English
    # video, making a healthy Korean model look catastrophically inaccurate.
    source_file = getattr(args, "file", None)
    start_s = getattr(args, "start", None)
    if getattr(args, "sample", False) and not source_file:
        source_file = sample_clip_path(lang)
        if start_s is None:
            # The PR film changes treatment at 28 s -- before it is the titles
            # demo on black, after it is CWI applied to real footage, which is
            # what this product does. Start there. The Korean clip is 13 s of
            # FLEURS narration and has no such split, so it is never skipped.
            start_s = 28.0 if lang != "ko" else 0.0
        print(
            f"[live] streaming bundled {lang} sample: "
            f"{Path(source_file).name}"
            + (f" from {start_s:g}s" if start_s else "")
        )
    if start_s is None:
        start_s = 0.0
    # --once processes to EOF and exits; a loop would never reach EOF.
    loop = getattr(args, "loop", False) and not getattr(args, "once", False)
    if loop and source_file:
        print("[live] looping clip — Ctrl-C to quit")
    if source_file:
        blocks = file_blocks(source_file, realtime=realtime, loop=loop,
                             start_s=float(start_s))
    elif node_link is not None:
        blocks = net_blocks(stop, node_link)
    else:
        blocks = mic_blocks(stop, device=mic_device)

    if whisper_model:
        from faster_whisper import WhisperModel

        ct2_device = "cuda" if device == "cuda" else "cpu"
        model = WhisperModel(whisper_model, device=ct2_device,
                             compute_type="float16" if ct2_device == "cuda" else "int8")
        print(f"[live] legacy whisper-{whisper_model} ready (lang={lang})")
        if broadcaster is not None:
            language_session.set_stage("listening")
            broadcaster.publish({
                "type": "boot",
                "stage": "listening",
                "language": lang,
            })
        events = word_events(utterances(blocks, live_cfg), model, lang, cfg)
    else:
        t_load = time.perf_counter()
        model, draft_model, verifier, tracker, detector, onset = _load_live_stack(cfg)
        model_label = live_cfg.get("model_label", "streaming transducer")
        print(f"[live] local ASR ready in {time.perf_counter() - t_load:.1f}s: "
              + model_label
              + (" + 160ms draft readahead" if draft_model is not None else "")
              + (" + endpoint verifier" if verifier is not None else "")
              + f" (lang={lang}, CPU)"
              + (" + speaker attribution" if tracker else "")
              + (" + non-speech sound lane" if detector else "")
              + (" + phoneme onset hints" if onset else ""))
        if broadcaster is not None:
            language_session.set_stage("listening")
            broadcaster.publish({
                "type": "boot",
                "stage": "listening",
                "language": lang,
            })
        # The level meter reports the array's bearing when one is attached.
        # Built here rather than inside `streaming_events` so the node's
        # direction can reach it without threading the link through the
        # recognizer's signature.
        input_gain = InputGain(cfg)
        if node_link is not None:
            input_gain.direction_source = lambda: node_link.direction_deg
            # Same injection shape for attribution: the tracker's direction
            # prior was plumbed end to end but never fed, so `direction_deg`
            # only ever reached the compass. With an array attached, WHO spoke
            # now has spatial evidence beside the voice evidence.
            if tracker is not None:
                tracker.direction_for_span = node_link.bearing_for_span
                # The ring's "who is where" marks. NOT gated on
                # `direction_slot_colouring`: that switch decides whether
                # geometry may colour an undecided WORD, which is a claim about
                # identity. This only describes where decisions the voice lane
                # already made were heard from, so turning the colouring off
                # must not blind the compass.
                tracker.speaker_bearings = SpeakerBearingMap()
                input_gain.speaker_bearings_source = (
                    lambda: tracker.speaker_bearings.marks)
                # One map shared by both captioners and the level meter, so the
                # compass marks the same slots the captions are coloured from.
                speaker_cfg = (cfg.get("live", {}) or {}).get(
                    "speaker_attribution", {}) or {}
                if speaker_cfg.get("direction_slot_colouring", True):
                    tracker.direction_speakers = DirectionSpeakerMap(
                        tolerance_deg=float(speaker_cfg.get(
                            "direction_slot_tolerance_deg", 35.0)))
                    input_gain.speaker_slots_source = (
                        lambda: tracker.direction_speakers.slots)
        events = streaming_events(
            blocks, model, cfg, draft_model, verifier=verifier,
            gain=input_gain,
            speaker_tracker=tracker, sound_detector=detector,
            onset_detector=onset,
        )
    events_path = out / "live_events.jsonl"

    if getattr(args, "once", False):  # headless: process source to EOF, no server
        with open(events_path, "w", encoding="utf-8") as f:
            n = 0
            for ev in events:
                if _is_durable_record(ev):
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    n += 1
        print(f"[live] {n} durable events -> {events_path}")
        return

    if server is None:  # defensive: non-headless paths normally start above
        server, broadcaster = _start_server(
            page,
            port,
            not getattr(args, "no_open", False),
            static_root=static_root,
            legacy_page=legacy_route,
            font_path=font_path,
            korean_font_path=korean_font_path,
            runtime_config=runtime_config,
            language_session=language_session,
        )

    # Haptics actuate on the durable record and nothing else. `cue_for_word`
    # returns None for the overwhelming majority of words, which is the rule:
    # continuous vibration measured as distracting, so the motors fire on
    # speaker changes and emphasis only.
    haptic_intensity = float(
        (cfg.get("haptics", {}) or {}).get("intensity", 0.8)
    )
    cue_seq = 0

    try:
        with open(events_path, "w", encoding="utf-8") as f:
            for ev in events:
                broadcaster.publish(ev)
                if _is_durable_record(ev):
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    f.flush()
                    if node_link is not None and ev.get("type") == "word":
                        cue = haptics.cue_for_word(ev, haptic_intensity)
                        if cue is not None:
                            cue_seq += 1
                            node_link.send_cue(cue.flag, cue.direction_deg,
                                               cue.intensity, cue_seq)
        print("[live] audio source finished — server still running (Ctrl-C to quit)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.shutdown()
        print("\n[live] stopped")
