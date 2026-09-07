#!/usr/bin/env python3
"""Phase 0 probe: what can the ReSpeaker XVF3800 actually tell us?

Run this ON THE PI, with the array plugged in, BEFORE anything is built on it.
It answers the two questions no amount of code review settles:

  1. Does the array enumerate as a capture device, at what rate and how many
     channels?
  2. Can direction of arrival be read, by what command, in what units?

**The DoA answer is deliberately not hardcoded.** The XVF3800 exposes a control
interface over USB and Seeed ships a host utility for it, but the exact command
name and output format vary by firmware and vendor build. Guessing it and
building on the guess is how this project has produced confidently wrong
results before, so this probe TRIES a list of candidates, prints exactly what
each one returned, and asks you to record the winner. Nothing downstream should
assume a format until this has been run against the real board.

    python3 scripts/hw/probe_array.py              # audio + DoA discovery
    python3 scripts/hw/probe_array.py --watch      # stream DoA once it is found
    python3 scripts/hw/probe_array.py --seconds 5  # longer capture sample
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

# Candidate control utilities, in the order they are worth trying. The XMOS
# host app is usually one of these names; Seeed's build has been shipped under
# several. Extend rather than replace -- a name that worked once is evidence.
CONTROL_BINARIES = [
    "xvf_host",
    "xvf3800_host",
    "vfctrl_usb",
    "vfctrl",
    "dfu-util",          # not a control app, but its presence confirms USB access
]

# Candidate commands for direction of arrival, again by likelihood. XMOS names
# this differently across generations (DOA / doa / azimuth).
DOA_COMMANDS = [
    ["GET_DOA"],
    ["--get", "DOA"],
    ["get", "doa"],
    ["GET_AEC_AZIMUTH"],
    ["GET_DIRECTION"],
]


def _run(cmd: list[str], timeout: float = 4.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except OSError as e:
        return 1, f"{e}"


def probe_audio(seconds: float) -> None:
    print("=" * 68)
    print("1. AUDIO — does the array enumerate, and in what shape?")
    print("=" * 68)
    try:
        import sounddevice as sd
    except Exception as e:                                   # noqa: BLE001
        print(f"  sounddevice unavailable ({e}) — pip install sounddevice")
        return

    array = None
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        name = dev["name"]
        flag = ""
        if any(k in name.lower() for k in ("xvf", "respeaker", "xmos")):
            array = (index, dev)
            flag = "   <-- looks like the array"
        print(f"  [{index}] {name}  "
              f"in={dev['max_input_channels']}ch @ "
              f"{int(dev['default_samplerate'])}Hz{flag}")

    if array is None:
        print("\n  No device matched xvf/respeaker/xmos.")
        print("  If the array is plugged in, note its index above and pass it")
        print("  to the node as --device; the name match is only a convenience.")
        return

    index, dev = array
    rate = int(dev["default_samplerate"])
    channels = int(dev["max_input_channels"])
    print(f"\n  Capturing {seconds:.1f}s from [{index}] at {rate}Hz x{channels}...")
    try:
        import numpy as np
        rec = sd.rec(int(seconds * rate), samplerate=rate,
                     channels=channels, dtype="float32", device=index)
        sd.wait()
    except Exception as e:                                   # noqa: BLE001
        print(f"  capture FAILED: {e}")
        return

    print(f"  got {rec.shape[0]} frames x {rec.shape[1]} channels")

    # Whole-file RMS cannot identify the processed channel: over a capture that
    # is mostly room tone, every channel reports the room. Score the frames
    # where the sound actually is -- and first check that any of it is SPEECH,
    # because a silent room and a knocked case both produce "loud" frames and
    # neither says anything about which channel carries a voice.
    hop = int(0.020 * rate)
    n = rec.shape[0] // hop
    if n < 25:
        print("  capture too short to segment — use --seconds 3 or more")
        return
    blocks = rec[: n * hop].reshape(n, hop, rec.shape[1])
    frame_db = 20 * np.log10(np.maximum(
        np.sqrt((blocks ** 2).mean(axis=1)), 1e-9))
    mix = frame_db.max(axis=1)
    loud = mix >= np.percentile(mix, 90)

    spec = np.abs(np.fft.rfft(blocks[:, :, 0] * np.hanning(hop), axis=1))
    freqs = np.fft.rfftfreq(hop, 1.0 / rate)
    voice = spec[:, (freqs >= 300) & (freqs < 3400)].mean(axis=1)
    rumble = spec[:, (freqs >= 20) & (freqs < 200)].mean(axis=1)
    tilt = float(np.median(20 * np.log10(
        np.maximum(voice[loud], 1e-9) / np.maximum(rumble[loud], 1e-9))))
    span = float(np.percentile(mix, 99) - np.percentile(mix, 20))

    print(f"\n  loud-frame voice tilt (300-3400Hz over 20-200Hz): {tilt:+.1f} dB")
    print(f"  dynamic range (p99-p20):                          {span:.1f} dB")
    if tilt < 0 or span < 18:
        print("\n  >>> NO SPEECH IN THIS CAPTURE. A negative tilt is a thump or")
        print("  >>> hum, not a voice, and under ~18 dB of range is an empty")
        print("  >>> room. The channel comparison below is NOT EVIDENCE — the")
        print("  >>> run is INVALID. Re-run while somebody talks into the array.")

    print("\n            whole-capture    quiet frames     LOUD frames")
    for ch in range(rec.shape[1]):
        whole = 20 * np.log10(max(float(np.sqrt((rec[:, ch] ** 2).mean())), 1e-9))
        quiet = float(np.median(frame_db[~loud, ch]))
        hot = float(np.median(frame_db[loud, ch]))
        peak = float(np.max(np.abs(rec[:, ch])))
        silent = "   (silent — reserved channel?)" if peak < 1e-5 else ""
        print(f"    ch{ch}:  {whole:10.1f}  {quiet:14.1f}  {hot:14.1f}"
              f"   ({hot - quiet:+.1f} dB over floor){silent}")

    print("\n  RECORD THIS: the pipeline wants 16 kHz MONO. The node picks a")
    print("  channel; the processed/beamformed one is the channel that rises")
    print("  MOST over its own floor when a person speaks — not the loudest")
    print("  one overall. On the XVF3800 the beamformed output is exposed")
    print("  separately from the raw mics; averaging them undoes the")
    print("  beamforming the board just did.")


def probe_control_interface() -> None:
    """Does the BOARD expose a control interface, whatever is on PATH?

    A missing host utility and a board that cannot be controlled look
    identical from PATH alone, and they call for opposite responses: install
    a program, versus abandon direction and say so. Only the USB descriptors
    tell them apart, so read them before reporting either.
    """
    print()
    print("=" * 68)
    print("2. CONTROL INTERFACE — can the BOARD be controlled at all?")
    print("=" * 68)
    code, out = _run(["lsusb", "-v", "-d", "2886:"], timeout=10.0)
    if code == 127:
        print("  lsusb unavailable — skipping (Linux only; on macOS use")
        print("  system_profiler SPUSBDataType).")
        return
    if not out or "bInterfaceClass" not in out:
        print("  No descriptor detail at all — is the array plugged in?")
        return

    # As an ordinary user, lsusb prints the config descriptor but CANNOT read
    # the string descriptors: every iInterface comes back blank. Keying the
    # verdict on those labels then reports "no control interface" for a board
    # that has one -- the stop-the-project answer, arrived at by lacking
    # permission to see. Refuse to conclude instead.
    blind = "couldn't open device" in out.lower()

    interfaces: list[tuple[str, str]] = []
    cls = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("bInterfaceClass"):
            cls = s.split(None, 1)[1] if len(s.split()) > 1 else s
        elif s.startswith("iInterface") and cls is not None:
            parts = s.split(None, 2)
            interfaces.append((cls, parts[2] if len(parts) > 2 else ""))
            cls = None

    for cls, label in interfaces:
        note = ""
        if "Vendor Specific" in cls and (not label or "control" in label.lower()):
            note = "   <-- DoA is read through this"
        elif "Device Firmware" in cls or "DFU" in label.upper():
            note = "   (firmware update — do not touch unless reflashing)"
        unnamed = "(name needs root)" if blind else "(unnamed)"
        print(f"  {cls:34s} {label or unnamed}{note}")

    controllable = any("Vendor Specific" in c for c, _ in interfaces)
    print()
    if blind:
        print("  PARTIAL READ — 'Couldn't open device'. Interface CLASSES are")
        print("  visible above but their NAMES are not, so this run cannot")
        print("  confirm which vendor interface is the control one.")
        print("  Re-run as root before believing either answer:")
        print("    sudo python3 scripts/hw/probe_array.py --skip-audio")
        print("  (sudo loses a --user pip install, which is why audio is")
        print("  skipped there and the two sections are run separately.)")
        if not controllable:
            print("\n  No vendor-specific interface was visible even so — but a")
            print("  blind read is not evidence of absence. Do NOT conclude.")
        return
    if controllable:
        print("  The board EXPOSES a control interface in its stock firmware.")
        print("  DoA is therefore a question of installing a HOST utility, not")
        print("  of modifying the array. Those are different decisions; do not")
        print("  report 'DoA unavailable' when only the host tool is missing.")
    else:
        print("  NO control interface found on this board/firmware.")
        print("  This is the case docs/HARDWARE.md says to STOP on: direction is")
        print("  the point of the motors, and salience-without-direction is a")
        print("  different, smaller feature — not a booth-day discovery.")


def probe_doa_usb(watch: bool) -> bool:
    """Read DoA over the board's own control interface, no host binary.

    This is tried BEFORE hunting for a utility on PATH, because a missing
    `xvf_host` was never the actual requirement: the protocol is control
    transfers on EP0 and pyusb speaks it directly.
    """
    print()
    print("=" * 68)
    print("3. DIRECTION OF ARRIVAL — direct, over the control interface")
    print("=" * 68)
    try:
        from xvf_control import open_array
    except ImportError:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        try:
            from xvf_control import open_array
        except ImportError as e:                             # noqa: BLE001
            print(f"  xvf_control unavailable ({e})")
            return False

    try:
        array = open_array()
    except Exception as e:                                   # noqa: BLE001
        print(f"  could not open the control lane: {e}")
        return False
    if array is None:
        print("  pyusb missing or array not found — pip install pyusb")
        print("  (needs a libusb: brew install libusb / apt install libusb-1.0-0)")
        return False

    try:
        print(f"  firmware VERSION: {array.version()}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  control transfer FAILED: {e}")
        print("  On Linux this is usually permissions — add a udev rule for")
        print(f"  {VID_PID}, or re-run with sudo to confirm that is the cause.")
        return False

    print("  The control lane is OPEN. DoA needs no host binary.")
    reps = 15 if watch else 4
    # Beam order is fixed by the firmware: beam 1, beam 2, free-running,
    # auto-select. The first two are the independently steered talker beams, so
    # two of them carrying speech AT DIFFERENT BEARINGS is the whole question
    # "can this array hear two people at once" -- answered by measurement here
    # rather than by the single auto-select bearing the node currently ships.
    names = ("b1", "b2", "free", "auto")
    print(f"\n    {'bearing':>9}  {'spch':>4} | {'beam azimuths (deg)':^33}"
          f" | {'speech energy':^27} | live beams")
    print(f"    {'':>9}  {'':>4} | " + " ".join(f"{n:>7}" for n in names)
          + " | " + " ".join(f"{n:>5}" for n in names) + " |")
    voiced = 0
    multi = 0
    talker_multi = 0
    for _ in range(reps):
        bearing = array.read_bearing()
        azimuths = array.beam_azimuths_deg()
        energies = array.beam_speech_energy()
        az = " ".join(f"{a:7.1f}" for a in azimuths)
        if energies is None:
            # Range-checked out: reading two commands close together
            # intermittently returns one's data for the other. Say so; a
            # silently dropped row would understate how often this happens.
            en, live = "  (corrupt read — discarded)      ", ""
        else:
            en = " ".join(f"{e:5.2f}" for e in energies)
            hot = [i for i, e in enumerate(energies) if e > 0.0]
            live = " ".join(names[i] for i in hot) or "-"
            if len(hot) >= 2:
                multi += 1
            # Only beams 1 and 2 are separately steered talkers; the free and
            # auto beams can echo one of them, so counting all four would
            # overstate separation.
            if energies[0] > 0.0 and energies[1] > 0.0:
                talker_multi += 1
        if bearing is None:
            print(f"    {'(omitted)':>9}  {0:>4} | {az} | {en} | {live}")
        else:
            voiced += 1
            print(f"    {bearing:8.1f}°  {1:>4} | {az} | {en} | {live}")
        time.sleep(0.4)

    print()
    if voiced:
        print(f"  Speech detected on {voiced}/{reps} reads and the bearing")
        print("  updated. RECORD the reference direction: which way is 0 deg on")
        print("  the case? The compass renders this straight into a rotation.")
    else:
        print("  Speech was never detected, so the bearing NEVER UPDATED and is")
        print("  reported as omitted above — correctly. This run says the lane")
        print("  reads; it does NOT say the bearing tracks a voice. That needs")
        print("  somebody talking and moving around the array.")
    print()
    if talker_multi:
        print(f"  BOTH TALKER BEAMS carried speech on {talker_multi}/{reps} reads.")
        print("  Compare their azimuth columns (b1, b2): two DIFFERENT bearings")
        print("  is the array separating two people in space. The same bearing")
        print("  on both is one talker held by two beams, which is not.")
    elif multi:
        print(f"  Several beams were live on {multi}/{reps} reads, but never b1")
        print("  and b2 together — that is one talker echoed onto the free and")
        print("  auto beams, NOT two directions.")
    else:
        print("  Never more than one beam live. With two people talking at")
        print("  different angles this is the read that says the array cannot")
        print("  separate them; with one talker it says nothing either way.")
    array.close()
    return True


VID_PID = "2886:001a"


def probe_control() -> str | None:
    print()
    print("=" * 68)
    print("4. HOST UTILITY — is one present? (optional; the direct path above")
    print("   already covers DoA)")
    print("=" * 68)
    found = None
    for name in CONTROL_BINARIES:
        path = shutil.which(name)
        if path:
            print(f"  FOUND  {name:14s} -> {path}")
            if found is None and name != "dfu-util":
                found = name
        else:
            print(f"  --     {name}")
    if found is None:
        print("\n  No XMOS control utility on PATH.")
        print("  DoA needs one. Get it from Seeed's XVF3800 documentation or")
        print("  build XMOS's host app, then re-run this probe.")
    return found


def probe_doa(binary: str, watch: bool) -> None:
    print()
    print("=" * 68)
    print(f"4. DIRECTION OF ARRIVAL — via {binary}")
    print("=" * 68)
    working: list[str] | None = None
    for args in DOA_COMMANDS:
        cmd = [binary, *args]
        code, out = _run(cmd)
        status = "ok " if code == 0 else f"rc={code}"
        print(f"  {status}  {' '.join(cmd)}")
        if out:
            for line in out.splitlines()[:4]:
                print(f"         | {line}")
        if code == 0 and out and "not found" not in out.lower():
            working = cmd
            break

    if working is None:
        print("\n  None of the candidates returned a value.")
        print("  Run the utility's own help and find the DoA command yourself:")
        print(f"    {binary} --help")
        print("  Then ADD IT to DOA_COMMANDS at the top of this file, so the")
        print("  next person does not have to rediscover it.")
        return

    print(f"\n  WORKING COMMAND: {' '.join(working)}")
    print("  RECORD THIS in docs/HARDWARE.md — the node reads DoA with it.")
    print("  Note the UNITS and RANGE: the pipeline wants degrees 0..360, and")
    print("  the compass renders the number straight into a CSS rotation.")

    if not watch:
        print("\n  Re-run with --watch to stream it and check it tracks a voice.")
        return

    print("\n  Watching — speak from different sides of the array, Ctrl-C to stop.")
    try:
        while True:
            code, out = _run(working)
            print(f"    {time.strftime('%H:%M:%S')}  {out.splitlines()[-1] if out else '(empty)'}")
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n  stopped")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=2.0,
                    help="capture length for the audio probe (default: 2)")
    ap.add_argument("--watch", action="store_true",
                    help="stream DoA once a working command is found")
    ap.add_argument("--skip-audio", action="store_true")
    args = ap.parse_args()

    if not args.skip_audio:
        probe_audio(args.seconds)
    probe_control_interface()
    if not probe_doa_usb(args.watch):
        binary = probe_control()
        if binary:
            probe_doa(binary, args.watch)

    print()
    print("=" * 68)
    print("Next: scripts/hw/probe_motor.py, then wire the node.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
