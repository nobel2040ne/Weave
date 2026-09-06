"""Weave CLI.

  python -m autocwi live                           # live CWI captions (primary)
  python -m autocwi run <input> --out out/ [--whisper small] [--speakers 2] [--stub]
  python -m autocwi transcribe|diarize|prosody|fuse ...
  python -m autocwi cc <spec.json> [--media <media>]    # CWI closed captions

Offline stages read/write JSON intermediates in --out, so they can be run,
inspected, and swapped independently; the pipeline's product is the
CaptionSpec (spec.json), the stable contract for any consumer:

  words.json -> segments.json -> prosody.json -> spec.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import audio
from .config import load_config
from .device import pick_device, seed_everything
from .schema import (
    Media,
    ProsodyList,
    SegmentList,
    WordList,
    load_model,
    save_model,
)

WORDS_JSON = "words.json"
SEGMENTS_JSON = "segments.json"
PROSODY_JSON = "prosody.json"
SPEC_JSON = "spec.json"


def _common(p: argparse.ArgumentParser, needs_input: bool = True) -> None:
    if needs_input:
        p.add_argument("input", help="input media file (mp4/wav/...)")
    p.add_argument("--out", default="out", help="output directory (default: out/)")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--stub", action="store_true",
                   help="use deterministic placeholder stages (no models needed)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="autocwi", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="full offline pipeline: media -> spec.json")
    _common(run)
    run.add_argument("--whisper", default=None, help="whisper size: base|small|medium")
    run.add_argument("--speakers", type=int, default=None, help="fix speaker count")
    run.add_argument("--lang", default=None, help="whisper language hint")

    tr = sub.add_parser("transcribe", help="media -> words.json")
    _common(tr)
    tr.add_argument("--whisper", default=None)
    tr.add_argument("--lang", default=None)

    di = sub.add_parser("diarize", help="media -> segments.json")
    _common(di)
    di.add_argument("--speakers", type=int, default=None)

    pr = sub.add_parser("prosody", help="media + words.json -> prosody.json")
    _common(pr)

    fu = sub.add_parser("fuse", help="words+segments+prosody -> spec.json")
    _common(fu)

    ccp = sub.add_parser("cc", help="spec.json -> closed-caption playback page")
    ccp.add_argument("spec", help="path to spec.json (from `run`/`fuse`)")
    ccp.add_argument("--out", default="out")
    ccp.add_argument("--config", default=None)
    ccp.add_argument("--media", default=None,
                     help="optional video/audio to play behind the captions")
    ccp.add_argument("--no-open", dest="no_open", action="store_true")
    ccp.add_argument("--tune", action="store_true",
                     help="add the live motion tuner (writes tuner.html)")

    tn = sub.add_parser(
        "tune", help="live motion tuner: loop a line and adjust the motion by hand")
    tn.add_argument("spec", nargs="?", default=None,
                    help="spec.json to tune against (default: a built-in line)")
    tn.add_argument("--out", default="out")
    tn.add_argument("--config", default=None)
    tn.add_argument("--media", default=None)
    tn.add_argument("--no-open", dest="no_open", action="store_true")

    lv = sub.add_parser("live", help="live CWI captions from the microphone")
    lv.add_argument("--out", default="out")
    lv.add_argument("--config", default=None)
    lv.add_argument(
        "--whisper", default=None, metavar="MODEL",
        help="use the legacy pause-segmented Whisper path (e.g. base.en) instead of streaming",
    )
    lv.add_argument(
        "--lang",
        # Kept as an explicit list rather than read from config: this is the
        # deterministic/headless bypass, and a typo here must fail loudly at
        # parse time instead of loading the wrong recognizer.
        choices=("en", "ko", "multi"),
        default=None,
        help="skip the startup picker: English (en), Korean (ko), or bilingual "
             "auto-detect (multi)",
    )
    lv.add_argument("--file", default=None,
                    help="stream a wav/mp4 file at real-time pace instead of the mic")
    lv.add_argument("--sample", action="store_true",
                    help="stream the bundled sample for the selected language")
    # The PR film changes treatment at 28 s: before it is the titles demo on
    # black (a per-character wave), after it is CWI applied to real footage
    # (whole-word colour and size) -- which is what this product does. The
    # English sample therefore starts there by default. `--start 0` plays the
    # whole film; the Korean clip is 13 s and is never skipped.
    lv.add_argument("--start", type=float, default=None, metavar="SECONDS",
                    help="skip this many seconds of --sample/--file "
                         "(default: 28 for the English sample, 0 otherwise)")
    lv.add_argument(
        "--diarizer",
        choices=("auto", "sortformer", "embedding", "off"),
        default=None,
        help="speaker backend: native streaming Sortformer when available, "
             "the embedding fallback, or disabled",
    )
    lv.add_argument("--loop", action="store_true",
                    help="restart the --file/--sample clip when it ends (continuous demo)")
    lv.add_argument("--port", type=int, default=None)
    lv.add_argument("--no-open", dest="no_open", action="store_true",
                    help="don't open the browser automatically")
    lv.add_argument("--once", action="store_true",
                    help="headless: process --file to EOF, write events, exit")
    lv.add_argument("--device", default=None,
                    help="input device index or name (default: system default)")
    lv.add_argument("--list-devices", dest="list_devices", action="store_true",
                    help="print the available capture devices and exit")
    lv.add_argument("--gain", type=float, default=None, metavar="DB",
                    help="fixed input gain in dB for the recognizer; disables "
                         "the adaptive gain that lifts quiet speech")
    lv.add_argument("--no-gain", dest="no_gain", action="store_true",
                    help="feed the recognizer the raw input level")
    lv.add_argument("--direction-trust", dest="direction_trust", type=float,
                    default=None, metavar="RATIO",
                    help="0..1 direction-sensor share of speaker attribution "
                         "(needs the mic array). 0 = voice only; higher trusts "
                         "the array's bearing more when it and the voice "
                         "embedding disagree")
    # --- hardware node (ReSpeaker array + Pi + haptics) --------------------
    # The array plugs into the Pi, which cannot host the recognizers, so audio
    # and direction arrive over the network and haptic cues go back.
    lv.add_argument("--node", action="store_true",
                    help="capture from the hardware node instead of a local mic")
    lv.add_argument("--node-port", dest="node_port", type=int, default=7338,
                    metavar="PORT",
                    help="port the hardware node connects to (default: 7338)")
    # Loopback by default. `Local and offline by default` is a hard rule; the
    # node needs a LAN address, so widening the bind is explicit and opt-in,
    # never a side effect of --node.
    lv.add_argument("--host", default="127.0.0.1", metavar="ADDR",
                    help="address to serve the studio and node port on "
                         "(default: 127.0.0.1; use 0.0.0.0 to let the "
                         "hardware node reach this machine over the LAN)")

    return ap


# --------------------------------------------------------------------------
# Stage runners (each reads/writes the JSON intermediates)
# --------------------------------------------------------------------------

def tuner_spec(cfg: dict) -> dict:
    """A short authored line for the motion tuner, when no spec is to hand.

    Deliberately exercises every knob at once: a long word so the character
    sweep is visible, two-letter words so the ripple's rate can be checked
    against them, one clearly loud word and one quiet one to bracket the
    intonation envelope, a run of ordinary words that the deadband should hold
    perfectly still, and a second speaker so the colour turn is not the only
    thing being judged.
    """

    rows = [
        # text,        start, end,  loudness, pitch_hz, speaker
        ("precisely",  0.60, 1.18, 0.52, 168.0, "S1"),
        ("as",         1.20, 1.34, 0.48, 160.0, "S1"),
        ("each",       1.36, 1.60, 0.55, 172.0, "S1"),
        ("word",       1.62, 1.92, 0.62, 180.0, "S1"),
        ("is",         1.94, 2.08, 0.45, 158.0, "S1"),
        ("spoken.",    2.10, 2.68, 0.58, 150.0, "S1"),
        ("so",         3.40, 3.56, 0.50, 165.0, "S2"),
        ("you",        3.58, 3.74, 0.48, 162.0, "S2"),
        ("can",        3.76, 3.96, 0.52, 170.0, "S2"),
        ("feel",       3.98, 4.26, 0.56, 176.0, "S2"),
        ("when",       4.28, 4.50, 0.50, 168.0, "S2"),
        ("my",         4.52, 4.68, 0.47, 160.0, "S2"),
        ("voice",      4.70, 5.02, 0.58, 174.0, "S2"),
        ("gets",       5.04, 5.28, 0.55, 170.0, "S2"),
        ("louder",     5.30, 5.92, 0.95, 210.0, "S2"),
        ("or",         5.94, 6.08, 0.42, 150.0, "S2"),
        ("softer.",    6.10, 6.80, 0.12, 128.0, "S2"),
    ]
    palette = list(cfg["palette"])
    words = [{
        "text": text, "start": start, "end": end, "speaker": speaker,
        "loudness": loud, "pitch": 0.5, "loudness_db": -34.0 + 22 * loud,
        "pitch_hz": hz, "voiced_frac": 0.9, "conf": 0.95,
    } for text, start, end, loud, hz, speaker in rows]
    return {
        "version": "1.0",
        "media": {"path": "tuner", "duration": 8.2, "fps": 30.0},
        "speakers": {"S1": {"color": palette[0]}, "S2": {"color": palette[2]}},
        "words": words,
        "mapping": cfg["mapping"],
    }


def stage_cc(args, cfg, _device) -> None:
    """Render a finished CaptionSpec as a CWI closed-caption playback page.

    Live mode can only approximate CWI (read-ahead needs the line in advance);
    this is the faithful reference the live renderer is measured against.
    """

    import webbrowser

    from .ccpage import render_cc

    # Validate anything loaded from disk. A malformed spec should fail here,
    # not render silently wrong -- this used to be a bare json.loads.
    from .schema import CaptionSpec, load_model
    if getattr(args, "spec", None) in (None, "-"):
        spec = tuner_spec(cfg)
    else:
        spec = load_model(CaptionSpec, args.spec).model_dump(exclude_none=True)
    media = args.media
    if media:
        media = Path(media).resolve().as_uri()
    tune = getattr(args, "tune", False) or args.cmd == "tune"
    page = render_cc(cfg, spec, args.out, media=media, tune=tune)
    print(f"[{args.cmd}] {len(spec['words'])} words -> {page}")
    if tune:
        print("[tune] sliders take effect on the next frame; "
              "'Show config.yaml' prints the values to keep")
    if not getattr(args, "no_open", False):
        webbrowser.open(Path(page).resolve().as_uri())


def _wav(args) -> Path:
    return audio.extract_wav(args.input, args.out)


def stage_transcribe(args, cfg, device) -> None:
    if args.stub:
        from .stubs import stub_transcribe
        words = stub_transcribe(audio.probe(args.input)["duration"])
    else:
        from .asr import transcribe
        words = transcribe(
            _wav(args),
            model_size=getattr(args, "whisper", None) or cfg["asr"]["model"],
            lang=getattr(args, "lang", None),
            beam_size=cfg["asr"]["beam_size"],
            temperature=cfg["asr"]["temperature"],
            device=device,
        )
    save_model(WordList(words=words), Path(args.out) / WORDS_JSON)
    print(f"[transcribe] {len(words)} words -> {args.out}/{WORDS_JSON}")


def stage_diarize(args, cfg, device) -> None:
    if args.stub:
        from .stubs import stub_diarize
        segments = stub_diarize(audio.probe(args.input)["duration"])
    else:
        from .diarize import diarize
        segments = diarize(
            _wav(args),
            num_speakers=getattr(args, "speakers", None),
            device=device,
            model=cfg["diarization"]["model"],
        )
    save_model(SegmentList(segments=segments), Path(args.out) / SEGMENTS_JSON)
    print(f"[diarize] {len(segments)} segments -> {args.out}/{SEGMENTS_JSON}")


def stage_prosody(args, cfg, device) -> None:
    words = load_model(WordList, Path(args.out) / WORDS_JSON).words
    if args.stub:
        from .stubs import stub_prosody
        features = stub_prosody(words, seed=cfg["seed"])
    else:
        from .prosody import prosody
        features = prosody(
            _wav(args), words,
            pitch_floor_hz=cfg["prosody"]["pitch_floor_hz"],
            pitch_ceiling_hz=cfg["prosody"]["pitch_ceiling_hz"],
        )
    save_model(ProsodyList(features=features), Path(args.out) / PROSODY_JSON)
    print(f"[prosody] {len(features)} features -> {args.out}/{PROSODY_JSON}")


def stage_fuse(args, cfg, _device) -> None:
    from .fuse import fuse

    out = Path(args.out)
    info = audio.probe(args.input)
    spec = fuse(
        words=load_model(WordList, out / WORDS_JSON).words,
        segments=load_model(SegmentList, out / SEGMENTS_JSON).segments,
        features=load_model(ProsodyList, out / PROSODY_JSON).features,
        media=Media(path=str(args.input), duration=round(info["duration"], 3),
                    fps=info["fps"]),
        config=cfg,
    )
    save_model(spec, out / SPEC_JSON)
    print(f"[fuse] {len(spec.words)} words, {len(spec.speakers)} speakers -> {args.out}/{SPEC_JSON}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = pick_device()

    if args.cmd == "live":
        from .live import run_live
        run_live(args, cfg, device)
        return

    stages = {
        "transcribe": [stage_transcribe],
        "diarize": [stage_diarize],
        "prosody": [stage_prosody],
        "fuse": [stage_fuse],
        "run": [stage_transcribe, stage_diarize, stage_prosody, stage_fuse],
        "cc": [stage_cc],
        "tune": [stage_cc],
    }
    for stage in stages[args.cmd]:
        stage(args, cfg, device)


if __name__ == "__main__":
    main()
