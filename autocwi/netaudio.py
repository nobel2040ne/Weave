"""Wire protocol for the hardware node (ReSpeaker array + Pi Zero 2 W).

The Pi cannot host the recognizers, so the capture path crosses the network:
the Pi captures and ships audio, the Mac runs the pipeline, haptic cues come
back. This module is the framing both ends share.

- **Audio is float32.** int16 would halve a bandwidth that is already trivial,
  and `AudioChunk.samples` must stay at the true captured level — prosody
  measures `loudness_db` from it and that drives the volume -> size channel.
- **A sequence gap is a real capture gap.** TCP does not lose data mid-stream,
  so a jump in `seq` means the SENDER dropped blocks. Capture is lossless by
  rule, so it surfaces as `AudioChunk.discontinuity` rather than being hidden.
- **One connection carries everything**, so DoA needs no clock of its own: it
  references the audio sequence number it was observed against.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

# `CWI1` also acts as a resync token: a reader that loses framing scans for it
# rather than closing the connection, so one corrupt frame is not a dropout.
MAGIC = b"CWI1"

KIND_HELLO = 1   # node -> host, once per connection: rate, channels, node id
KIND_AUDIO = 2   # node -> host, float32 mono samples
KIND_DOA = 3     # node -> host, direction of arrival + the audio seq it refers to
KIND_CUE = 4     # host -> node, one haptic actuation

_HEADER = struct.Struct("!4sBII")   # magic, kind, seq, payload length
HEADER_SIZE = _HEADER.size          # 13 bytes

# A frame larger than this is a framing error, not a big frame.
MAX_PAYLOAD = 1 << 20


class ProtocolError(ValueError):
    """Raised for a malformed frame that resynchronisation cannot rescue."""


@dataclass(frozen=True)
class Frame:
    kind: int
    seq: int
    payload: bytes

    def json(self) -> dict:
        """Decode a control payload. Audio frames are not JSON -- use `samples`."""
        return json.loads(self.payload.decode("utf-8"))

    def samples(self) -> np.ndarray:
        """Decode an audio payload into the float32 mono block the pipeline wants."""
        return np.frombuffer(self.payload, dtype="<f4").astype(np.float32)


def pack(kind: int, seq: int, payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload {len(payload)} exceeds {MAX_PAYLOAD}")
    return _HEADER.pack(MAGIC, kind, seq & 0xFFFFFFFF, len(payload)) + payload


def pack_audio(seq: int, samples: np.ndarray) -> bytes:
    """Frame one block of mono audio.

    Little-endian float32 explicitly, so a big-endian host on either side reads
    the same numbers -- the Pi and the Mac are both LE today, which is exactly
    the sort of assumption that stops being true silently.
    """
    block = np.ascontiguousarray(samples, dtype="<f4")
    if block.ndim != 1:
        raise ProtocolError(f"audio must be mono, got shape {block.shape}")
    return pack(KIND_AUDIO, seq, block.tobytes())


def pack_json(kind: int, seq: int, obj: dict) -> bytes:
    return pack(kind, seq, json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def pack_hello(sample_rate: int, block: int, node: str = "weave-node") -> bytes:
    return pack_json(KIND_HELLO, 0, {
        "node": node,
        "sample_rate": sample_rate,
        "block": block,
        "format": "f32le",
    })


def pack_doa(audio_seq: int, doa_deg: float, confidence: float = 1.0,
             beams: Sequence[float] | None = None) -> bytes:
    """Frame a direction observation against the audio block it was measured on.

    ``beams`` is OPTIONAL and carries every steered talker beam that was
    reporting speech, so a second simultaneous talker survives the wire instead
    of being collapsed into the dominant bearing. Omitted when empty -- an
    absent field means nothing was measured, which is the same contract
    ``doa_deg`` keeps, and a reader that predates this field is unaffected.
    """
    return pack_json(KIND_DOA, audio_seq, {
        "doa_deg": round(float(doa_deg) % 360.0, 2),
        "confidence": round(float(confidence), 3),
        **({"beams": [round(float(b) % 360.0, 2) for b in beams]}
           if beams else {}),
    })


def pack_cue(seq: int, flag: str, direction_deg: float | None,
             intensity: float) -> bytes:
    """Frame one haptic actuation.

    `direction_deg` is None when the word carried no direction. The node must
    fall back to its whole ring rather than inventing a bearing -- the same rule
    the compass follows when it shows `awaiting array`.
    """
    body: dict = {"flag": flag, "intensity": round(float(intensity), 3)}
    if direction_deg is not None:
        body["direction_deg"] = round(float(direction_deg) % 360.0, 2)
    return pack_json(KIND_CUE, seq, body)


class FrameReader:
    """Incremental framer over a byte stream.

    A socket read returns whatever arrived, which splits and coalesces frames
    arbitrarily, so the reader buffers and yields only whole frames. On a bad
    header it scans forward for the next `MAGIC` instead of raising: one corrupt
    frame should cost one frame, not the capture.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self.resyncs = 0

    def feed(self, data: bytes) -> Iterator[Frame]:
        self._buf.extend(data)
        while True:
            if len(self._buf) < HEADER_SIZE:
                return
            magic, kind, seq, length = _HEADER.unpack_from(self._buf, 0)
            if magic != MAGIC or length > MAX_PAYLOAD:
                if not self._resync():
                    return
                continue
            if len(self._buf) < HEADER_SIZE + length:
                return
            payload = bytes(self._buf[HEADER_SIZE:HEADER_SIZE + length])
            del self._buf[:HEADER_SIZE + length]
            yield Frame(kind=kind, seq=seq, payload=payload)

    def _resync(self) -> bool:
        """Drop to the next plausible frame start. False when none is buffered."""
        found = self._buf.find(MAGIC, 1)
        if found < 0:
            # Keep the last few bytes: MAGIC may be split across two reads.
            keep = max(0, len(self._buf) - (len(MAGIC) - 1))
            del self._buf[:keep]
            return False
        del self._buf[:found]
        self.resyncs += 1
        return True


class SequenceTracker:
    """Turns sender-side drops into an explicit discontinuity.

    Because the transport is reliable, this never fires on network loss -- only
    when the node itself could not keep up. That distinction is the whole point:
    it is a capture gap, and the pipeline already has a field for one.
    """

    def __init__(self) -> None:
        self.expected: int | None = None
        self.dropped = 0
        self.gaps = 0

    def observe(self, seq: int) -> bool:
        """Record a block. True when audio was lost immediately before it."""
        if self.expected is None or seq == self.expected:
            self.expected = seq + 1
            return False
        if seq < self.expected:
            # A retransmitted or reordered block cannot happen over TCP; treat
            # it as a node restart rather than trusting a backwards counter.
            self.expected = seq + 1
            self.gaps += 1
            return True
        self.dropped += seq - self.expected
        self.gaps += 1
        self.expected = seq + 1
        return True
