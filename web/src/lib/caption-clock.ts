/** THE PRESENTATION PLAYHEAD — what makes CWI 2.2.1 possible in open
   captions. */

/** Acoustic seconds since capture start, as the server timestamps them. */
export interface ClockSample {
  /** `level.t` / `word.t`, in ms. */
  acousticMs: number;
  /** `performance.now()` when it was received. */
  monotonicMs: number;
}

export interface PlayheadClock {
  /** `acousticMs - monotonicMs`. */
  offsetMs: number;
  /** When `offsetMs` was last revised, for the drift decay. */
  updatedAtMs: number;
  /** False until the first sample; the caller must not present captions yet. */
  started: boolean;
  /** Bumped whenever the clock resyncs onto a NEW capture timeline. */
  epoch: number;
}

export const IDLE_CLOCK: Readonly<PlayheadClock> = Object.freeze({
  offsetMs: 0,
  updatedAtMs: 0,
  started: false,
  epoch: 0,
});

/** How far the acoustic clock may jump back before it counts as a new
   capture rather than as jitter. */
export const RESYNC_TOLERANCE_MS = 1500;

/** Drift bleed-off, in ms of offset per second of wall time. */
export const DRIFT_DECAY_MS_PER_S = 5;

/* NO CATCH-UP SLEW HERE, AND THE REASON IS MEASURED (2026-08-01). */

function finite(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

/** Fold one acoustic reading into the clock. */
export function advanceClock(
  clock: PlayheadClock,
  acousticMs: number,
  monotonicMs: number,
  resyncToleranceMs = RESYNC_TOLERANCE_MS,
): PlayheadClock {
  const acoustic = finite(acousticMs);
  const monotonic = finite(monotonicMs);
  if (!Number.isFinite(acoustic) || !Number.isFinite(monotonic)) return clock;

  const sampleOffset = acoustic - monotonic;
  if (!clock.started) {
    return {
      offsetMs: sampleOffset,
      updatedAtMs: monotonic,
      started: true,
      epoch: clock.epoch,
    };
  }

  const elapsedS = Math.max(0, monotonic - clock.updatedAtMs) / 1000;
  const relaxed = clock.offsetMs - elapsedS * DRIFT_DECAY_MS_PER_S;

  // A large backwards step is a new capture (loop restart, restarted server),
  // not a late packet. Snap rather than holding a playhead that would sit in
  // the far future of the new timeline and colour every incoming word at once.
  if (sampleOffset < relaxed - Math.max(0, resyncToleranceMs)) {
    return {
      offsetMs: sampleOffset,
      updatedAtMs: monotonic,
      started: true,
      epoch: clock.epoch + 1,
    };
  }

  const offsetMs = Math.max(relaxed, sampleOffset);
  if (offsetMs === clock.offsetMs && monotonic === clock.updatedAtMs) {
    return clock;
  }
  return {
    offsetMs,
    updatedAtMs: monotonic,
    started: true,
    epoch: clock.epoch,
  };
}

/** Current acoustic time, interpolated between samples. */
export function acousticNowMs(
  clock: PlayheadClock,
  monotonicMs: number,
): number {
  if (!clock.started) return Number.NEGATIVE_INFINITY;
  return monotonicMs + clock.offsetMs;
}

/** The caption playhead: acoustic time minus the read-ahead delay. */
export function presentationNowMs(
  clock: PlayheadClock,
  monotonicMs: number,
  delayMs: number,
): number {
  if (!clock.started) return Number.NEGATIVE_INFINITY;
  return monotonicMs + clock.offsetMs - Math.max(0, delayMs);
}

/** When, on `performance.now()`'s timeline, the playhead reaches
   `acousticMs`. */
export function monotonicTimeForAcousticMs(
  clock: PlayheadClock,
  acousticMs: number,
  delayMs: number,
): number {
  return acousticMs - clock.offsetMs + Math.max(0, delayMs);
}

/** Read-ahead actually available right now, in ms. */
export function readAheadMs(
  clock: PlayheadClock,
  newestAcousticMs: number,
  monotonicMs: number,
  delayMs: number,
): number {
  const playhead = presentationNowMs(clock, monotonicMs, delayMs);
  if (!Number.isFinite(playhead) || !Number.isFinite(newestAcousticMs)) {
    return 0;
  }
  return Math.max(0, newestAcousticMs - playhead);
}

/** The moment a word turns colour, and whether the read-ahead floor decided it.

    Three inputs settle it: the word's own acoustic turn, the per-word floor
    (`now + minReadAheadMs`, because a time delay alone cannot guarantee every
    word is legible before it is spoken), and -- the 2026-09-04 addition -- the
    turn of the previous FLOOR-CLAMPED word, so a burst released together does
    not collapse onto one moment.

    A transducer emits words in bursts at each endpoint. When those words are
    behind the playhead their honest turn is in the past, so each takes the
    floor -- and because they are all scheduled in one commit they share the
    same `nowMs`, hence the same floor, hence turn in the SAME frame. Measured
    on live Korean, bursts of four words carrying 1.28-1.76s of acoustic spread
    arrived within 50ms and would pop as one wall. `lastFloorTurnMs` chains such
    words by `catchupGapMs` into a ripple. A word whose acoustic turn is still
    AHEAD of the floor is untouched: it is already spaced by its own onset, and
    holding it back would flatten genuine fast speech, whose overlapping pops
    are the design working. The chain is self-limiting -- once the playhead
    idles, the floor overtakes it and it resets -- so a burst can never push a
    word unboundedly into the future.
*/
export interface TurnMoment {
  turnAtMs: number;
  /** True when the read-ahead floor (not the acoustic turn) set the moment. */
  clamped: boolean;
}

export function turnMomentFor(
  acousticTurnMs: number,
  nowMs: number,
  minReadAheadMs: number,
  lastFloorTurnMs: number,
  catchupGapMs: number,
): TurnMoment {
  const floorMs = nowMs + Math.max(0, minReadAheadMs);
  const clamped = floorMs >= acousticTurnMs;
  if (!clamped) {
    return {turnAtMs: acousticTurnMs, clamped: false};
  }
  const gap = Math.max(0, catchupGapMs);
  const turnAtMs = Math.max(floorMs, lastFloorTurnMs + gap);
  return {turnAtMs, clamped: true};
}
