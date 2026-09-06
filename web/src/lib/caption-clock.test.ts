import assert from "node:assert/strict";
import test from "node:test";
import {
  advanceClock,
  acousticNowMs,
  DRIFT_DECAY_MS_PER_S,
  IDLE_CLOCK,
  monotonicTimeForAcousticMs,
  presentationNowMs,
  readAheadMs,
  turnMomentFor,
} from "./caption-clock.ts";

test("no captions are presented before the first acoustic sample", () => {
  assert.equal(IDLE_CLOCK.started, false);
  assert.equal(acousticNowMs(IDLE_CLOCK, 1_000), Number.NEGATIVE_INFINITY);
  assert.equal(
    presentationNowMs(IDLE_CLOCK, 1_000, 2_500),
    Number.NEGATIVE_INFINITY,
  );
});

test("the playhead trails acoustic time by exactly the read-ahead delay", () => {
  // Capture is 30 s in when performance.now() reads 10 s.
  const clock = advanceClock(IDLE_CLOCK, 30_000, 10_000);
  assert.equal(acousticNowMs(clock, 10_000), 30_000);
  assert.equal(presentationNowMs(clock, 10_000, 2_500), 27_500);
  // It keeps running between samples.
  assert.equal(presentationNowMs(clock, 10_500, 2_500), 28_000);
});

test("transport jitter cannot drag the playhead backwards", () => {
  // Three readings of the same stream: the middle one was delivered 180 ms
  // late, so it reports an older acoustic time for a later arrival. Only a
  // maximum filter survives that; an average or last-wins would stutter.
  let clock = advanceClock(IDLE_CLOCK, 30_000, 10_000);
  const trueOffset = clock.offsetMs;
  clock = advanceClock(clock, 30_064, 10_244);
  clock = advanceClock(clock, 30_128, 10_128);
  assert.equal(clock.offsetMs, trueOffset);
});

test("a genuine capture restart resyncs and bumps the epoch", () => {
  // `--sample --loop` returns to t=0. Holding the old offset would put the
  // playhead ~30 s into the future of the new timeline and colour every
  // incoming word the instant it arrived.
  let clock = advanceClock(IDLE_CLOCK, 30_000, 10_000);
  assert.equal(clock.epoch, 0);
  clock = advanceClock(clock, 120, 10_500);
  assert.equal(acousticNowMs(clock, 10_500), 120);
  // The epoch is what tells already-spoken words to settle rather than being
  // re-derived onto the new timeline and reverting to read-ahead white.
  assert.equal(clock.epoch, 1);
});

test("ordinary samples never bump the epoch", () => {
  let clock = advanceClock(IDLE_CLOCK, 30_000, 10_000);
  for (let index = 1; index <= 20; index += 1) {
    clock = advanceClock(clock, 30_000 + index * 64, 10_000 + index * 64);
  }
  assert.equal(clock.epoch, 0);
});

test("the offset relaxes slowly enough to be invisible", () => {
  let clock = advanceClock(IDLE_CLOCK, 30_000, 10_000);
  const before = clock.offsetMs;
  // One second later the source reports one second more audio, on the dot.
  clock = advanceClock(clock, 31_000, 11_000);
  assert.equal(clock.offsetMs, before);
  // With no new sample the decay is bounded to the documented rate.
  clock = advanceClock(clock, 31_000, 12_000);
  assert.equal(clock.offsetMs, before - DRIFT_DECAY_MS_PER_S);
});

test("a word's turn is frozen as an absolute monotonic moment", () => {
  const clock = advanceClock(IDLE_CLOCK, 30_000, 10_000);
  // A word spoken at t=31 s, presented 2.5 s late, turns at monotonic 13.5 s.
  const turnAt = monotonicTimeForAcousticMs(clock, 31_000, 2_500);
  assert.equal(turnAt, 13_500);
  // Re-deriving the CSS delay at any later moment stays consistent, which is
  // what lets a remount resume rather than restart the motion.
  assert.equal(turnAt - 10_000, 3_500);
  assert.equal(turnAt - 13_000, 500);
  assert.equal(turnAt - 14_000, -500);
});

test("read-ahead is what the delay buys over recognizer latency", () => {
  const clock = advanceClock(IDLE_CLOCK, 30_000, 10_000);
  // The recognizer has delivered up to t=28.9 s (1.1 s behind the capture
  // head). The playhead sits at 27.5 s, so 1.4 s of text is on screen in
  // white, ahead of the colour. That is CWI 2.2.1, measured.
  assert.equal(readAheadMs(clock, 28_900, 10_000, 2_500), 1_400);
  // A delay shorter than the recognizer's latency buys nothing at all.
  assert.equal(readAheadMs(clock, 28_900, 10_000, 900), 0);
});

test("a word whose acoustic turn is ahead of the floor is untouched", () => {
  // Onset comfortably in the future: use acoustic time, do not clamp, do not
  // touch the floor chain. This is the fast-speech path -- its overlapping
  // pops are the design working and must not be staggered.
  const m = turnMomentFor(5_000, 1_000, 420, Number.NEGATIVE_INFINITY, 60);
  assert.equal(m.clamped, false);
  assert.equal(m.turnAtMs, 5_000);
});

test("a caught-up word takes the read-ahead floor, never less", () => {
  // Acoustic turn already in the past -> floor = now + minReadAhead.
  const m = turnMomentFor(200, 1_000, 420, Number.NEGATIVE_INFINITY, 60);
  assert.equal(m.clamped, true);
  assert.equal(m.turnAtMs, 1_420);
});

test("a burst of caught-up words RIPPLES instead of popping as one wall", () => {
  // Four words released in one commit: same `now`, all behind the playhead.
  // Without chaining they would share turnAtMs = 1_420 and pop in one frame.
  // Each carries the previous floor turn forward, exactly as the hook does.
  const now = 1_000;
  const gap = 60;
  let last = Number.NEGATIVE_INFINITY;
  const turns: number[] = [];
  for (const acoustic of [100, 250, 500, 900]) {
    const m = turnMomentFor(acoustic, now, 420, last, gap);
    assert.equal(m.clamped, true);
    last = m.turnAtMs;
    turns.push(m.turnAtMs);
  }
  assert.deepEqual(turns, [1_420, 1_480, 1_540, 1_600]);
  // The wall is gone: no two words share a turn moment.
  assert.equal(new Set(turns).size, turns.length);
});

test("the floor chain resets once the playhead catches up", () => {
  // A stale chain value from an earlier burst must not delay a later word that
  // arrives after a quiet gap: `max(floor, last+gap)` lets the floor win.
  const stale = 1_600;
  const m = turnMomentFor(200, 5_000, 420, stale, 60);
  assert.equal(m.turnAtMs, 5_420); // floor, not stale+gap
});

test("an acoustic-timed word does not advance the floor chain", () => {
  // A future-onset word between two caught-up bursts must leave the chain
  // where it was, or it would inject a phantom gap. The hook only advances
  // `lastFloorTurn` when `clamped` is true; assert the flag that gates it.
  assert.equal(turnMomentFor(9_000, 1_000, 420, 2_000, 60).clamped, false);
});
