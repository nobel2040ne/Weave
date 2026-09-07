import assert from "node:assert/strict";
import test from "node:test";
import {
  initialBearingState,
  smoothBearing,
  type BearingState,
} from "./bearing-ballistics.ts";

const TAU = 400;
const STEP = 120;   // one level event period

/** Feed a bearing repeatedly, one event period apart. */
const hold = (from: BearingState, deg: number | null, events: number) => {
  let s = from;
  for (let i = 0; i < events; i++) {
    s = smoothBearing(s, deg, s.atMs + STEP, TAU);
  }
  return s;
};

test("the first reading is taken exactly, not eased in from nowhere", () => {
  const s = smoothBearing(initialBearingState, 90, 1000, TAU);
  assert.equal(s.deg, 90);
});

test("a step is approached gradually, not snapped to", () => {
  const start = smoothBearing(initialBearingState, 0, 0, TAU);
  const one = smoothBearing(start, 90, STEP, TAU);
  assert.ok(
    one.deg !== null && one.deg > 5 && one.deg < 45,
    `one event should cover part of the step, got ${one.deg}`,
  );
});

test("...and converges on it if the talker stays put", () => {
  const start = smoothBearing(initialBearingState, 0, 0, TAU);
  const settled = hold(start, 90, 40);
  assert.ok(
    settled.deg !== null && Math.abs(settled.deg - 90) < 0.5,
    `should have arrived, got ${settled.deg}`,
  );
});

test("IT TAKES THE SHORT WAY ROUND THE SEAM, never the long way", () => {
  // 350 -> 10 is 20deg. Averaging linearly would walk it through 180.
  let s = smoothBearing(initialBearingState, 350, 0, TAU);
  for (let i = 0; i < 12; i++) {
    s = smoothBearing(s, 10, s.atMs + STEP, TAU);
    const d = s.deg as number;
    const nearSeam = d >= 340 || d <= 20;
    assert.ok(nearSeam, `walked the long way: reached ${d}`);
  }
  assert.ok(Math.abs((s.deg as number) - 10) < 1);
});

test("an absent reading HOLDS rather than drifting anywhere", () => {
  const start = smoothBearing(initialBearingState, 200, 0, TAU);
  const paused = hold(start, null, 20);
  assert.equal(paused.deg, 200, "the last measured bearing is kept");
});

test("tau 0 disables the smoothing entirely", () => {
  const start = smoothBearing(initialBearingState, 0, 0, 0);
  const next = smoothBearing(start, 170, STEP, 0);
  assert.equal(next.deg, 170, "passes straight through");
});

test("a bigger tau moves less per event", () => {
  const start = smoothBearing(initialBearingState, 0, 0, TAU);
  const fast = smoothBearing(start, 90, STEP, 150);
  const slow = smoothBearing(start, 90, STEP, 900);
  assert.ok(
    (fast.deg as number) > (slow.deg as number),
    `fast ${fast.deg} should lead slow ${slow.deg}`,
  );
});

test("a late event moves further, because dt is real elapsed time", () => {
  // Otherwise a dropped event quietly changes how fast the arrow travels.
  const start = smoothBearing(initialBearingState, 0, 0, TAU);
  const onTime = smoothBearing(start, 90, STEP, TAU);
  const late = smoothBearing(start, 90, STEP * 3, TAU);
  assert.ok((late.deg as number) > (onTime.deg as number));
});

test("every output is a real angle in 0..360", () => {
  let s = initialBearingState;
  for (const deg of [0, 359, 180, 181, 1, 270, 90, 350, 10, 45, null, 300]) {
    s = smoothBearing(s, deg, s.atMs + STEP, TAU);
    if (s.deg === null) continue;
    assert.ok(
      Number.isFinite(s.deg) && s.deg >= 0 && s.deg < 360,
      `got ${s.deg} after ${deg}`,
    );
  }
});
