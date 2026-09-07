import assert from "node:assert/strict";
import test from "node:test";
import {
  COMPASS_SECTORS,
  SECTOR_SPAN_DEG,
  initialSectorState,
  sectorCentre,
  sectorFor,
  updateSectorNeedles,
  type SectorState,
} from "./compass-sectors.ts";

/** The clamp lands on `edge - EDGE_EPSILON`, which is a float: 120 - 0.05 is
    119.94999999999999. Compare with a tolerance rather than pinning the exact
    binary value into an assertion. */
const near = (actual: number, expected: number, why: string) =>
  assert.ok(
    Math.abs(actual - expected) < 1e-6,
    `${why}: expected ~${expected}, got ${actual}`,
  );

const feed = (bearings: Array<number | null>): SectorState =>
  bearings.reduce<SectorState>(
    (state, deg) => updateSectorNeedles(state, deg),
    initialSectorState(),
  );

test("the dial divides into three equal sectors", () => {
  assert.equal(COMPASS_SECTORS, 3);
  assert.equal(SECTOR_SPAN_DEG, 120);
  assert.equal(sectorFor(0), 0);
  assert.equal(sectorFor(119.9), 0);
  assert.equal(sectorFor(120), 1);
  assert.equal(sectorFor(239.9), 1);
  assert.equal(sectorFor(240), 2);
  assert.equal(sectorFor(359.9), 2);
});

test("sectorFor wraps rather than falling off either end", () => {
  assert.equal(sectorFor(360), 0);
  assert.equal(sectorFor(-1), 2);
  assert.equal(sectorFor(720 + 130), 1);
});

test("ALL THREE ARROWS EXIST FROM THE FIRST FRAME, resting at their centres", () => {
  // Drawing an arrow only once its zone had been heard was the first attempt,
  // and it read as no change: one talker sitting still is one zone, so one
  // arrow -- exactly what was there before.
  const state = initialSectorState();
  assert.equal(state.active, null);
  assert.deepEqual(state.needles.map((n) => n.angleDeg), [60, 180, 300]);
  assert.deepEqual(state.needles.map((n) => n.heard), [false, false, false]);
  for (const [i, needle] of state.needles.entries()) {
    assert.equal(sectorFor(needle.angleDeg), i);
    assert.equal(needle.angleDeg, sectorCentre(i));
  }
});

test("a bearing lights only its own sector's needle", () => {
  const state = feed([200]);
  assert.equal(state.active, 1);
  assert.deepEqual(state.needles.map((n) => n.angleDeg), [60, 200, 300]);
  assert.deepEqual(
    state.needles.map((n) => n.heard), [false, true, false],
    "the other two are resting, not pointing at anything measured",
  );
});

test("NO NEEDLE EVER LEAVES ITS SECTOR, which is the whole point", () => {
  // The pathological case the single needle handled worst: a bearing that
  // crosses the 0/360 seam used to sweep the long way round the dial. Here it
  // is two different needles, so nothing travels. 30deg rather than 10deg
  // because 10 is inside the seam's hysteresis band and is deliberately kept
  // by sector 2 -- that case is its own test below.
  const state = feed([350, 30, 30]);
  assert.equal(state.active, 0);
  assert.equal(state.needles[2].angleDeg, 350, "sector 2 holds where it was");
  assert.equal(state.needles[0].angleDeg, 30, "sector 0 takes the new bearing");
  for (const [i, needle] of state.needles.entries()) {
    assert.equal(
      sectorFor(needle.angleDeg), i,
      `needle ${i} is at ${needle.angleDeg}, outside its own sector`,
    );
  }
});

test("an idle needle holds its last bearing instead of blanking", () => {
  const state = feed([200, null]);
  assert.equal(state.active, null, "nothing is live");
  assert.equal(state.needles[1].angleDeg, 200, "but the room is not forgotten");
  assert.equal(
    state.latest, 1,
    "and it stays the bright one -- a pause must not fade the whole dial",
  );
});

test("hysteresis survives a pause, so resuming on a boundary does not hand over", () => {
  // `active` is null across the pause; keying the hysteresis on it would let
  // the first event back jump sectors, which is the flicker this prevents.
  const state = feed([110, null, 125]);
  assert.equal(state.active, 0);
  near(state.needles[0].angleDeg, 119.95, "a hair inside its own zone");
  assert.equal(state.needles[1].heard, false, "zone 1 was never handed it");
});

test("a bearing just past the edge stays with the needle that had it", () => {
  // Hysteresis default is 12deg, so 125 is still sector 0's while it is active.
  const state = feed([110, 125]);
  assert.equal(state.active, 0);
  near(
    state.needles[0].angleDeg, 119.95,
    "pinned just inside the edge, so it stays in its own sector",
  );
  assert.equal(state.needles[1].heard, false, "sector 1 was never handed it");
});

test("...but hands over once the bearing is clearly into the neighbour", () => {
  // Two updates: the crossing parks on the seam, the next reaches the bearing.
  const state = feed([110, 140, 140]);
  assert.equal(state.active, 1);
  assert.equal(state.needles[1].angleDeg, 140);
});

test("hysteresis holds across the 0/360 seam, not just the inner edges", () => {
  // 355 is sector 2's; 3 is 8deg past the seam, inside the 12deg band.
  const state = feed([355, 3]);
  assert.equal(state.active, 2, "the seam is a boundary like any other");
  near(state.needles[2].angleDeg, 359.95, "pinned just inside the seam");
  assert.equal(state.needles[0].heard, false);
});

test("a talker sitting ON a boundary does not alternate needles", () => {
  // This is what hysteresis exists for: without it the two readings below
  // would light different needles every event.
  const state = feed([118, 122, 118, 122, 119]);
  assert.equal(state.active, 0);
  assert.equal(state.needles[1].heard, false);
});

test("zero hysteresis hands over at the boundary exactly", () => {
  let state = updateSectorNeedles(initialSectorState(), 110, 0);
  state = updateSectorNeedles(state, 121, 0);
  assert.equal(state.active, 1, "handed over at the boundary");
  state = updateSectorNeedles(state, 121, 0);
  assert.equal(state.needles[1].angleDeg, 121);
});

test("a pause does not reset which needles have been placed", () => {
  const state = feed([30, 200, 200, null, null]);
  assert.deepEqual(state.needles.map((n) => n.angleDeg), [30, 200, 300]);
  assert.deepEqual(state.needles.map((n) => n.heard), [true, true, false]);
});

test("INVARIANT: there are always exactly three arrows", () => {
  // The count is fixed, not a function of how many speakers or zones have been
  // heard. Nothing on the dial may add or remove one.
  for (const feedIn of [[], [10], [10, 200], [10, 200, 300, 45, 181, 359]]) {
    const state = feed(feedIn);
    assert.equal(
      state.needles.length, 3,
      `after ${JSON.stringify(feedIn)} there were ${state.needles.length}`,
    );
  }
});

test("INVARIANT: every arrow is inside its OWN zone, whatever it is fed", () => {
  // Includes the hysteresis clamp, which used to land on 120.000 exactly --
  // a value `sectorFor` reads as the next zone, so the arrow was sitting in
  // its neighbour's third by the dial's own arithmetic.
  let state = initialSectorState();
  const bearings = [
    0, 119.9, 120, 120.1, 125, 239, 240, 241, 355, 359.9, 360, 3, 60,
    -5, 725, 118, 122, 118, 180, 300, 61, null, 240.05, 119.95,
  ];
  for (const deg of bearings) {
    state = updateSectorNeedles(state, deg);
    for (const [i, needle] of state.needles.entries()) {
      assert.equal(
        sectorFor(needle.angleDeg), i,
        `after ${deg}: arrow ${i} sits at ${needle.angleDeg}, `
        + `which belongs to zone ${sectorFor(needle.angleDeg)}`,
      );
      assert.ok(
        needle.angleDeg >= 0 && needle.angleDeg < 360,
        `arrow ${i} is at ${needle.angleDeg}, outside 0..360`,
      );
    }
  }
});

test("a handover parks the incoming arrow ON THE SEAM, not at its centre", () => {
  // Coming in from the resting centre was a swing of up to 60deg that the
  // talker never made -- the stutter at the crossing.
  const state = feed([110, 140]);
  assert.equal(state.active, 1);
  assert.equal(state.entering, true, "this update is the crossing");
  near(state.needles[1].angleDeg, 120, "parked on the shared edge");
  assert.notEqual(state.needles[1].angleDeg, 180, "NOT its resting centre");
});

test("...and reaches the real bearing on the next update, eased", () => {
  const state = feed([110, 140, 140]);
  assert.equal(state.entering, false, "no longer entering");
  assert.equal(state.needles[1].angleDeg, 140);
});

test("the seam it enters from is the one it actually crossed", () => {
  // Zone 1 borders zone 0 at 120 and zone 2 at 240. Arriving from zone 2 must
  // enter at 240, not at 120. 200 rather than 230, because 230 is inside zone
  // 2's hysteresis band and never hands over at all.
  const state = feed([250, 200]);
  assert.equal(state.active, 1);
  near(state.needles[1].angleDeg, 239.95, "entered at the 240 seam");
});

test("entering never parks an arrow outside its own zone", () => {
  let state = initialSectorState();
  for (const deg of [110, 140, 250, 230, 10, 350, 130, 5, 200, 300, 61]) {
    state = updateSectorNeedles(state, deg);
    for (const [i, needle] of state.needles.entries()) {
      assert.equal(
        sectorFor(needle.angleDeg), i,
        `after ${deg}: arrow ${i} parked at ${needle.angleDeg}`,
      );
    }
  }
});

test("a pause clears `entering` so the arrow is not frozen mid-crossing", () => {
  const state = feed([110, 140, null]);
  assert.equal(state.entering, false);
  assert.equal(state.latest, 1, "but the zone is still the bright one");
});
