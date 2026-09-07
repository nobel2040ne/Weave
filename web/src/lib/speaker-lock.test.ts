import assert from "node:assert/strict";
import test from "node:test";
import {resolveSpeakerLock, type SpeakerLock} from "./speaker-lock.ts";

const LOCK_MS = 6000;

test("an undecided word starts no clock", () => {
  // Grey is `speaker: null`, and attribution lands at a median 4.48s. Starting
  // the window on arrival would lock most words grey before anyone decided.
  const result = resolveSpeakerLock(undefined, null, 1000, LOCK_MS);
  assert.equal(result.speaker, null);
  assert.equal(result.lock, undefined);
  assert.equal(result.frozen, false);
});

test("the first decided speaker starts the clock and passes through", () => {
  const result = resolveSpeakerLock(undefined, "S1", 1000, LOCK_MS);
  assert.equal(result.speaker, "S1");
  assert.deepEqual(result.lock, {speaker: "S1", decidedAt: 1000});
  assert.equal(result.frozen, false);
});

test("a correction inside the window is applied", () => {
  const lock: SpeakerLock = {speaker: "S1", decidedAt: 1000};
  const result = resolveSpeakerLock(lock, "S2", 1000 + LOCK_MS - 1, LOCK_MS);
  assert.equal(result.speaker, "S2");
  assert.equal(result.frozen, false);
});

test("a correction inside the window does NOT postpone the deadline", () => {
  // Otherwise a word being fought over never settles: every flip would buy
  // another full window, which is the churn this exists to stop.
  const lock: SpeakerLock = {speaker: "S1", decidedAt: 1000};
  const mid = resolveSpeakerLock(lock, "S2", 4000, LOCK_MS);
  assert.deepEqual(mid.lock, {speaker: "S2", decidedAt: 1000});

  const after = resolveSpeakerLock(mid.lock, "S3", 1000 + LOCK_MS, LOCK_MS);
  assert.equal(after.speaker, "S2");
  assert.equal(after.frozen, true);
});

test("a correction after the window is refused", () => {
  const lock: SpeakerLock = {speaker: "S1", decidedAt: 1000};
  const result = resolveSpeakerLock(lock, "S2", 1000 + LOCK_MS, LOCK_MS);
  assert.equal(result.speaker, "S1");
  assert.equal(result.frozen, true);
});

test("re-asserting the SAME speaker after the window is not a refusal", () => {
  // The endpoint pass re-emits words it agrees with. Counting those would make
  // the frozen tally read as flicker that never happened.
  const lock: SpeakerLock = {speaker: "S1", decidedAt: 1000};
  const result = resolveSpeakerLock(lock, "S1", 9999999, LOCK_MS);
  assert.equal(result.speaker, "S1");
  assert.equal(result.frozen, false);
});

test("a null after the window keeps the colour and is not a refusal", () => {
  // A later hypothesis with no attribution is not a correction, and letting it
  // through would drop a coloured word back to grey -- flicker in the other
  // direction.
  const lock: SpeakerLock = {speaker: "S1", decidedAt: 1000};
  const result = resolveSpeakerLock(lock, null, 9999999, LOCK_MS);
  assert.equal(result.speaker, "S1");
  assert.equal(result.frozen, false);
});

test("zero or negative disables the lock entirely", () => {
  for (const off of [0, -1]) {
    const lock: SpeakerLock = {speaker: "S1", decidedAt: 0};
    const result = resolveSpeakerLock(lock, "S2", 9999999, off);
    assert.equal(result.speaker, "S2", `lockMs=${off} must pass through`);
    assert.equal(result.frozen, false);
  }
});
