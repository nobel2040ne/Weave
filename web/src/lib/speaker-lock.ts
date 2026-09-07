/* COLOUR IS FROZEN A FIXED TIME AFTER IT IS DECIDED, NOT WHEN THE WORD ARRIVES.
 *
 * Text is frozen by the playhead (`settledTextRef`): the moment a word turns,
 * its letters are history. Colour cannot use that clock. Text arrives at a
 * median 0.62 s and durable attribution at 4.48 s, so a word turns long before
 * anyone knows who said it -- which is why the turn is deliberately NOT gated
 * on attribution, and why a later correction is allowed to ease the colour in.
 *
 * What that permits, and what this closes, is the endpoint pass repainting a
 * word the viewer read minutes ago. Measured on real Korean dialogue, 18% of
 * words flip `provisional -> corrected`; on spontaneous multi-speaker Korean
 * the flicker runs 7.5-18%. Past some age a recolour stops being a correction
 * and becomes noise: the viewer has already attributed the line and moved on,
 * so changing it costs more than the residual error it fixes.
 *
 * So the clock starts when the word is FIRST given a speaker, not when it
 * appears. Starting it at arrival would freeze most words grey, because grey
 * is what `speaker: null` means and attribution routinely lands seconds later.
 * While the lock is open the speaker may still change freely -- that is the
 * endpoint pass doing its job. When it closes, whatever colour is being worn
 * at that moment is the answer.
 *
 * This is a DISPLAY freeze. `SpeakerTracker` still corrects itself, the events
 * still carry the tracker's latest word, and every probe scoring attribution
 * accuracy sees the unfrozen truth. Only the painted word stops moving.
 */

/** What a word was first decided to be, and when. */
export type SpeakerLock = {
  /** The speaker being worn. Updated while the lock is open. */
  speaker: string;
  /** When the word was FIRST given any speaker. Never moves. */
  decidedAt: number;
};

export type SpeakerLockResult = {
  /** The speaker the word should display. */
  speaker: string | null | undefined;
  /** The lock to remember, or undefined while the word is still undecided. */
  lock: SpeakerLock | undefined;
  /** True when a real recolour was refused -- countable, for the diagnostics. */
  frozen: boolean;
};

/** Has this word been wearing a colour long enough to keep it? */
export function resolveSpeakerLock(
  lock: SpeakerLock | undefined,
  incoming: string | null | undefined,
  nowMs: number,
  lockMs: number,
): SpeakerLockResult {
  // Not a positive duration: the feature is off and nothing is remembered.
  if (!(lockMs > 0)) return {speaker: incoming, lock, frozen: false};

  const decided = typeof incoming === "string" && incoming.length > 0;

  if (!lock) {
    // An undecided word stays undecided. Grey is reserved for `speaker: null`
    // and must not start a clock, or the first real attribution -- which is
    // the one the design system actually cares about -- would arrive locked
    // out.
    if (!decided) return {speaker: incoming, lock: undefined, frozen: false};
    return {
      speaker: incoming,
      lock: {speaker: incoming, decidedAt: nowMs},
      frozen: false,
    };
  }

  if (nowMs - lock.decidedAt >= lockMs) {
    // Closed. Report the refusal only when a decided speaker actually
    // disagreed; a `null` from a later hypothesis is not a correction, and
    // counting it would make the diagnostics read as churn that never was.
    return {
      speaker: lock.speaker,
      lock,
      frozen: decided && incoming !== lock.speaker,
    };
  }

  // Still open. `decidedAt` does NOT move: the window is measured from the
  // first colour, so a word being fought over cannot postpone its own
  // deadline indefinitely.
  if (!decided) return {speaker: incoming, lock, frozen: false};
  return {
    speaker: incoming,
    lock: {speaker: incoming, decidedAt: lock.decidedAt},
    frozen: false,
  };
}
