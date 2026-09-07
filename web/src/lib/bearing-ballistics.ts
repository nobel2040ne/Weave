/* THE BEARING IS SMOOTHED BEFORE CSS, FOR THE SAME REASON THE LEVEL IS.
 *
 * `direction_deg` is a per-block measurement arriving every
 * `level_event_period_s` (~120 ms), and the array HOLDS its last bearing while
 * a room is quiet rather than blanking -- so the channel's natural shape is
 * long flat stretches broken by steps, which is exactly the "sits still, then
 * suddenly jumps" the dial was showing.
 *
 * A LONGER CSS TRANSITION IS NOT THE FIX, and this project already measured
 * that on the level channel (2026-08-13, "SMOOTHED, NOT SHRUNK"): a 340 ms ease
 * against a 120 ms event period never converges, so the mark just chases the
 * input and never settles. The needle's transition was 180 ms against the same
 * period. Smoothing has to happen to the VALUE, before it reaches CSS.
 *
 * Circular, not linear: 350 deg and 10 deg average to 0, not to 180. The state
 * is carried as a unit vector and eased there, which makes the wrap free and
 * means a bearing sweeping past the seam glides rather than unwinding the long
 * way round.
 *
 * Deliberately NOT asymmetric. `useMeterBallistics` uses a fast attack so a
 * speech onset still registers, because a level meter that misses onsets is
 * lying about the sound. A bearing has no such asymmetry: a large step is
 * either a different talker or noise, and in both cases a glide is what the
 * viewer wants -- snapping to it is the complaint, not the feature.
 *
 * This feeds `updateSectorNeedles`, so the zone handover keys off the SMOOTHED
 * bearing. That is deliberate too: a noisy bearing sitting on a zone edge would
 * otherwise hand the arrow back and forth on the noise alone.
 */

/** Time for the arrow to cover ~63% of a step. 0 disables the smoothing. */
export const DEFAULT_BEARING_TAU_MS = 400;

const norm = (deg: number): number => ((deg % 360) + 360) % 360;

export type BearingState = {
  /** The smoothed bearing, degrees, or null before the first reading. */
  deg: number | null;
  /** When it was last advanced, ms, for the dt the easing is scaled by. */
  atMs: number;
};

export const initialBearingState: BearingState = {deg: null, atMs: 0};

/** Ease the held bearing toward `incoming`, the short way round.
 *
 *  `dt` is real elapsed time rather than a fixed per-event step, so a dropped
 *  or late event does not quietly change how fast the arrow moves. */
export function smoothBearing(
  state: BearingState,
  incoming: number | null,
  nowMs: number,
  tauMs: number = DEFAULT_BEARING_TAU_MS,
): BearingState {
  // Nothing measured: HOLD. Easing toward a default would walk the arrow
  // somewhere nobody spoke from, and the dial says "not measured" already.
  if (incoming === null || !Number.isFinite(incoming)) {
    return state.deg === null ? state : {...state, atMs: nowMs};
  }
  const target = norm(incoming);

  // First reading, or smoothing switched off: take it exactly. Easing in from
  // nowhere would draw a sweep from an arbitrary starting angle.
  if (state.deg === null || !(tauMs > 0)) {
    return {deg: target, atMs: nowMs};
  }

  const dt = Math.max(0, nowMs - state.atMs);
  const alpha = 1 - Math.exp(-dt / tauMs);
  if (alpha <= 0) return state;

  // Ease as a vector so the wrap costs nothing and the path is always the
  // short way round.
  const rad = Math.PI / 180;
  const sin = Math.sin(state.deg * rad) * (1 - alpha)
    + Math.sin(target * rad) * alpha;
  const cos = Math.cos(state.deg * rad) * (1 - alpha)
    + Math.cos(target * rad) * alpha;

  // Both components collapsing means the previous and target bearings are
  // opposite and have cancelled: there is no short way round, so take the
  // target rather than returning an angle atan2 invented from noise.
  if (Math.abs(sin) < 1e-9 && Math.abs(cos) < 1e-9) {
    return {deg: target, atMs: nowMs};
  }
  return {deg: norm(Math.atan2(sin, cos) / rad), atMs: nowMs};
}
