/* THREE NEEDLES, ONE PER 120 DEG SECTOR, INSTEAD OF ONE THAT CROSSES THE DIAL.
 *
 * The live needle followed `direction_deg` around the whole circle, so any
 * change of talker was a sweep: 350 deg to 10 deg is 20 deg of room and 340 deg
 * of travel, and even a legitimate move across the case read as the instrument
 * flinging itself about. The eye follows the SWEEP rather than the bearing.
 *
 * Splitting the dial into three fixed sectors bounds it. Each needle owns
 * [0,120), [120,240) or [240,360) and can never leave it, so the longest move
 * any needle can make is 120 deg and no needle ever crosses the middle. A
 * bearing that jumps between sectors is handed to a DIFFERENT needle, which is
 * a change of which line is lit rather than a line travelling.
 *
 * ALL THREE ARROWS ARE ALWAYS ON THE DIAL. Drawing one only once its zone had
 * been heard was the first attempt and it read as no change at all: a single
 * talker sitting still is one zone, so one arrow, which is what was there
 * before. The invariant the viewer should be able to rely on is that each
 * third of the room HAS an arrow -- so an unheard zone rests at its centre,
 * and a zone that has been heard holds its last real bearing rather than
 * springing back. The last bearing was measured; a centre never was, which is
 * why `heard` is carried separately and drawn differently.
 *
 * HYSTERESIS IS LOAD-BEARING, because the sector edges are arbitrary and a
 * talker may sit on one. Without it, someone at 119-121 deg alternates between
 * two needles every event and the dial flickers worse than the single needle
 * ever did. A bearing just past the boundary stays with the needle that
 * already had it, pinned to the edge, until it is clearly into the neighbour.
 *
 * All angles here are DIAL degrees -- `compassBearing` has already applied the
 * 180 deg display offset. Nothing in this file reaches the event stream or
 * `autocwi/haptics.py`.
 */

export const COMPASS_SECTORS = 3;
export const SECTOR_SPAN_DEG = 360 / COMPASS_SECTORS;
/** How far past its edge a needle keeps a bearing before handing it over. */
export const SECTOR_HYSTERESIS_DEG = 12;
/* Clamping to the edge EXACTLY would put the arrow on `120.000`, which
   `sectorFor` reads as the NEXT sector -- so an arrow held by hysteresis would
   be sitting, by the dial's own arithmetic, in its neighbour's third. Landing
   a hair inside keeps "every arrow is inside its own zone" true as a checkable
   property rather than approximately true. Below the 0.1deg the style is
   rendered at, so nothing moves on screen. */
const EDGE_EPSILON_DEG = 0.05;

export type SectorNeedle = {
  /** Dial degrees, always inside this needle's own sector. Starts at the
      sector's centre so all three arrows are on the dial from the first
      frame -- the point is that each zone HAS an arrow, not that arrows
      appear as zones are visited. */
  angleDeg: number;
  /** Has this zone actually heard anything? Its arrow is drawn either way;
      this is what separates "resting" from "pointing at a real bearing". */
  heard: boolean;
};

export type SectorState = {
  needles: SectorNeedle[];
  /** Which sector owns the live bearing, or null when nothing is measured. */
  active: number | null;
  /** True on the update where the live zone CHANGED. The incoming arrow has
      been parked on the shared edge and must not animate into it -- see
      `entryEdge`. One update later it eases from there to the real bearing,
      so the lit tip crosses the seam instead of appearing mid-zone. */
  entering: boolean;
  /** The last sector that WAS live, held across pauses. `active` is the honest
      answer to "is anything being heard right now"; this is what the dial
      draws bright, because a pause must not fade the whole instrument -- the
      dial already says "not measured now" through `data-measured`. */
  latest: number | null;
};

const norm = (deg: number): number => ((deg % 360) + 360) % 360;

/** The middle of a sector: where its arrow rests before the zone is heard. */
export function sectorCentre(index: number): number {
  return index * SECTOR_SPAN_DEG + SECTOR_SPAN_DEG / 2;
}

export function initialSectorState(): SectorState {
  return {
    needles: Array.from({length: COMPASS_SECTORS}, (_, i) => ({
      angleDeg: sectorCentre(i),
      heard: false,
    })),
    active: null,
    latest: null,
    entering: false,
  };
}

/** Circular separation between two bearings, degrees, 0..180. */
function apart(a: number, b: number): number {
  const d = Math.abs(norm(a) - norm(b));
  return d > 180 ? 360 - d : d;
}

/** WHERE A ZONE IS ENTERED FROM. On a handover the outgoing arrow is sitting
 *  on the boundary (hysteresis pinned it there), so the incoming one starts at
 *  that same boundary and eases inward. Coming in from its resting centre
 *  instead was a swing of up to 60 deg in whatever direction the zone's middle
 *  happened to be -- motion the talker never made, which is what read as a
 *  stutter at the crossing. */
function entryEdge(index: number, fromAngle: number): number {
  const start = norm(index * SECTOR_SPAN_DEG);
  const end = norm(index * SECTOR_SPAN_DEG + SECTOR_SPAN_DEG - EDGE_EPSILON_DEG);
  return apart(fromAngle, start) <= apart(fromAngle, end) ? start : end;
}

/** Which needle owns this bearing, ignoring hysteresis. */
export function sectorFor(dialDeg: number): number {
  return Math.min(
    COMPASS_SECTORS - 1,
    Math.floor(norm(dialDeg) / SECTOR_SPAN_DEG),
  );
}

/** Where `dialDeg` sits relative to sector `index`, and what it clamps to.
 *  `over` is how far outside the sector it is, 0 when inside. */
function against(dialDeg: number, index: number): {over: number; angle: number} {
  const start = index * SECTOR_SPAN_DEG;
  // Position of the bearing measured forward from the sector's start edge.
  const rel = norm(dialDeg - start);
  if (rel < SECTOR_SPAN_DEG) return {over: 0, angle: norm(dialDeg)};
  const pastEnd = rel - SECTOR_SPAN_DEG;
  const beforeStart = 360 - rel;
  return pastEnd <= beforeStart
    ? {over: pastEnd, angle: norm(start + SECTOR_SPAN_DEG - EDGE_EPSILON_DEG)}
    : {over: beforeStart, angle: norm(start)};
}

/** Route one bearing to its needle. Every other needle holds what it had. */
export function updateSectorNeedles(
  state: SectorState,
  dialDeg: number | null,
  hysteresisDeg: number = SECTOR_HYSTERESIS_DEG,
): SectorState {
  // Nothing measured: no needle is live, and every needle keeps its bearing.
  // Deliberately NOT a reset -- a pause must not blank the room.
  if (dialDeg === null || !Number.isFinite(dialDeg)) {
    // `latest` deliberately survives: the needle that was live stays the one
    // drawn bright, holding its last bearing.
    return state.active === null && !state.entering
      ? state
      : {...state, active: null, entering: false};
  }

  let index = sectorFor(dialDeg);
  let angle = norm(dialDeg);

  // The needle that already had it keeps it while the bearing is only just
  // over the line, pinned to that line so it stays inside its own sector.
  // Keyed on `latest`, not `active`, so the hysteresis survives a pause -- a
  // talker who stops and resumes on a boundary would otherwise hand over on
  // the first event back, which is the flicker this exists to prevent.
  const holder = state.active ?? state.latest;
  if (holder !== null && holder !== index) {
    const held = against(dialDeg, holder);
    if (held.over > 0 && held.over <= hysteresisDeg) {
      index = holder;
      angle = held.angle;
    }
  }

  /* A HANDOVER PARKS THE INCOMING ARROW ON THE SEAM AND STOPS THERE for this
     update. It reaches the real bearing on the next one, eased, so the two
     arrows read as one mark crossing a line rather than one vanishing and
     another appearing 60 deg away. Costs a single event period (~120 ms) of
     lag at the crossing, which is what buys the continuity. */
  const handover = state.latest !== null && state.latest !== index;
  const placed = handover
    ? entryEdge(index, state.needles[state.latest as number].angleDeg)
    : angle;

  const needles = state.needles.map((needle, i) =>
    i === index ? {angleDeg: placed, heard: true} : needle,
  );
  return {needles, active: index, latest: index, entering: handover};
}
