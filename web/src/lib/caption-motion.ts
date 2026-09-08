/** The one-shot caption motion described by CWI 2.2.3 and 2.3. */

export interface VoiceTypeRanges {
  /** Reachable multiples of the 2.3.5 baseline. The PDF's span is 0.6..2.4. */
  scale: readonly [number, number];
  /** How much of the 2.3.6 excursion is used above the 2.3.5 baseline. */
  scaleResponse: number;
  /** ...and below it. Deliberately smaller. The two directions are not
     symmetric in what they cost. */
  scaleResponseQuiet: number;
  /** Fraction of each side's range where the size does not move at all. */
  scaleDeadband: number;
  /** Where the size mapping pivots, as a fraction of normalised loudness. */
  scalePivot?: number;
  /** Exponent applied to the re-spanned deviation before the response. 1 is
     the straight mapping legacy uses. */
  scaleCurve?: number;
  /** Control points `[normalisedLoudness, crestAboveOne]`, ascending in both. */
  scalePoints?: Array<[number, number]>;
  /** Reachable `font-weight`. 400 is fixed at the 2.3.8 neutral band. */
  weight: readonly [number, number];
  /** 2.3.8's neutral band in Hz: every voice inside it is Regular 400. THIS
     is what decides how far apart two speakers look — `weight` only bounds
     the extremes, which real median F0s never reach. */
  toneBand?: readonly [number, number];
  /** 2.3.9's vocal span in Hz, the domain the band sits inside. */
  toneSpan?: readonly [number, number];
  /** How far the glyph's PROPORTION follows its weight, 0..1. A thin word
     grows taller and narrower, a heavy one shorter and wider — 2.3.10's
     diagonal extended to the vertical, which the PDF does not specify. */
  proportionCoupling?: number;
  /** How much of the WORD's own pitch enters the weight, 0..1, against the
     speaker's running median. 0 is weight as pure speaker identity — one
     talker is one weight however they modulate; 1 is 2.3.9 read literally
     per word. Between the two the speaker's register is the anchor and the
     word's pitch moves it. */
  tonePitchTracking?: number;
  /** How much of the weight range an emphasised word takes, on top of 2.3.9. */
  weightEmphasis: number;
  /** Reachable `font-stretch` %. 100 is the neutral width. */
  width: readonly [number, number];
}

export interface CaptionType {
  scale: number;
  weight: number;
  width: number;
}

export interface CaptionMotionPlan {
  /** The exact type before and after motion, and it is ALWAYS normal.
     `docs/MOTION.md` § "Return to normal": every word begins and ends at
     5% / Regular 400 / width 100. The PDF's static pages show voice type
     persisting, the recordings show it returning, and the user chose
     returning. Do not put the voice in here. */
  rest: CaptionType;
  /** 2.3.9's register half alone — the voice with no emphasis in it. A CREST
     target, not a rest state: it is what an unemphasised word rises to, so
     the register renders on its way up instead of never rendering at all. */
  registerWeight: number;
  /** CWI 2.3 type at the expressive crest of the motion. */
  voice: CaptionType;
  /** CWI 2.2.3's eye-guiding cue: ONE growth, anchored at the baseline. There
     is no elevation term. */
  sync: {
    scale: number;
  };
}

export const NORMAL_CAPTION_TYPE: Readonly<CaptionType> = Object.freeze({
  scale: 1,
  weight: 400,
  width: 100,
});

/** CWI 2.3.6: smallest 3%, largest 12%; 2.3.5: normal baseline 5%. */
const SIZE_MIN_PCT = 3;
const SIZE_MAX_PCT = 12;
const SIZE_BASELINE_PCT = 5;

/** CWI 2.3.8/2.3.9 pitch anchors — the fallback when the served runtime
   config carries none. `config.yaml` owns the live values. */
const PITCH_NEUTRAL_LOW_HZ = 160;
const PITCH_NEUTRAL_HIGH_HZ = 200;
const PITCH_FLOOR_HZ = 80;
const PITCH_CEILING_HZ = 250;

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.max(minimum, Math.min(maximum, value));

/** Low/rich (+1) through neutral (0) to high/thin (-1). The whole neutral
   band maps to 0, so ordinary F0 jitter moves no weight at all. */
export function voiceTone(
  pitchHz: number,
  band: readonly [number, number] = [PITCH_NEUTRAL_LOW_HZ, PITCH_NEUTRAL_HIGH_HZ],
  span: readonly [number, number] = [PITCH_FLOOR_HZ, PITCH_CEILING_HZ],
): number {
  if (!Number.isFinite(pitchHz) || pitchHz <= 0) return 0;
  // A malformed band must not divide by zero and paint every voice Regular.
  const low = Math.min(band[0], band[1]);
  const high = Math.max(band[0], band[1]);
  const floor = Math.min(span[0], low);
  const ceiling = Math.max(span[1], high);
  if (pitchHz < low) {
    const reach = low - floor;
    return reach > 1e-6 ? (low - Math.max(pitchHz, floor)) / reach : 0;
  }
  if (pitchHz <= high) return 0;
  const reach = ceiling - high;
  return reach > 1e-6
    ? -(Math.min(pitchHz, ceiling) - high) / reach
    : 0;
}

/** Where the 2.3.5 baseline sits on the server's normalised 0..1 loudness. */
const LOUDNESS_PIVOT =
  (SIZE_BASELINE_PCT - SIZE_MIN_PCT) / (SIZE_MAX_PCT - SIZE_MIN_PCT);

/** Volume -> type size, anchored on §2.3.5 and bounded by §2.3.6. THE
   DEADBAND IS WHAT MAKES THE QUIET HALF USABLE. */
export function voiceScale(
  loudness: number,
  ranges: VoiceTypeRanges,
): number {
  const level = clamp(Number.isFinite(loudness) ? loudness : 0.5, 0, 1);
  const points = ranges.scalePoints;
  if (points && points.length > 1) {
    // Piecewise-linear through the fitted control points. Monotone by
    // construction, so a louder word can never render smaller than a quieter
    // one -- the property the parametric mapping guaranteed for free.
    let above = 1 + points[points.length - 1][1];
    for (let i = 1; i < points.length; i += 1) {
      const [x0, y0] = points[i - 1];
      const [x1, y1] = points[i];
      if (level <= x1) {
        const span = x1 - x0;
        const t = span > 1e-9 ? (level - x0) / span : 0;
        above = 1 + y0 + clamp(t, 0, 1) * (y1 - y0);
        break;
      }
    }
    return clamp(above, ranges.scale[0], ranges.scale[1]);
  }
  /* THE PIVOT IS WHY OUR DISTRIBUTION WAS TWO-HUMPED. */
  const pivot = clamp(ranges.scalePivot ?? LOUDNESS_PIVOT, 0, 0.99);
  const deviation = level - pivot;
  const sideRange = deviation >= 0 ? 1 - pivot : Math.max(1e-6, pivot);
  const dead = clamp(ranges.scaleDeadband, 0, 0.95) * sideRange;
  const [minimum, maximum] = ranges.scale;
  if (Math.abs(deviation) <= dead) return 1;

  // Re-span what is left of the side so the mapping stays continuous at the
  // band edge and still reaches 2.3.6's limit at the extreme.
  const spanned = (Math.abs(deviation) - dead) / Math.max(1e-6, sideRange - dead);
  const shaped = Math.pow(clamp(spanned, 0, 1),
                          Math.max(0.1, ranges.scaleCurve ?? 1));
  /* WHY `voice_scale_range[0]` DOES NOT MOVE THE QUIET FLOOR (2026-08-04). */
  const limit = deviation >= 0
    ? SIZE_MAX_PCT / SIZE_BASELINE_PCT
    : SIZE_MIN_PCT / SIZE_BASELINE_PCT;
  const response = deviation >= 0
    ? ranges.scaleResponse
    : ranges.scaleResponseQuiet;
  return clamp(1 + response * shaped * (limit - 1), minimum, maximum);
}

/** Pitch -> type weight, with §2.3.8's complete neutral band held at 400. */
export function voiceWeight(
  tone: number,
  ranges: VoiceTypeRanges,
  prominence = 0,
): number {
  const [floor, ceiling] = ranges.weight;
  const pushed = clamp(Number.isFinite(prominence) ? prominence : 0, 0, 1);
  const bold = ceiling - 400;
  const light = 400 - floor;
  // The register half, withdrawn as the voice is pushed: a pressed voice is
  // not an airy one, whatever its pitch.
  const register = tone < 0 ? tone * (1 - pushed) : tone;
  const shaped = register > 0 ? register * bold : register * light;
  // ...and the prominence half. The light half is now shallow enough
  // (`weight_range`'s floor is set from the reference's own worst lightening,
  // -53 from Regular) that this can never be out-run by it.
  const pressed = pushed * clamp(ranges.weightEmphasis ?? 0, 0, 1) * bold;
  return Math.round(clamp(400 + shaped + pressed, floor, ceiling));
}

/** Roboto Flex's vertical axes at their defaults. NotoSansKR carries none of
   them and ignores the lot, exactly as it already ignores `wdth`. */
const YTLC_DEFAULT = 514, YTLC_MIN = 416, YTLC_MAX = 570;
const YTUC_DEFAULT = 712, YTUC_MIN = 528, YTUC_MAX = 760;
const YTAS_DEFAULT = 750, YTAS_MIN = 649, YTAS_MAX = 854;

export interface VoiceProportion {
  /** Lowercase height. */
  ytlc: number;
  /** Uppercase height. */
  ytuc: number;
  /** Ascender height. */
  ytas: number;
}

/** THE GLYPH'S PROPORTION, FOLLOWING ITS WEIGHT (2026-09-09, at the user's
   request). Thin reads tall and narrow, heavy reads short and wide — the
   vertical half of 2.3.10's diagonal, which the PDF pairs with width but
   never states for height, so this is a deviation and it has a coupling of
   its own to turn off.

   Done on the typeface's OWN axes rather than a `scaleY`: Roboto Flex draws
   real height changes, where a transform would smear the stems and would
   also have to fight the hold-spring for the `transform` property. The axes
   are asymmetric — there is far more room below the defaults than above — so
   heavy words shorten more than thin words grow, which is how the face was
   drawn and not something to correct. */
export function voiceProportion(
  weight: number,
  ranges: VoiceTypeRanges,
): VoiceProportion {
  const [floor, ceiling] = ranges.weight;
  const coupling = clamp(
    Number.isFinite(ranges.proportionCoupling as number)
      ? (ranges.proportionCoupling as number) : 0, 0, 1);
  const side = weight >= 400
    ? Math.max(1e-6, ceiling - 400)
    : Math.max(1e-6, 400 - floor);
  /* +1 is as thin as this voice gets, -1 as heavy. */
  const lift = clamp(-(weight - 400) / side, -1, 1) * coupling;
  const axis = (base: number, low: number, high: number) =>
    Math.round(base + lift * (lift > 0 ? high - base : base - low));
  return {
    ytlc: axis(YTLC_DEFAULT, YTLC_MIN, YTLC_MAX),
    ytuc: axis(YTUC_DEFAULT, YTUC_MIN, YTUC_MAX),
    ytas: axis(YTAS_DEFAULT, YTAS_MIN, YTAS_MAX),
  };
}

/** Harmonics -> width, constrained to §2.3.10's heavy+wide / light+condensed
   diagonal. */
export function voiceWidth(
  tone: number,
  texture: number,
  ranges: VoiceTypeRanges,
): number {
  const harmonics = Number.isFinite(texture)
    ? clamp((0.5 - texture) * 2, -1, 1)
    : 0;
  let blended = clamp(tone * 0.7 + harmonics * 0.3, -1, 1);
  if (tone > 0) blended = Math.max(blended, 0);
  if (tone < 0) blended = Math.min(blended, 0);
  const [floor, ceiling] = ranges.width;
  const shaped = blended > 0
    ? blended * (ceiling - 100)
    : blended * (100 - floor);
  return Math.round(clamp(100 + shaped, floor, ceiling));
}

/** The size band `voiceScale` can ACTUALLY produce, which is not
   `ranges.scale`. */
export function reachableScaleRange(
  ranges: VoiceTypeRanges,
): [number, number] {
  const loud = Math.min(
    ranges.scale[1],
    1 + ranges.scaleResponse * (SIZE_MAX_PCT / SIZE_BASELINE_PCT - 1),
  );
  const quiet = Math.max(
    ranges.scale[0],
    1 - ranges.scaleResponseQuiet * (1 - SIZE_MIN_PCT / SIZE_BASELINE_PCT),
  );

  return [quiet, loud];
}

/** How loud this word is for its speaker, 0..1, BEFORE the size deadband. */
export function prominenceOf(loudness: number): number {
  const level = clamp(Number.isFinite(loudness) ? loudness : LOUDNESS_PIVOT, 0, 1);
  return clamp((level - LOUDNESS_PIVOT) / (1 - LOUDNESS_PIVOT), 0, 1);
}

/** How far up the reachable §2.3.6 range this word's size sits, 0..1. */
export function emphasisOf(scale: number, ranges: VoiceTypeRanges): number {
  const top = Math.max(reachableScaleRange(ranges)[1], 1 + 1e-6);
  return clamp((scale - 1) / (top - 1), 0, 1);
}

/** How far this word's size departs from normal, 0..1, on whichever side it
   is. */
export function voiceDeviationOf(
  scale: number,
  ranges: VoiceTypeRanges,
): number {
  const [quiet, loud] = reachableScaleRange(ranges);
  const span = scale >= 1
    ? Math.max(1e-6, loud - 1)
    : Math.max(1e-6, 1 - quiet);
  return clamp(Math.abs(scale - 1) / span, 0, 1);
}

/** The tone that drives WEIGHT: the speaker's register, moved by how this
   word's own pitch departs from it.

   2.3.9 is a property of the voice, and a running median is what makes one
   talker read as one weight. But it is also what makes a talker who
   deliberately drops or lifts their pitch render identically either way — on
   a demo stage the presenter modulates and nothing moves. `tonePitchTracking`
   is how much of the word's own pitch is allowed in. A word with no voiced
   pitch keeps the speaker's register rather than drifting to Regular. */
export function weightToneFor(
  {pitchHz, registerHz}: {pitchHz: number; registerHz?: number},
  ranges: VoiceTypeRanges,
): number {
  const hasRegister = typeof registerHz === "number" &&
    Number.isFinite(registerHz) && registerHz > 0;
  const registerTone = voiceTone(
    hasRegister ? (registerHz as number) : pitchHz,
    ranges.toneBand, ranges.toneSpan);
  const tracking = clamp(
    Number.isFinite(ranges.tonePitchTracking as number)
      ? (ranges.tonePitchTracking as number) : 0, 0, 1);
  if (!tracking || !Number.isFinite(pitchHz) || pitchHz <= 0) return registerTone;
  const wordTone = voiceTone(pitchHz, ranges.toneBand, ranges.toneSpan);
  return registerTone + (wordTone - registerTone) * tracking;
}

export function voiceTypeFor(
  {loudness, pitchHz, texture, registerHz}: {
    loudness: number;
    pitchHz: number;
    texture: number;
    registerHz?: number;
  },
  ranges: VoiceTypeRanges,
): CaptionType {
  /* THE REGISTER IS THE ANCHOR (2026-08-03), AND THE WORD'S OWN PITCH MOVES
     IT (2026-09-09) -- see `weightToneFor`. A shout cannot render thin either
     way: `voiceWeight` withdraws the lightening as prominence rises. */
  const tone = weightToneFor({pitchHz, registerHz}, ranges);
  const scale = voiceScale(loudness, ranges);
  return {
    scale,
    weight: voiceWeight(tone, ranges, prominenceOf(loudness)),
    // Width stays on the WORD's pitch: 2.3.10's diagonal is about the sound of
    // the utterance, and unlike weight it has no Light floor to fall into.
    width: voiceWidth(
      voiceTone(pitchHz, ranges.toneBand, ranges.toneSpan), texture, ranges),
  };
}

/** Build the whole motion without inspecting rendered geometry. Expression
   only interpolates the §2.3 voice target. */
/** The word's own intonation contour, sampled evenly across its spoken span. */
export interface VoiceEnvelope {
  loudness: readonly number[];
  pitch: readonly number[];
  texture: readonly number[];
}

/** Linear interpolation of a sampled contour at a 0..1 position. */
export function sampleContour(
  values: readonly number[],
  position: number,
): number {
  if (!values.length) return Number.NaN;
  if (values.length === 1) return values[0];
  const at = clamp(position, 0, 1) * (values.length - 1);
  const low = Math.floor(at);
  const high = Math.min(values.length - 1, low + 1);
  return values[low] + (values[high] - values[low]) * (at - low);
}

/** CWI 2.3 IS PER CHARACTER. One `CaptionType` for each letter of the word. */
export function characterVoiceTypes(
  characterCount: number,
  envelope: VoiceEnvelope | null | undefined,
  wordVoice: {
    loudness: number; pitchHz: number; texture: number;
    /** The SPEAKER's median F0 -- 2.3.7's register half. */
    registerHz?: number;
  },
  ranges: VoiceTypeRanges,
  expression = 1,
): CaptionType[] {
  const count = Math.max(0, Math.floor(characterCount));
  const amount = clamp(Number.isFinite(expression) ? expression : 1, 0, 1);
  const types: CaptionType[] = [];
  for (let index = 0; index < count; index += 1) {
    const position = count === 1 ? 0.5 : (index + 0.5) / count;
    const loudness = envelope?.loudness?.length
      ? sampleContour(envelope.loudness, position)
      : wordVoice.loudness;
    const pitchHz = envelope?.pitch?.length
      ? sampleContour(envelope.pitch, position)
      : wordVoice.pitchHz;
    const texture = envelope?.texture?.length
      ? sampleContour(envelope.texture, position)
      : wordVoice.texture;
    const target = voiceTypeFor(
      {loudness, pitchHz, texture, registerHz: wordVoice.registerHz}, ranges);
    types.push({
      scale: 1 + (target.scale - 1) * amount,
      weight: Math.round(400 + (target.weight - 400) * amount),
      width: Math.round(100 + (target.width - 100) * amount),
    });
  }
  return types;
}

/** The widest any character of this word gets, for the layout reservation. */
export function peakCharacterScale(types: readonly CaptionType[]): number {
  return types.reduce((peak, type) => Math.max(peak, type.scale), 1);
}

export function captionMotionFor(
  voice: {
    loudness: number; pitchHz: number; texture: number;
    /** The SPEAKER's median F0 -- 2.3.7's register half. */
    registerHz?: number;
  },
  ranges: VoiceTypeRanges,
  expression: number,
  syncPop: number,
  /** What fraction of the pop an UNEMPHASISED word takes, 0..1.
     1 is the historic behaviour -- a flat step, the same for every word that
     pops at all -- and is the default so LEGACY is untouched. */
  syncPopFloor = 1,
): CaptionMotionPlan {
  const target = voiceTypeFor(voice, ranges);
  const amount = clamp(Number.isFinite(expression) ? expression : 1, 0, 1);
  const voiceScale = 1 + (target.scale - 1) * amount;
  /* THE VOICE'S OWN WEIGHT, WITH NO EMPHASIS IN IT. 2.3.9 makes weight a
     property of the SPEAKER — low voices heavier — and the film's hand reading
     says the same from the other side: weight is withheld per LINE, not spent
     per word. It is a CREST target, never a rest state (see `rest` above).
     What it buys is that `quietWord` used to send an unemphasised word to a
     flat 400, so for most words the register never rendered at any point in
     the motion; now such a word rises to its speaker's weight and returns. */
  const registerWeight = Math.round(
    400 + (voiceWeight(weightToneFor(voice, ranges), ranges, 0) - 400) * amount);
  /* THE POP IS PROPORTIONAL TO EMPHASIS, NOT A FIXED STEP.
     As a constant it was a binary decision on a continuous quantity: a word
     0.6% over the gate got exactly what the loudest word in the session got,
     and a word 0.6% under got nothing at all. That cannot produce the band
     the reference actually lives in -- across assets/reference_specs 90% of
     words move and 60% land between 1.02x and 1.15x -- at ANY gate threshold,
     because the only two reachable values were 1.000 and 1+syncPop. Moving
     the gate just traded "too many identical big pops" for "79% of words with
     no size motion", which is how this was found. */
  const floor = clamp(Number.isFinite(syncPopFloor) ? syncPopFloor : 1, 0, 1);
  const lean = floor + (1 - floor) * emphasisOf(voiceScale, ranges);
  return {
    rest: {...NORMAL_CAPTION_TYPE},
    registerWeight,
    voice: {
      scale: voiceScale,
      weight: Math.round(400 + (target.weight - 400) * amount),
      width: Math.round(100 + (target.width - 100) * amount),
    },
    sync: {
      scale: 1 + Math.max(0, syncPop) * lean,
    },
  };
}
