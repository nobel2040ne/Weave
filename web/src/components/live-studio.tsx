"use client";

import {
  AudioWaveform,
  AlignEndHorizontal,
  Captions,
  CircleGauge,
  Eye,
  Globe2,
  Mic2,
  Minus,
  PanelRightClose,
  PanelRightOpen,
  SlidersHorizontal,
  Sparkles,
  Sun,
  Type,
  X,
} from "lucide-react";
import {
  memo,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import {
  useCaptionStream,
  type LanguageSession,
  type LiveLanguageOption,
  type RuntimeConfig,
} from "@/hooks/use-caption-stream";
import {
  buildCaptionParagraphs,
  planCaptionStackMotion,
  selectStableCaptionStack,
  type CaptionParagraph,
  createStageMemory,
  type StageMemory,
  type CaptionStackPosition,
} from "@/lib/caption-paragraphs";
import type {CaptionWord, LevelEvent} from "@/lib/caption-store";
import {
  captionMotionFor,
  characterVoiceTypes,
  emphasisOf,
  voiceDeviationOf,
  type CaptionType,
  type VoiceTypeRanges,
} from "@/lib/caption-motion";
import {
  acousticTimeMs,
  charTurnDelayMs,
  crestDurationMs,
  crestWindowMs,
  FILM_WORD_TURN_MS,
  HOLD_ENVELOPE_EMPHASIS,
  naturalMotionDurationMs,
} from "@/lib/motion-timing";
import {baselineOffsetEm, formatBaselineEm} from "@/lib/glyph-metrics";
import {waveGrain, wordIsWide} from "@/lib/hangul";
import {FILM_PEAK_FRACTION, fallDurationMs} from "@/lib/motion-timing";
import {groupLeads, liftsInGroups, syllableGroups} from "@/lib/syllables";

import {
  assignSpeakerColors,
  speakerColor,
  speakerStatus,
  type SpeakerColorMap,
} from "@/lib/speaker-colors";
import {planStageLayout, rowBudgetEm} from "@/lib/stage-layout";

type CSSVars = CSSProperties & Record<`--${string}`, string | number>;
type ViewMode = "stage" | "transcript";
interface SettingsState {
  captionScale: number;
  motionIntensity: number;
  reducedMotion: boolean;
  highContrast: boolean;
  lightStage: boolean;
  hangulWave: boolean;
  enhancedMotion: boolean;
  rollingCaptions: boolean;
  captionRules: boolean;
}

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

const STACK_SHIFT_DURATION_MS = 540;
const STACK_ENTER_DURATION_MS = 620;
const STACK_EASING = "cubic-bezier(.18,.72,.22,1)";
/* HOW LONG THE BOX TAKES TO OPEN over a newly appended word. */
const ROW_GROW_DURATION_MS = 160;
/* How far past its turn a word may still animate. */
const SETTLE_GRACE_MS = 70;

const number = (value: unknown, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

/** How much of a word's span was actually voiced, 1 when unknown. */
function voicedFraction(word: {voiced_frac?: number | null}): number {
  const value = Number(word?.voiced_frac);
  return Number.isFinite(value) && value > 0 ? Math.min(1, value) : 1;
}

/** The CWI 2.3 crest bounds, as the runtime config declares them. */
function voiceRanges(runtime: RuntimeConfig, enhanced = false): VoiceTypeRanges {
  return {
    scale: runtime.voiceScaleRange,
    scaleResponse: enhanced
      ? runtime.voiceScaleResponseEnhanced
      : runtime.voiceScaleResponse,
    scaleResponseQuiet: runtime.voiceScaleResponseQuiet,
    scaleDeadband: enhanced
      ? runtime.voiceScaleDeadbandEnhanced
      : runtime.voiceScaleDeadband,
    ...(enhanced ? {scalePivot: runtime.voiceScalePivotEnhanced,
                    scaleCurve: runtime.voiceScaleCurveEnhanced,
                    scalePoints: runtime.voiceScalePointsEnhanced} : {}),
    weight: runtime.weightRange,
    weightEmphasis: runtime.weightEmphasis,
    width: runtime.widthRange,
  };
}

function speakerNumber(speaker: string | null): string {
  if (!speaker) return "—";
  const match = speaker.match(/\d+/);
  return String(match ? Number(match[0]) : 1).padStart(2, "0");
}

/** Assign once per roster, then swap the themed palette by INDEX, so the
   light stage cannot renumber anybody. */
function useSpeakerColors(
  paragraphs: CaptionParagraph[],
  runtime: RuntimeConfig,
  lightStage: boolean,
): SpeakerColorMap {
  return useMemo(() => {
    // Order of FIRST APPEARANCE, which is what config.yaml has always claimed.
    const order: string[] = [];
    const seen = new Set<string>();
    for (const paragraph of paragraphs) {
      for (const {word} of paragraph.words) {
        const speaker = word.speaker;
        if (!speaker || seen.has(speaker)) continue;
        seen.add(speaker);
        order.push(speaker);
      }
    }
    const mains = runtime.palette.length - runtime.paletteSupportCount;
    const assigned = assignSpeakerColors(order, {
      main: runtime.palette.slice(0, mains),
      support: runtime.palette.slice(mains),
    });
    if (!lightStage || !runtime.paletteLight.length) return assigned;
    const themed: SpeakerColorMap = new Map();
    for (const [speaker, entry] of assigned) {
      themed.set(speaker, {
        ...entry,
        color: entry.index >= 0
          ? runtime.paletteLight[entry.index] ?? entry.color
          : entry.color,
      });
    }
    return themed;
  }, [paragraphs, runtime, lightStage]);
}

function useElapsed(startedAt: number): string {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const seconds = Math.max(0, Math.floor((now - startedAt) / 1000));
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

/* The character spans of a word that have never been given a turn moment. */
function unarmedCharacters(element: HTMLElement): HTMLElement[] {
  return Array.from(
    element.querySelectorAll<HTMLElement>(".caption-character, .character-sizer"),
  ).filter((span) => !span.style.getPropertyValue("--char-turn-delay"));
}

/** Per-word values that must survive a word being rebuilt in another row. */
interface WordMemo {
  /** The motion clock: frozen the first time the word was ever drawn. */
  clock: Map<string, {duration: number; sweepMs: number; crestMs: number}>;
  /** The resolved hold, once the gate has closed on it. */
  hold: Map<string, number>;
  /* THE 2.3 VOICE AXES, frozen at first sight like the clock beside them. */
  voice: Map<string, ReturnType<typeof captionMotionFor>>;
}

const MotionWord = memo(function MotionWord({
  id,
  word,
  color,
  intensity,
  runtime,
  holdGapS,
  holdSettled,
  paceGapS,
  clockEpoch,
  scheduleWord,
  memo,
  hangulWave,
  enhancedMotion,
}: {
  id: string;
  word: CaptionWord;
  color: string;
  intensity: number;
  runtime: RuntimeConfig;
  /** Settings toggle: give wide-script characters their own Hangul grain. */
  hangulWave: boolean;
  /** Settings toggle: anticipate the crest and sweep the whole spoken word. */
  enhancedMotion: boolean;
  /* WHAT THIS WORD FROZE AT FIRST SIGHT, KEPT OUTSIDE THE WORD (2026-08-06). */
  memo: WordMemo;
  /** Silence before this word, in seconds -- how long it had to wait. */
  holdGapS: number;
  /* Whether `holdGapS` is FINAL -- the parent has seen this word's next
     neighbour. */
  holdSettled: boolean;
  /** Onset-to-onset interval to the NEXT word: one word at this speech rate. */
  paceGapS: number;
  clockEpoch: number | null;
  scheduleWord: (
    id: string,
    word: CaptionWord,
    durationMs: number,
  ) => {turnAtMs: number; epoch: number} | null;
}) {
  const wordRef = useRef<HTMLSpanElement>(null);
  const armedRef = useRef<{turnAtMs: number; epoch: number} | null>(null);
  // How many letters the colour wipe was laid out across when this word armed.
  const charSpanRef = useRef<number | null>(null);
  const motionWord = word;
  const loudness = clamp(number(motionWord.loudness, 0.5), 0, 1);
  const deliveryConfidence = clamp(
    number(motionWord.delivery_confidence),
    0,
    1,
  );
  const deliveryEnabled = runtime.deliveryMotionEnabled &&
    deliveryConfidence >= runtime.deliveryMinConfidence;
  // 0.5 is the neutral of the breathiness proxy, so an unavailable reading
  // leaves the width axis on the pitch term alone rather than pushing it wide.
  const texture = deliveryEnabled
    ? clamp(number(motionWord.delivery_texture, 0.5), 0, 1)
    : 0.5;
  // Retained as a readout only. CWI has no delivery-specific motion families:
  // every word receives the same 2.2.3 cue while the continuous 2.3 axes shape
  // the glyph inside that one motion window.
  const deliveryProfile = deliveryEnabled
    ? String(motionWord.delivery_profile ?? "steady")
    : "steady";
  const pitch = number(motionWord.pitch_hz, 0);
  /* The SPEAKER's running median F0. */
  const register = number(motionWord.pitch_register_hz, 0);
  /* FROZEN AT FIRST SIGHT -- see `WordMemo.voice`. */
  const motion = (() => {
    const remembered = memo.voice.get(id);
    if (remembered) return remembered;
    const fresh = captionMotionFor(
      {loudness, pitchHz: pitch, texture, registerHz: register},
      voiceRanges(runtime, enhancedMotion),
      intensity,
      /* 2.2.3's amplitude is per clock: the PDF's verbatim 15% on legacy,
         and the smaller figure the PR film actually renders on enhanced. */
      enhancedMotion ? runtime.syncPopEnhanced : runtime.syncPop,
      /* Enhanced scales the pop with emphasis; legacy keeps the flat step,
         which is what its 1 does. */
      enhancedMotion ? runtime.syncPopFloorEnhanced : 1,
    );
    memo.voice.set(id, fresh);
    return fresh;
  })();

  /* `Array.from`, not `split("")`: a Hangul syllable block or any astral
     character must stay one unit. */
  const characters = Array.from(word.text);
  /* Which colour-turn path this word takes. */
  const isWide = wordIsWide(word.text);
  /* WHICH LETTERS LEAVE THE LINE TOGETHER. */
  /* Only a word of six letters or more raises its syllables independently --
     see `liftsInGroups`. */
  /* Bounded by `scripts/ink_collision.py`: this travels toward the row
     above, on top of the word-level lift. */
  const groupLift = !isWide && liftsInGroups(word.text) ? ".155em" : "0em";
  const waveLead = isWide
    ? characters.map((_, i) => i)
    : (() => {
        const groups = syllableGroups(word.text);
        const leads = groupLeads(groups);
        return characters.map((_, i) => leads[groups[i] ?? 0] ?? i);
      })();
  /* The envelope no longer sets each character's TYPE -- 2.3 is per word
     (see below). */
  /* Measured against the REACHABLE size band, not the configured clamp:
     the response stops short of the clamp on the quiet side. */
  const voiceDeviation = voiceDeviationOf(
    motion.voice.scale,
    voiceRanges(runtime),
  );
  const waveSuppression = Math.max(
    runtime.characterWaveFloor,
    1 - voiceDeviation * runtime.characterWaveFalloff,
  );

  const characterTypes: CaptionType[] = characterVoiceTypes(
    characters.length,
    motionWord.env_loudness && motionWord.env_pitch && motionWord.env_texture
      ? {
          loudness: motionWord.env_loudness,
          pitch: motionWord.env_pitch,
          texture: motionWord.env_texture,
        }
      : null,
    {loudness, pitchHz: pitch, texture, registerHz: register},
    voiceRanges(runtime),
    intensity,
  );

  // One span per character, keyed by INDEX so a verifier respelling reuses the
  // existing spans instead of remounting the word and restarting its motion.
  const renderCharacters = (crest: boolean) =>
    characters.map((character, index) => (
      <span
        className={crest ? "character-sizer" : "caption-character"}
        key={index}
        data-char-index={index}
        style={crest ? {"--char-index": index} as CSSVars : {
          // Where this letter sits in the wave, and how hard it stretches.
          "--char-index": index,
          /* EACH LETTER TILTS SLIGHTLY DIFFERENTLY, and letters move in
             SYLLABLES. */
          "--char-tilt": `${(Math.sin(index * 2.399) * 1.35).toFixed(2)}deg`,
          /* HOW FAR THIS LETTER'S SYLLABLE LEAVES THE LINE. */
          "--group-lift-em": groupLift,
          "--char-group": waveLead[index] ?? index,
          /* THE TWO SCOPES TRADE OFF. */
          "--char-wave": (waveSuppression * clamp(
            0.85 + ((characterTypes[index]?.scale ?? 1) - motion.voice.scale) * 1.6,
            0.45,
            1.30,
          )).toFixed(3),
          /* Hangul-structural grain, wide script + toggle only. */
          ...(hangulWave && isWide ? (() => {
            const {ay, ax} = waveGrain(character.codePointAt(0) ?? 0);
            return {"--wave-ay": ay.toFixed(4), "--wave-ax": ax.toFixed(4)};
          })() : {}),
        } as CSSVars}
      >
        {character === " " ? "\u00a0" : character}
      </span>
    ));

  /* Frozen at mount: a verifier respelling can revise `end`, and a caption
     already in flight must not have its clock reshaped underneath it. */
  /* How much of the hold this word earns, 0..1, from the silence around it. */
  /* AND A WORD THAT SWELLS DOES NOT LIFT. THE TWO CHANNELS ARE INDEPENDENT. */
  /* ONE-SIDED: only SWELLING withdraws the lift, not shrinking. */
  const holdEmphasis = motion.voice.scale > 1
    ? emphasisOf(motion.voice.scale, voiceRanges(runtime))
    : 0;
  /* FROZEN AT MOUNT, like the duration and the axes -- and for a sharper
     reason than either. */
  /* ...AND THE FREEZE IS AT THE TURN, NOT AT THE MOUNT (2026-08-05).
     Freezing is right -- see above. */
  const holdTarget = clamp(
    (holdGapS - runtime.holdMinS) /
      Math.max(1e-6, runtime.holdFullS - runtime.holdMinS),
    0,
    1,
  ) * (holdEmphasis >= HOLD_ENVELOPE_EMPHASIS ? 0 : 1);
  /* Remembered outside this component, so a word rebuilt in another row does
     not forget what it already earned. */
  const [holdAmount, setHoldAmount] = useState(
    () => memo.hold.get(id) ?? holdTarget,
  );
  const holdFrozen = useRef(memo.hold.has(id));
  useEffect(() => {
    if (holdFrozen.current) return;
    const armed = armedRef.current;
    if (armed && performance.now() >= armed.turnAtMs) {
      // The playhead has passed: history. Keep what it actually wore.
      holdFrozen.current = true;
      memo.hold.set(id, holdAmount);
      return;
    }
    if (!holdSettled) return;
    holdFrozen.current = true;
    memo.hold.set(id, holdTarget);
    setHoldAmount((current) => (current === holdTarget ? current : holdTarget));
  }, [holdSettled, holdTarget, holdAmount, id, memo]);
  const holdEnvelope = holdEmphasis >= HOLD_ENVELOPE_EMPHASIS;
  /* How long this word takes to come back down: long enough to reach the
     next word's turn, so the stage is never still while anyone is talking. */
  const fallRef = useRef<number | null>(null);
  /* ...AND IT STOPS ACCEPTING ONE ONCE THE WORD HAS TURNED. */
  const turnedRef = useRef(false);
  if (fallRef.current === null && paceGapS > 0 && !turnedRef.current) {
    fallRef.current = paceGapS;
  }

  const [{duration, sweepMs, crestMs}] = useState(() => {
    /* Frozen the first time this WORD was drawn, not this component -- they
       are different moments once a word can change row. */
    const remembered = memo.clock.get(id);
    if (remembered) return remembered;
    /* One word at the current speech rate: the AE template's range selector
       is exactly one word wide. */
    const push = emphasisOf(motion.voice.scale, voiceRanges(runtime));
    const naturalMs = naturalMotionDurationMs(motionWord, runtime, paceGapS);
    const spokenMs = Math.max(0, number(word.end) - number(word.start)) * 1000;
    /* Legacy finishes the wipe before the word is done being said -- a
       boundary travelling across the characters, which is 2.2.2. The PR film
       does not: its range selector is in WORD units, so a word turns at once. */
    const sweep = enhancedMotion
      ? FILM_WORD_TURN_MS
      : clamp(spokenMs * 0.72, 0, runtime.wordMotionMaxMs);
    /* HOW LONG THE SIZE CHANGE LASTS -- and the legacy answer is far too
       long. */
    /* AND THE BIGGER THE SWELL, THE LONGER THE WINDOW. */
    const crest = enhancedMotion
      ? Math.min(
          runtime.wordMotionMaxMs,
          runtime.wordMotionEnhancedMs + push * runtime.wordMotionEnhancedEmphasisMs,
        )
      : crestDurationMs(
          sweep,
          // The word's REAL speech: `end` runs to the next onset, so the raw
          // span is an inter-onset interval and `voiced_frac` is what makes it
          // speech. This anchors the crest's floor to the speaker.
          crestWindowMs(push, runtime, spokenMs * voicedFraction(word)),
          push, runtime.wordMotionMaxMs,
        );
    const clock = {
      duration: naturalMs,
      sweepMs: sweep,
      // The crest is the SLOW clock -- emphasis, not speech rate.
      crestMs: crest,
    };
    memo.clock.set(id, clock);
    return clock;
  });
  /* A word with no emphasis holds Regular -- see `--voice-weight` below. */
  const quietWord = enhancedMotion && motion.voice.scale <= 1.005;
  const style: CSSVars = {
    "--speaker-color": color,
    /* The word's normalised loudness, published so `scripts/motion_diff.py`
       can read the distribution the size mapping is fed. */
    "--voice-loudness": loudness.toFixed(4),
    /* The size cue trails the turn on the enhanced clock and starts with it on
       legacy, where `calc(X + 0ms)` is X and the delay arithmetic is untouched. */
    "--crest-lag": `${enhancedMotion ? runtime.crestLagMs : 0}ms`,
    /* EVERY WORD POPS, BY AN AMOUNT THAT IS NOT THE SAME (2026-08-20).
       This used to read `quietWord ? 1 : motion.sync.scale` -- a binary gate
       that gave a word either the whole pop or none of it. Both of that gate's
       settings were the wrong shape: loose, it drew forty identical pops at
       2.5 words/s and read as noise; tight, it sent 79% of words to exactly
       1.000 and the size channel went flat. The reference does neither -- 90%
       of its words move and 60% of them only slightly -- so the amplitude is
       proportional now (`sync_pop_floor_enhanced`) and there is nothing left
       for a gate to decide. `quietWord` still governs WEIGHT, which the film
       really does withhold, per line rather than per word. */
    "--sync-pop": motion.sync.scale.toFixed(3),
    /* ...AND WHAT AN UNEMPHASISED WORD DOES INSTEAD IS LIFT. */
    "--word-lift-em": `${runtime.wordLiftEmEnhanced}em`,
    "--motion-duration": `${duration.toFixed(0)}ms`,
    // The 2.3 crest takes its own window so its rise tracks the colour wipe
    // instead of leading it (`crestDurationMs`); the pop and wave keep the
    // natural window.
    "--crest-duration": `${crestMs.toFixed(0)}ms`,
    /* The wide-script wipe sweeps over exactly this window. */
    "--sweep-duration": `${sweepMs.toFixed(0)}ms`,
    // Anticipation. 0ms on legacy -> `calc(turn-delay - 0ms)` is turn-delay.
    /* Ordinary words pulse; emphatic ones rise, HOLD and fall. A rise time
       and a fall time cannot express the second shape, which is why fitting
       two endpoints kept failing -- `animation-name` picks between them. */
    "--voice-envelope": holdEnvelope
      ? "voice-phase-hold"
      : (enhancedMotion ? "voice-phase-film" : "voice-phase"),
    /* The two halves of the split envelope. Only read when `data-tail` is
       `extended`; harmless otherwise. */
    "--rise-ms": `${(crestMs * FILM_PEAK_FRACTION).toFixed(0)}ms`,
    "--fall-ms": `${fallDurationMs(crestMs, fallRef.current ?? 0).toFixed(0)}ms`,
    "--sync-envelope": enhancedMotion ? "word-sync-pop-film" : "word-sync-pop",
    // CWI 2.3 is a WORD-level property: in intonation.mov every glyph of
    // "louder" is the same size and weight, and every glyph of "softer" is
    // uniformly small. Only the wave below is per character.
    /* A LIFTED WORD IS AT REST. THE EXCLUSION RUNS BOTH WAYS. */
    "--voice-scale": (1 + (motion.voice.scale - 1) * (1 - holdAmount))
      .toFixed(3),
    /* AN UNEMPHASISED WORD HOLDS REGULAR. */
    "--voice-weight": String(
      quietWord
        ? 400
        : Math.round(400 + (motion.voice.weight - 400) * (1 - holdAmount)),
    ),
    "--voice-width": `${motion.voice.width}%`,
    /* HOW LONG ONE GLYPH STAYS ELEVATED. */
    // The wave hands off letter to letter across ~55% of the window, so it
    // travels visibly instead of pulsing the word as one block.
    "--wave-span": `${(duration * 0.72).toFixed(0)}ms`,
    /* A word that waited crouches, springs, floats and lands as it turns. */
    "--hold-lift": `${(holdAmount * runtime.holdLiftEm).toFixed(3)}em`,
    "--hold-spring": holdAmount.toFixed(3),
    /* BINARY, unlike `--hold-spring`. */
    "--hold-gate": holdAmount > 0 ? "1" : "0",
    "--hold-pre": `${runtime.holdPreMs}ms`,
    "--hold-hold": `${runtime.holdHoldMs}ms`,
    "--hold-land": `${runtime.holdLandMs}ms`,
  };
  const status = speakerStatus(word);

  /* ARM THE WORD ONCE, THEN LEAVE IT ALONE. */
  useLayoutEffect(() => {
    const element = wordRef.current;
    if (!element) return;
    if (armedRef.current === null) {
      // No acoustic clock yet. `clockEpoch` brings us back when there is one.
      // The crest window is the word's longest-running animation, so it is
      // what `activeMotions` should count.
      const armed = scheduleWord(id, word, Math.max(duration, crestMs));
      if (armed === null) return;
      armedRef.current = armed;
    }
    /* A capture restart (a looping sample, a restarted server) invalidates
       the timeline this word was placed on. */
    if (clockEpoch !== null && armedRef.current.epoch !== clockEpoch) {
      element.dataset.armed = "stale";
      element.style.setProperty("--turn-delay", "-600000ms");
      // A character appended AFTER the restart settles with the word it belongs
      // to. Without this it keeps the stylesheet's 600s default and paints in
      // read-ahead ink beside its own already-coloured word -- see the loop
      // below, which is where that defect actually lived.
      for (const span of unarmedCharacters(element)) {
        span.style.setProperty("--char-turn-delay", "-600000ms");
        span.style.setProperty("--char-wave-delay", "-600000ms");
      }
      return;
    }
    const rearming = element.dataset.armed !== "true";
    element.dataset.armed = "true";
    const turnDelay = armedRef.current.turnAtMs - performance.now();
    // Past its turn this word is history: nothing about its motion may change
    // again. See `turnedRef` above.
    if (turnDelay <= 0) turnedRef.current = true;
    // The word's own delay is written ONCE: rewriting `animation-delay` shifts
    // a running animation, which is the hazard `data-armed` exists to prevent.
    if (rearming) {
      /* A WORD WHOSE TURN HAS PASSED SETTLES. */
      const settled = turnDelay < -SETTLE_GRACE_MS;
      element.style.setProperty(
        "--turn-delay",
        settled ? "-600000ms" : `${Math.round(turnDelay)}ms`,
      );
      // The wipe is laid out across the letters the word had WHEN IT WAS ARMED,
      // and that denominator is then frozen with the sweep. See the loop below.
      charSpanRef.current = Math.max(1, characters.length);
    }
    const perWord = charSpanRef.current ?? Math.max(1, characters.length);

    /* THE COLOUR TURN IS A WIPE THROUGH THE WORD, NOT A SWITCH. */
    // `perWord` is FROZEN AT THE ARM, not read from the current length:
    // appending to the denominator moves every existing letter's position in
    // the wipe, so a word that grew would hand a late character an EARLIER
    // delay than one already running ("Some" -> "Something": char 3 at .75
    // sweep, new char 4 at .44) and the boundary would visibly travel
    // backwards. Past the frozen length the wipe is over, so the tail turns
    // with its last letter. `sweepMs` freezes at mount alongside the crest
    // window, so the rise and the wipe are computed from the same number.
    // THE COLOUR TRAVELS PER LETTER; THE LIFT TRAVELS PER SYLLABLE. Watched on
    // the film, "seen" lifts as `se` + `en`, "button" as `but` + `ton` -- the
    // letters of a syllable leave the line together -- while the colour
    // boundary still crosses one letter at a time. So the wave gets its own
    // delay, read from the group's FIRST letter (`waveLead`), and every letter
    // of that group shares it. Frozen on the same terms as the turn delay, and
    // written in the same pass, so `unarmedCharacters` (which tests only
    // `--char-turn-delay`) can keep testing one property.
    for (const span of unarmedCharacters(element)) {
      const index = Number(span.dataset.charIndex ?? 0);
      span.style.setProperty(
        "--char-turn-delay",
        `${charTurnDelayMs(turnDelay, index, perWord, sweepMs)}ms`,
      );
      span.style.setProperty(
        "--char-wave-delay",
        `${charTurnDelayMs(turnDelay, waveLead[index] ?? index, perWord, sweepMs)}ms`,
      );
    }
    // `duration` is deliberately absent: like `crestMs` and `sweepMs` it is
    // frozen at mount, and re-running this effect would rewrite
    // `animation-delay` on a RUNNING animation, which shifts it -- the exact
    // hazard `data-armed` exists to prevent.
    // `characters.length` IS load-bearing: it is what brings the effect back
    // when a word grows, and every new span is armed on that pass.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clockEpoch, crestMs, id, scheduleWord, sweepMs, word, characters.length]);

  return (
    <span
      className="caption-word"
      /* Unconditional on the enhanced clock, and that is the point: it used
         to also require a known next onset, so a word whose neighbour arrived
         late flipped AFTER it had turned -- an animation-name change, which
         restarts the animation and runs the motion a second time. */
      data-tail={
        enhancedMotion && !holdEnvelope ? "extended" : "natural"
      }
      data-status={status}
      data-final={word.final ? "true" : "false"}
      data-sustain={word.sustain_active ? "true" : "false"}
      data-delivery={deliveryProfile}
      // CWI 2.1.5: an off-camera voice keeps its speaker colour and is set in
      // italic. Live capture has no camera and never sets this; it is here so
      // the SSE contract can carry the distinction rather than inventing it.
      data-off-camera={word.off_camera ? "true" : "false"}
      /* Wide scripts take the continuous wipe: a Hangul block is 0.91em
         against Latin's 0.43em, so a per-glyph step is a switch, not a
         sweep. `wordIsWide` can never return true for an all-Latin word. */
      data-script={isWide ? "wide" : "narrow"}
      data-word-id={id}
      style={style}
      ref={wordRef}
      title={`${deliveryProfile} delivery · ${
        pitch > 0 ? `${Math.round(pitch)} Hz` : "unvoiced"
      } · ${number(word.loudness_db, -72).toFixed(1)} dB · ${
        motion.voice.weight
      }/${motion.voice.width}%`}
    >
      <span className="word-sizer word-sizer-normal" aria-hidden="true">
        {word.text}
      </span>
      <span className="word-sizer word-sizer-crest" aria-hidden="true">
        {renderCharacters(true)}
      </span>
      <span className="word-glyph" aria-label={word.text}>
        <span className="word-ink" aria-hidden="true">
          {renderCharacters(false)}
        </span>
        {/* THE CONTINUOUS WIPE, wide script only. A second ink layer in the
            speaker colour, revealed by an animated clip. It carries the SAME
            character spans as the base, so both layers stretch identically and
            stay registered while the wave runs -- precisely what a
            `background-clip: text` gradient could NOT do, because the gradient
            lives in the ancestor's coordinate space while the wave is a
            transform on the descendants. Checked on a spike before building.
            The speaker colour is inherited from `.caption-word`, so a late
            attribution correction rewrites `--speaker-color` and recolours this
            layer directly -- the same property the `to`-less colour keyframe
            gives the narrow path. */}
        {isWide ? (
          <span className="word-ink word-ink-turned" aria-hidden="true">
            {renderCharacters(false)}
          </span>
        ) : null}
      </span>
    </span>
  );
});

/* THE DIAL IS DRAWN 180 DEG ROUND FROM THE ARRAY'S OWN FRAME (2026-08-14, at
   the user's direction, decided watching a real talker beside the case). The
   array's 0 deg and the dial's front tick were pointing opposite ways, so a
   speaker in front of the case drew a needle behind it. This rotates the
   BEARINGS -- needle, speaker arcs and the spoken label -- and leaves the tick
   scale alone: the front tick is the case's front and is the frame everything
   else is read against, so turning it too would move the reference instead of
   the reading.
   DISPLAY ONLY. `direction_deg` in the event stream and the bearing
   `autocwi/haptics.py` drives motors from are untouched -- if the motors point
   the wrong way as well, that is the same sign error at the source and belongs
   in the node, not here. */
const COMPASS_BEARING_OFFSET_DEG = 180;

/* How many per-speaker arrows the dial draws. Three reads as a room; more
   turns into a star-burst. The arcs still mark every speaker beyond this. */
const COMPASS_ARROWS = 3;
/* The XVF3800 steers exactly two talker beams; the other two slots are the
   free-running and auto-select beams, which can be echoing one of them. */
const COMPASS_BEAMS = 2;

const compassBearing = (deg: number): number =>
  (((deg + COMPASS_BEARING_OFFSET_DEG) % 360) + 360) % 360;

/* METER BALLISTICS FOR THE DIAL'S LEVEL CHANNELS.
   `rms_db` is a per-block measurement arriving every `level_event_period_s`
   (120 ms), and it is not a smooth signal: MEASURED over 191 live events, its
   frame-to-frame step is a median of 3.9 dB and a p90 of 12.6 dB. That reaches
   `--orb-scale`, which is a `transform: scale()` on `.voice-compass` ITSELF, so
   every tick, arc and needle moves with it -- a p90 step of 12.6 dB is 0.13x of
   scale, about 26 px of travel on a 200 px dial, eight times a second.

   Lengthening the CSS transition was tried first (2026-08-13, "SMOOTHED, NOT
   SHRUNK") and cannot fix it: a 340 ms ease against a 120 ms event period never
   reaches its target before the next one arrives, so the dial just chases
   continuously. The step has to be smaller before it reaches CSS.

   Asymmetric on purpose, the way any level meter is: a fast attack so a real
   onset still registers as one, a slow release so the dial settles instead of
   chattering between syllables. DISPLAY ONLY -- `level` itself is untouched, so
   the event stream, the probes and `autocwi/haptics.py` all still see the raw
   measurement. */
const METER_ATTACK_TAU_S = 0.18;
const METER_RELEASE_TAU_S = 0.55;

function ballistic(current: number, target: number, dt: number): number {
  const tau = target > current ? METER_ATTACK_TAU_S : METER_RELEASE_TAU_S;
  // Time-constant form, not a fixed coefficient: the event period is nominal
  // and a dropped or batched block would otherwise change the smoothing.
  const alpha = 1 - Math.exp(-Math.max(dt, 1e-3) / tau);
  return current + (target - current) * alpha;
}

function useMeterBallistics(
  volume: number,
  periodicity: number,
  t: number,
): {volume: number; periodicity: number} {
  const [smoothed, setSmoothed] = useState({volume, periodicity});
  const lastT = useRef<number | null>(null);
  useEffect(() => {
    const previous = lastT.current;
    const known = Number.isFinite(t);
    // The AUDIO clock, not the wall clock: the level event carries the source
    // position, and pacing off wall time would smooth differently whenever the
    // decoder ran behind.
    const dt = previous !== null && known
      ? Math.min(Math.max(t - previous, 0), 1)
      : 0.12;
    if (known) lastT.current = t;
    setSmoothed((current) => ({
      volume: ballistic(current.volume, volume, dt),
      periodicity: ballistic(current.periodicity, periodicity, dt),
    }));
  }, [t, volume, periodicity]);
  return smoothed;
}

/* A STATUS WORD THAT CHANGES EIGHT TIMES A SECOND IS NOT A READING.
   MEASURED over the same 191 events, `status` changed on 34% of frames --
   `idle` and `good` alternating through every pause between words, in a rail
   whose whole job is to answer "is the microphone working" at a glance. A new
   value has to hold before it is shown. `clipping` is exempt: it is the one
   value that is a warning, and a warning must not be delayed. */
const STATUS_DWELL_EVENTS = 4;

function useSettledStatus(status: string): string {
  const [shown, setShown] = useState(status);
  const pending = useRef<{value: string; count: number}>({value: status, count: 0});
  useEffect(() => {
    if (status === "clipping") {
      pending.current = {value: status, count: 0};
      setShown(status);
      return;
    }
    setShown((current) => {
      if (status === current) {
        pending.current = {value: current, count: 0};
        return current;
      }
      const seen = pending.current.value === status
        ? pending.current.count + 1
        : 1;
      pending.current = {value: status, count: seen};
      return seen >= STATUS_DWELL_EVENTS ? status : current;
    });
  }, [status]);
  return shown;
}

/* THE SIDE-GRID INSTRUMENT, AND NOW THE ONLY ONE. */
function VoiceCompass({
  level,
  color,
  speakerColors,
  sound,
}: {
  level: LevelEvent;
  color: string;
  speakerColors?: SpeakerColorMap;
  sound?: {label?: string; category?: string} | null;
}) {
  const rawVolume = clamp((number(level.rms_db, -72) + 60) / 45, 0, 1);
  const rawPeriodicity = clamp(number(level.pitch_confidence, 0), 0, 1);
  // The two channels that still reach the screen go through meter ballistics;
  // see `useMeterBallistics`.
  const {volume, periodicity} = useMeterBallistics(
    rawVolume, rawPeriodicity, number(level.t, Number.NaN),
  );
  const pitch = clamp((number(level.pitch_hz, 165) - 80) / 170, 0, 1);
  const brightness = clamp(
    (number(level.spectral_centroid_hz, 1600) - 500) / 3000,
    0,
    1,
  );
  const force = clamp(number(level.delivery_force, volume), 0, 1);
  const attack = clamp(number(level.delivery_attack), 0, 1);
  const contour = clamp(number(level.delivery_contour), -1, 1);
  const flow = clamp(number(level.delivery_flow, periodicity), 0, 1);
  const texture = clamp(number(level.delivery_texture, 1 - periodicity), 0, 1);
  const profile = String(level.delivery_profile ?? "steady");
  const direction = number(
    level.direction_deg ?? level.azimuth_deg,
    Number.NaN,
  );
  /* THE DIAL FALLS BACK TO 0°; THE EVENT DOES NOT (2026-08-13, at the user's
     direction, and knowingly against this project's own standing rule). */
  const directionMeasured = Number.isFinite(direction);
  const directionKnown = true;
  /* A LAPSED READING HOLDS ITS LAST BEARING; IT DOES NOT SNAP TO THE FRONT
     (2026-09-07, at the user's request). The array publishes a bearing only
     while it reports speech, so every pause used to fling the needle back to
     0deg -- invisible when the talker sits at the front of the case, and a
     constant snap-home anywhere else. Holding is also the more honest of the
     two: the last bearing was measured, 0deg never was. It stays dimmed by
     `data-measured="false"`, which is the claim being withheld, and 0deg
     remains the fallback only until the FIRST real reading arrives. */
  const [heldBearing, setHeldBearing] = useState<number | null>(null);
  useEffect(() => {
    if (Number.isFinite(direction)) setHeldBearing(compassBearing(direction));
  }, [direction]);
  const shownBearing = directionMeasured
    ? compassBearing(direction)
    : heldBearing ?? 0;
  /* Absent means the array measured nothing, so no needle is drawn -- the
     compass may hold a stale PRIMARY bearing, but it must not invent a second
     talker who was never heard. */
  const beamBearings = useMemo(() => {
    const raw = level.beam_bearings_deg;
    return Array.isArray(raw)
      ? raw.map(Number).filter((b) => Number.isFinite(b)).slice(0, COMPASS_BEAMS)
      : [];
  }, [level.beam_bearings_deg]);
  const style: CSSVars = {
    "--orb-color": color,
    /* THE PULSE HAS A RANGE WORTH SEEING. */
    "--orb-scale": (0.78 + volume * 0.46).toFixed(3),
    "--orb-halo": `${(volume * periodicity * 38).toFixed(1)}px`,
    "--pitch-y": `${(78 - pitch * 56).toFixed(1)}%`,
    "--texture-x": (0.68 + brightness * 0.78).toFixed(3),
    "--texture-size": (0.74 + volume * 0.48).toFixed(3),
    "--texture-opacity": (0.10 + periodicity * 0.48).toFixed(3),
    "--texture-angle": `${(-14 + brightness * 28 + contour * 18).toFixed(1)}deg`,
    "--delivery-tilt": `${(contour * 8).toFixed(1)}deg`,
    "--delivery-stretch-x": (0.96 + flow * 0.12).toFixed(3),
    "--delivery-stretch-y": (0.94 + force * 0.15).toFixed(3),
    "--delivery-energy": (0.18 + force * 0.62 + attack * 0.20).toFixed(3),
    "--delivery-texture": texture.toFixed(3),
    /* The unmeasured fallback is 0deg ON SCREEN, not through the offset: it
       means "assume the talker is at the front of the case", which is where the
       front tick is drawn. Passing it through would aim the default needle at
       the back. */
    "--direction-angle": `${shownBearing}deg`,
  };
  // Standing speaker positions, the SpeechCompass minimap idea: the live dot
  // is where sound is arriving NOW, these are where each speaker sits. Colours
  // are the caption colours, so the ring answers "who is where" with the same
  // vocabulary the text uses. Absent until the array has placed somebody.
  //
  // `speaker_bearings` NAMES its speaker and is preferred wherever it arrives:
  // the older `speaker_slots_deg` colours by array INDEX, which is the order
  // bearings were first seen and not the order speakers were identified, so
  // its marks can wear another speaker's colour. It also only ever grew while
  // attribution was undecided -- measured live, one mark for two speakers
  // across a 37 deg sweep -- which is why the ring never remembered anybody.
  const marksRaw = level.speaker_bearings;
  const marks = Array.isArray(marksRaw) ? marksRaw : [];
  const slotsRaw = level.speaker_slots_deg;
  const slots = marks.length
    ? marks.map((mark) => number(mark.deg, 0))
    : Array.isArray(slotsRaw) ? slotsRaw : [];
  const slotSpeaker = (index: number): string =>
    marks.length ? String(marks[index]?.speaker ?? "") : `S${index + 1}`;

  const label = `${profile} delivery, ${number(level.rms_db, -72).toFixed(1)} dB, ${
    number(level.pitch_hz) > 0 ? `${Math.round(number(level.pitch_hz))} Hz` : "unvoiced"
  }${directionMeasured
    ? `, ${Math.round(compassBearing(direction))} degrees`
    : ", direction not measured"}`;

  return (
    <div
      className="voice-compass"
      data-direction={directionKnown ? "known" : "unknown"}
      data-measured={directionMeasured ? "true" : "false"}
      data-delivery={profile}
      role="img"
      aria-label={`Voice compass: ${label}`}
      style={style}
    >
      {/* THE DIAL SHOWS WHERE PEOPLE ARE, AND NOTHING ELSE (2026-08-13).
          Two concentric radar rings and a full crosshair were drawing a radar
          set: none of it was data, and at 152px they were the busiest thing in
          a rail that had just been emptied for this dial's benefit. What a
          bearing needs is a frame of reference, which is four ticks -- front,
          both sides, behind -- and `.compass-front` marks which one is the
          front of the case, the one thing a viewer has to know to read any of
          the others. */}
      {/* A DIAL WITHOUT A SCALE CANNOT BE READ. Four ticks stripped the radar
          decoration and took the legibility with it -- on a bare circle there
          is nothing to place 231 degrees AGAINST. Twelve, every 30 degrees,
          with the four cardinals long: that is a scale, and a scale is
          information rather than ornament.
          The FRONT tick stays the brightest, because which way the case points
          is what every other bearing is relative to -- and is the one value
          this project has not measured against the real case. */}
      {Array.from({length: 12}, (_, i) => i * 30).map((deg) => (
        <span
          key={`tick-${deg}`}
          className="compass-tick"
          data-cardinal={deg % 90 === 0 ? "true" : "false"}
          data-front={deg === 0 ? "true" : "false"}
          style={{"--tick-angle": `${deg}deg`} as CSSVars}
        />
      ))}
      {slots.map((bearing, index) => {
        const angle = compassBearing(Number(bearing));
        /* AN ARC, NOT A DOT, AND ITS WIDTH IS THE UNCERTAINTY. */
        const spread = marks.length ? number(marks[index]?.spread, 26) : 26;
        const span = clamp(spread * 2, 8, 150);
        /* A speaker never leaves the ring once placed, so some marks are the
           last place somebody was seen. Drawn dimmer: still there, no longer
           a claim about the present. */
        const stale = Boolean(marks.length && marks[index]?.stale);
        return (
          <span
            key={`slot-${index}`}
            className="compass-slot"
            data-stale={stale ? "true" : "false"}
            style={{
              "--slot-angle": `${angle.toFixed(1)}deg`,
              "--slot-span": `${span.toFixed(1)}deg`,
              "--slot-color": speakerColors
                ? speakerColor(slotSpeaker(index), speakerColors)
                : "var(--accent)",
            } as CSSVars}
          />
        );
      })}
      {/* AN ARROW PER SPEAKER, NOT ONE NEEDLE THAT CHASES THE TALKER
          (2026-09-05, at the user's request). The single live needle tracks the
          INSTANTANEOUS bearing, so it flips to whoever speaks and reads as
          constant jitter. Each speaker's own standing bearing is stable, so a
          fixed arrow per speaker -- in that speaker's colour -- lets the viewer
          read the room at a glance instead of following a jumping line. Capped
          at COMPASS_ARROWS so a large cast cannot turn the dial into a
          star-burst; the arcs still mark everyone. Only appears with the mic
          array (no bearings without it), exactly like the needle. */}
      {slots.slice(0, COMPASS_ARROWS).map((bearing, index) => {
        const angle = compassBearing(Number(bearing));
        const stale = Boolean(marks.length && marks[index]?.stale);
        return (
          <span
            key={`arrow-${index}`}
            className="compass-arrow"
            data-stale={stale ? "true" : "false"}
            style={{
              "--arrow-angle": `${angle.toFixed(1)}deg`,
              "--arrow-color": speakerColors
                ? speakerColor(slotSpeaker(index), speakerColors)
                : "var(--accent)",
            } as CSSVars}
          />
        );
      })}
      {/* THE LIVE NEEDLE. With per-speaker arrows now carrying "who is where",
          this is only the CURRENT arrival, kept subtle so it no longer
          dominates. A 9px dot on the rim asked the viewer to find it and then
          work out which way it lay; a line from the centre states the bearing
          the way a compass has always stated it. The dot stays at its tip --
          it is what carries the ACTIVE speaker's colour. */}
      {/* WHAT IS HAPPENING IN THE ROOM, IN THE MIDDLE OF THE INSTRUMENT THAT
          DESCRIBES THE ROOM (2026-08-13, at the user's request). A non-speech
          sound is not a word -- it has no speaker, no colour and no place in a
          line of text -- but it IS an event around the listener, which is
          exactly what this dial is for.
          The stage keeps its own `[Toy music]` caption: CWI 2.4.4 requires the
          bracketed white label and this does not replace it. This is the same
          fact placed where the room is drawn. */}
      {sound?.label || sound?.category ? (
        <span className="compass-sound" data-category={sound.category ?? "environmental"}>
          <i>[{sound.label ?? sound.category}]</i>
        </span>
      ) : null}
      {/* A NEEDLE PER LIVE TALKER BEAM (2026-09-07, at the user's request).
          The array steers two beams independently, so two people speaking at
          once are two bearings -- and `direction_deg` carries only the
          dominant one, which silently drops the second talker. Drawn thinner
          and dimmer than the live needle because it IS looser evidence: the
          beam gate measures circular concentration 0.83 against the VAD
          gate's 1.00. Two beams on one bearing overlap into a single line,
          which is the honest picture of one talker held by both; no
          separation threshold is invented to force them apart. */}
      {beamBearings.map((bearing, index) => (
        <span
          key={`beam-${index}`}
          className="compass-beam"
          style={{
            "--beam-angle": `${compassBearing(bearing).toFixed(1)}deg`,
          } as CSSVars}
        ><b /><i /></span>
      ))}
      <span className="compass-direction"><b /><i /></span>
      {/* `.compass-texture` and `.compass-pitch` are gone with them. Both were
          voice channels -- brightness and F0 -- and both are on screen already
          as the caption's own texture and weight under CWI 2.3. The dial was
          restating the captions inside a diagram about space. */}
    </div>
  );
}

function CaptionFeed({
  paragraphs,
  timingWords,
  speakerColors,
  intensity,
  runtime,
  clockEpoch,
  scheduleWord,
  playheadMs,
  transcript,
  reducedMotion,
  hangulWave,
  enhancedMotion,
}: {
  paragraphs: CaptionParagraph[];
  /* THE MOTION CLOCK READS THE WORD LIST, NOT THE ROWS (2026-08-06). */
  timingWords: CaptionParagraph["words"];
  speakerColors: SpeakerColorMap;
  intensity: number;
  runtime: RuntimeConfig;
  clockEpoch: number | null;
  scheduleWord: (
    id: string,
    word: CaptionWord,
    durationMs: number,
  ) => {turnAtMs: number; epoch: number} | null;
  playheadMs: number;
  transcript: boolean;
  reducedMotion: boolean;
  /** Settings toggle, threaded to `MotionWord`. Wide script only. */
  hangulWave: boolean;
  /** Settings toggle, threaded to `MotionWord`. */
  enhancedMotion: boolean;
}) {
  /** Hold gap per word id, frozen at first sight; survives child remounts. */
  const holdMemoRef = useRef(new Map<string, number>());
  /* Everything else a word freezes at first sight, for the same reason -- see
     `WordMemo`. State, not a ref: `MotionWord` reads it while rendering. */
  const [wordMemo] = useState<WordMemo>(() => ({
    clock: new Map(),
    hold: new Map(),
    voice: new Map(),
  }));
  const rowNodes = useRef(new Map<string, HTMLElement>());
  const previousPositions = useRef<CaptionStackPosition[]>([]);
  const stackInitialized = useRef(false);
  const stackAnimations = useRef(new Map<string, Animation>());
  const seenRows = useRef(new Set<string>());
  /* Each caption row's last measured width, keyed by its position in the
     feed. */
  const rowWidths = useRef<number[]>([]);
  const rowGrowAnimations = useRef(new Map<number, Animation>());

  useLayoutEffect(() => {
    if (transcript || reducedMotion || paragraphs.length === 0) {
      for (const animation of stackAnimations.current.values()) {
        animation.cancel();
      }
      stackAnimations.current.clear();
      previousPositions.current = [];
      stackInitialized.current = false;
      return;
    }

    const currentPositions = paragraphs.flatMap((paragraph) => {
      const node = rowNodes.current.get(paragraph.id);
      if (!node) return [];
      const transform = getComputedStyle(node).transform;
      const animatedY = transform === "none"
        ? 0
        : new DOMMatrixReadOnly(transform).m42;
      return [{
        id: paragraph.id,
        top: node.getBoundingClientRect().top - animatedY,
      }];
    });
    if (!stackInitialized.current) {
      const motions = planCaptionStackMotion(
        [],
        currentPositions,
        seenRows.current,
      );
      previousPositions.current = currentPositions;
      for (const {id} of currentPositions) seenRows.current.add(id);
      stackInitialized.current = true;
      for (const motion of motions) {
        const node = rowNodes.current.get(motion.id);
        if (!node) continue;
        const targetOpacity = getComputedStyle(node).opacity;
        const animation = node.animate(
          [
            {opacity: 0, transform: "translateY(0.58em) scale(.985)"},
            {opacity: targetOpacity, transform: "translateY(0) scale(1)"},
          ],
          {
            duration: STACK_ENTER_DURATION_MS,
            easing: STACK_EASING,
          },
        );
        stackAnimations.current.set(motion.id, animation);
        animation.addEventListener("finish", () => {
          if (stackAnimations.current.get(motion.id) === animation) {
            stackAnimations.current.delete(motion.id);
          }
        }, {once: true});
      }
      return;
    }

    const motions = planCaptionStackMotion(
      previousPositions.current,
      currentPositions,
      seenRows.current,
    );
    for (const motion of motions) {
      const node = rowNodes.current.get(motion.id);
      if (!node) continue;
      stackAnimations.current.get(motion.id)?.cancel();
      const targetOpacity = getComputedStyle(node).opacity;
      const animation = node.animate(
        motion.kind === "shift"
          ? [
            {transform: `translateY(${motion.deltaY}px)`},
            {transform: "translateY(0)"},
          ]
          : [
            {opacity: 0, transform: "translateY(0.58em) scale(.985)"},
            {opacity: targetOpacity, transform: "translateY(0) scale(1)"},
          ],
        {
          duration: motion.kind === "shift"
            ? STACK_SHIFT_DURATION_MS
            : STACK_ENTER_DURATION_MS,
          easing: STACK_EASING,
        },
      );
      stackAnimations.current.set(motion.id, animation);
      animation.addEventListener("finish", () => {
        if (stackAnimations.current.get(motion.id) === animation) {
          stackAnimations.current.delete(motion.id);
        }
      }, {once: true});
    }
    for (const {id} of currentPositions) seenRows.current.add(id);
    previousPositions.current = currentPositions;
  }, [paragraphs, reducedMotion, transcript]);

  /* THE BOX OPENS SIDEWAYS OVER A NEWLY APPENDED WORD. */
  useLayoutEffect(() => {
    if (transcript || reducedMotion) {
      for (const animation of rowGrowAnimations.current.values()) {
        animation.cancel();
      }
      rowGrowAnimations.current.clear();
      rowWidths.current = [];
      return;
    }
    // Rows in paragraph order, from the nodes the stack effect already
    // tracks -- no second ref, and the order is the stack's own.
    const rows = paragraphs.flatMap((paragraph) => {
      const node = rowNodes.current.get(paragraph.id);
      return node
        ? Array.from(node.querySelectorAll<HTMLElement>(".caption-words"))
        : [];
    });
    const widths = rows.map((row) => row.getBoundingClientRect().width);
    const previous = rowWidths.current;
    rowWidths.current = widths;
    // The first pass has nothing to compare against, and a row that appears
    // with the paragraph it belongs to is the stack's ENTER transition --
    // opening it sideways as well would double the effect.
    if (previous.length === 0) return;
    rows.forEach((row, index) => {
      const before = previous[index];
      const after = widths[index];
      if (before === undefined || after - before < 1) return;
      rowGrowAnimations.current.get(index)?.cancel();
      const animation = row.animate(
        [
          {clipPath: `inset(0 ${(after - before).toFixed(1)}px 0 0)`},
          {clipPath: "inset(0 0 0 0)"},
        ],
        {duration: ROW_GROW_DURATION_MS, easing: "cubic-bezier(.22,.61,.36,1)"},
      );
      rowGrowAnimations.current.set(index, animation);
      animation.addEventListener("finish", () => {
        if (rowGrowAnimations.current.get(index) === animation) {
          rowGrowAnimations.current.delete(index);
        }
      }, {once: true});
    });
  }, [paragraphs, reducedMotion, transcript]);

  useEffect(() => () => {
    for (const animation of rowGrowAnimations.current.values()) {
      animation.cancel();
    }
    rowGrowAnimations.current.clear();
  }, []);

  useEffect(() => () => {
    for (const animation of stackAnimations.current.values()) {
      animation.cancel();
    }
    stackAnimations.current.clear();
  }, []);

  /* WHERE SPEECH ACTUALLY IS, which is no longer the bottom of the stack. */
  /* How long the speaker paused before each word. */
  /* KEYED BY WORD ID AND PERSISTED, BECAUSE A REMOUNT MUST NOT RE-DERIVE IT. */
  const holdGaps = new Map<string, number>();
  /* Ids whose hold gap is final. */
  const holdSettledIds = new Set<string>();
  /* ...AND HOW LONG THE MOTION LASTS, WHICH IS ONE WORD AT THE CURRENT
     SPEECH RATE. */
  const paceGaps = new Map<string, number>();
  {
    let previousOnset: number | null = null;
    let previousUtterance: number | null = null;
    const flat = timingWords;
    for (let index = 0; index < flat.length; index += 1) {
      const {id, word} = flat[index];
      /* A SENTENCE BREAK IS NOT A HELD WORD. */
      const utterance = number(word.utterance, Number.NaN);
      const startsUtterance = Number.isFinite(utterance) &&
        previousUtterance !== null && utterance !== previousUtterance;
      if (Number.isFinite(utterance)) previousUtterance = utterance;
      const onset = number(word.t ?? word.start, Number.NaN);
      /* ONSET TO ONSET, NOT END TO ONSET. */
      /* ...AND A LONGER PAUSE IS A SENTENCE BREAK, WHICH DOES NOT LIFT. */
      const next = flat[index + 1];
      const nextOnset = next
        ? number(next.word.t ?? next.word.start, Number.NaN)
        : Number.NaN;
      /* SILENCE ON BOTH SIDES, AND THIS IS WHAT THE ONE-WORD LOOKAHEAD BUYS. */
      if (previousOnset !== null && Number.isFinite(onset) && !startsUtterance) {
        const before = Math.max(0, onset - previousOnset);
        const after = Number.isFinite(nextOnset)
          ? Math.max(0, nextOnset - onset)
          : before;
        const gap = Math.min(before, after);
        /* ...BUT ONLY ONCE BOTH NEIGHBOURS EXIST. */
        const remembered = holdMemoRef.current.get(id);
        const settledGap = before <= runtime.holdMaxS ? gap : 0;
        if (remembered !== undefined) {
          holdGaps.set(id, remembered);
          holdSettledIds.add(id);
        } else {
          if (Number.isFinite(nextOnset)) {
            holdMemoRef.current.set(id, settledGap);
            holdSettledIds.add(id);
          }
          if (settledGap > 0) holdGaps.set(id, settledGap);
        }
      }
      if (Number.isFinite(onset) && Number.isFinite(nextOnset)) {
        paceGaps.set(id, Math.max(0, nextOnset - onset));
      }
      if (Number.isFinite(onset)) previousOnset = onset;
    }
  }

  const spokenRow = Math.max(
    0,
    paragraphs.findLastIndex((paragraph) => paragraph.words.some(
      ({word}) => acousticTimeMs(word) <= playheadMs,
    )),
  );

  // The feed container is rendered even while empty: it is the element the stage
  // measures to decide how many rows fit (`useStackCapacity`), and that answer is
  // needed BEFORE the first word arrives. The empty state is a sibling.
  return (
    <div className={transcript ? "transcript-feed" : "caption-feed"}>
      {paragraphs.map((paragraph, paragraphIndex) => {
        const color = speakerColor(paragraph.speaker, speakerColors);
        const isLatest = paragraphIndex === spokenRow;
        const firstWord = paragraph.words[0]?.word;
        return (
          <section
            className="caption-paragraph"
            data-current={isLatest ? "true" : "false"}
            data-row-id={paragraph.id}
            data-status={paragraph.status}
            key={paragraph.id}
            ref={(node) => {
              if (node) rowNodes.current.set(paragraph.id, node);
              else rowNodes.current.delete(paragraph.id);
            }}
            style={{"--speaker-color": color} as CSSVars}
          >
            <header>
              <span className="speaker-index">
                {paragraph.status === "mixed" ? "↔" : speakerNumber(paragraph.speaker)}
              </span>
              <span>{
                paragraph.status === "mixed"
                  ? "Speaker transition"
                  : paragraph.speaker
                    ? `Speaker ${speakerNumber(paragraph.speaker)}`
                    : "Attribution pending"
              }</span>
              {transcript && (
                <time>{number(firstWord?.start).toFixed(1)}s</time>
              )}
            </header>
            <div className="caption-surface">
              <div className="caption-words">
                {paragraph.words.map(({id, word}) => (
                  <MotionWord
                    hangulWave={hangulWave}
                    enhancedMotion={enhancedMotion}
                    id={id}
                    word={word}
                    holdGapS={holdGaps.get(id) ?? 0}
                    holdSettled={holdSettledIds.has(id)}
                    paceGapS={paceGaps.get(id) ?? 0}
                    clockEpoch={clockEpoch}
                    scheduleWord={scheduleWord}
                    memo={wordMemo}
                    color={speakerColor(word.speaker, speakerColors)}
                    intensity={intensity}
                    runtime={runtime}
                    key={id}
                  />
                ))}
              </div>
            </div>
          </section>
        );
      })}
    </div>
  );
}

/** Resolve the stage's two coupled unknowns: words per row, and rows
   retained. */
/** Find the baseline inside a caption word's box, so the cue can grow FROM
   it. */
function useGlyphBaseline(language: string | null): string | null {
  const [offset, setOffset] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    const measure = () => {
      if (cancelled) return;
      // The probe has to carry the REAL caption typography, so it borrows the
      // live class names rather than restating font-family/line-height here.
      const host = document.createElement("div");
      host.className = "caption-feed";
      host.setAttribute("aria-hidden", "true");
      host.style.cssText =
        "position:absolute;left:-9999px;top:0;visibility:hidden;" +
        "contain:strict;width:400px;height:200px;";
      // `.caption-words` is not optional scaffolding: it carries the caption
      // line-height, and the baseline's distance from the box bottom is
      // half-leading PLUS descent. Measuring without it returned an identical
      // 0.2500em for Roboto Flex and Noto Sans KR -- the giveaway that the
      // probe was reading `line-height: normal` rather than the real 1.38, and
      // leaving the pivot ~0.1em below the true baseline.
      const row = document.createElement("div");
      row.className = "caption-words";
      const word = document.createElement("span");
      word.className = "caption-word";
      const glyph = document.createElement("span");
      glyph.className = "word-glyph";
      // `position:absolute` on the real glyph would detach it from this probe's
      // flow; static keeps the same line box, which is what carries the metric.
      glyph.style.position = "static";
      glyph.textContent = "Hxg";
      // A zero-width inline-block sits with its bottom margin edge ON the
      // baseline. It is the only way to reach the baseline from JavaScript.
      const strut = document.createElement("i");
      strut.style.cssText = "display:inline-block;width:0;height:0;";
      glyph.appendChild(strut);
      word.appendChild(glyph);
      row.appendChild(word);
      host.appendChild(row);
      // MOUNT IT INSIDE THE SHELL. `--font-caption` is switched by
      // `.studio-shell[data-language="ko"]`, so a probe parented to
      // `document.body` silently measures Roboto Flex even in a Korean
      // session -- which is exactly how this returned an identical number for
      // both faces twice before.
      const shell = document.querySelector(".studio-shell") ?? document.body;
      shell.appendChild(host);
      const fontSize = parseFloat(getComputedStyle(glyph).fontSize);
      const measured = baselineOffsetEm(
        glyph.getBoundingClientRect(),
        strut.getBoundingClientRect(),
        fontSize,
      );
      host.remove();
      if (measured === null) return;
      const next = formatBaselineEm(measured);
      setOffset((current) => (current === next ? current : next));
    };
    // The caption faces are local @font-face downloads. Measuring before they
    // load returns the fallback face's metrics, which is a different number.
    document.fonts.ready.then(measure).catch(measure);
    return () => {
      cancelled = true;
    };
  }, [language]);
  return offset;
}

/** How wide one WIDE-script character is, in em, on the live caption face. */
function useWideCharEm(language: string | null): number | null {
  const [em, setEm] = useState<number | null>(null);
  useEffect(() => {
    let cancelled = false;
    const measure = () => {
      if (cancelled) return;
      const host = document.createElement("div");
      host.className = "caption-feed";
      host.setAttribute("aria-hidden", "true");
      // No `contain:strict` and no fixed width: this probe MEASURES width, so
      // it must be free to be as wide as the text makes it.
      host.style.cssText =
        "position:absolute;left:-9999px;top:0;visibility:hidden;" +
        "width:max-content;white-space:nowrap;";
      const row = document.createElement("div");
      row.className = "caption-words";
      const word = document.createElement("span");
      word.className = "caption-word";
      // A spread of real syllables rather than one repeated block, so a face
      // with per-glyph variation cannot be read off an unlucky sample. Hangul
      // measures uniform, which is itself worth confirming at runtime.
      const sample = "가나다라마바사아자차카타파하각힣";
      word.textContent = sample;
      row.appendChild(word);
      host.appendChild(row);
      // MOUNT IT INSIDE THE SHELL -- `--font-caption` is switched by
      // `.studio-shell[data-language="ko"]`, so a probe parented to
      // `document.body` measures Roboto Flex even in a Korean session. That is
      // the exact mistake `useGlyphBaseline` made twice.
      const shell = document.querySelector(".studio-shell") ?? document.body;
      shell.appendChild(host);
      const fontSize = parseFloat(getComputedStyle(word).fontSize);
      const width = word.getBoundingClientRect().width;
      host.remove();
      const count = Array.from(sample).length;
      if (!Number.isFinite(fontSize) || fontSize <= 0 || width <= 0) return;
      const measured = width / fontSize / count;
      // Sanity band. A wide character cannot be narrower than a Latin one, and
      // cannot exceed one em by much. Outside this the probe measured the wrong
      // face or the fallback, and a wrong number here CLIPS captions -- so
      // report nothing and let the chunker fall back to `charEm`, which is
      // today's behaviour rather than a new failure.
      if (measured < 0.5 || measured > 1.4) return;
      setEm((current) => (
        current !== null && Math.abs(current - measured) < 1e-4
          ? current
          : measured
      ));
    };
    document.fonts.ready.then(measure).catch(measure);
    return () => {
      cancelled = true;
    };
  }, [language]);
  return em;
}

function useStageLayout(
  stage: React.RefObject<HTMLDivElement | null>,
  runtime: RuntimeConfig,
  hasRows: boolean,
  captionScale: number,
  view: ViewMode,
  // A ResizeObserver on the stage cannot see a theme swap, and the theme changes
  // a row's height, so it has to be an explicit dependency.
  theme: string,
): {wordsPerRow: number; rows: number; rowBudgetEm: number} {
  const [layout, setLayout] = useState(() => ({
    wordsPerRow: runtime.stageWordsPerBlock,
    rows: runtime.stageParagraphHistory,
    rowBudgetEm: rowBudgetEm(runtime.stageWordsPerBlock, 1.45, 6.60),
  }));
  useEffect(() => {
    const node = stage.current;
    if (!node || view !== "stage") return;
    const measure = () => {
      const feed = node.querySelector<HTMLElement>(".caption-feed");
      if (!feed) return;
      const styles = getComputedStyle(feed);
      const fontSize = parseFloat(styles.fontSize);
      const stageHeight = node.clientHeight;
      if (!fontSize || !stageHeight) return;
      // Chrome reports a percentage `max-height` as the literal "92%" instead of
      // resolving it, so recover the fraction rather than parsing a px value.
      const maxHeightFraction = styles.maxHeight.endsWith("%")
        ? parseFloat(styles.maxHeight) / 100
        : (parseFloat(styles.maxHeight) || stageHeight) / stageHeight;
      // A rendered row is exact; before the first word its line box is the only
      // estimate there is.
      const row = feed.querySelector<HTMLElement>(".caption-paragraph");
      const rowHeightEm = row
        ? row.getBoundingClientRect().height / fontSize
        : 1.38;
      const next = planStageLayout({
        feedWidthPx: feed.getBoundingClientRect().width,
        stageHeightPx: stageHeight,
        gutterEm: (parseFloat(styles.paddingLeft) +
          parseFloat(styles.paddingRight)) / fontSize,
        verticalGutterEm: (parseFloat(styles.paddingTop) +
          parseFloat(styles.paddingBottom)) / fontSize,
        wordEmLinear: parseFloat(styles.getPropertyValue("--word-em-linear")),
        wordEmSpread: parseFloat(styles.getPropertyValue("--word-em-spread")),
        rowHeightEm,
        maxHeightFraction,
        heightCapPx:
          parseFloat(styles.getPropertyValue("--caption-height-cap")) *
          captionScale,
        minWords: runtime.stageWordsMin,
        maxWords: runtime.stageWordsPerBlock,
        minRows: runtime.stageMinRows,
      });
      const budget = rowBudgetEm(
        next.wordsPerRow,
        parseFloat(styles.getPropertyValue("--word-em-linear")) || 1.45,
        parseFloat(styles.getPropertyValue("--word-em-spread")) || 6.60,
      );
      setLayout((current) => (
        current.wordsPerRow === next.wordsPerRow &&
        current.rows === next.rows &&
        current.rowBudgetEm === budget
          ? current
          : {wordsPerRow: next.wordsPerRow, rows: next.rows, rowBudgetEm: budget}
      ));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [stage, runtime, hasRows, captionScale, view, theme]);
  return layout;
}

function SignalWaveform({values}: {values: number[]}) {
  return (
    <div className="signal-waveform" aria-label="Recent input level">
      {values.map((db, index) => {
        const height = clamp((db + 72) / 57, 0.08, 1);
        return (
          <i
            key={index}
            style={{"--bar-height": height.toFixed(3)} as CSSVars}
          />
        );
      })}
    </div>
  );
}

function LanguageGate({
  session,
  error,
  onSelect,
}: {
  session: LanguageSession;
  error: string | null;
  onSelect: (language: string) => void;
}) {
  const selected = session.languages.find(
    (language) => language.id === session.language,
  );
  const selecting = session.state === "selecting";
  const unavailable = session.state === "unavailable";

  return (
    <section
      className="language-gate"
      role="dialog"
      aria-modal="true"
      aria-label="Caption language setup"
    >
      <div className="language-gate-card">
        {/* EVERY LINE OF COPY IS GONE (2026-09-06, at the user's direction):
            the brand header, the "What language will be spoken?" heading, the
            eyebrow, the descriptions and the "On-device processing" footer.
            While choosing, the three options ARE the screen -- a centred
            stack of pill buttons, each just the English label. A heading
            renders only when there is a state to report: preparing, or the
            engine being unreachable. */}
        {!selecting && (
          <div className="language-gate-copy">
            <div className="language-gate-state">
              {/* A bare ring BESIDE the heading (2026-09-06, at the user's
                  direction). The boxed "Loading locally / <label> speech
                  model" card restated the "Preparing <label>" heading; the
                  ring alone says the rest. */}
              {!unavailable && (
                <span
                  className="language-loader"
                  role="status"
                  aria-label="Loading"
                />
              )}
              <h1>
                {unavailable
                  ? "The local caption engine is unavailable"
                  : selected
                    ? `Preparing ${selected.label}`
                    : "Preparing language setup"}
              </h1>
            </div>
            {unavailable && (
              <p>
                Open this studio through “python -m autocwi live” so it can
                reach the local session controller.
              </p>
            )}
          </div>
        )}

        {selecting && (
          <div className="language-options" role="list" aria-label="Caption language">
            {session.languages.map((language: LiveLanguageOption) => (
              <button
                className="language-option"
                key={language.id}
                onClick={() => onSelect(language.id)}
                type="button"
              >
                {language.label}
              </button>
            ))}
          </div>
        )}

        {error && <p className="language-error" role="alert">{error}</p>}
      </div>
    </section>
  );
}

function SettingsPanel({
  settings,
  setSettings,
  close,
}: {
  settings: SettingsState;
  setSettings: (settings: SettingsState) => void;
  close: () => void;
}) {
  return (
    <aside className="settings-panel" aria-label="Presentation settings">
      <header>
        <h2>Presentation</h2>
        <button className="icon-button" onClick={close} aria-label="Close settings">
          <X size={17} />
        </button>
      </header>

      <label className="range-control">
        <span><Captions size={15} /> Caption scale</span>
        <output>{Math.round(settings.captionScale * 100)}%</output>
        {/* 100% is now the LARGEST size that still fits a full row across the
            stage, not an arbitrary middle. The width cap is a hard bound -- rows
            do not wrap, so anything above it is clipped, not wrapped -- which is
            why the slider only reduces. */}
        <input
          type="range"
          min="0.6"
          max="1"
          step="0.05"
          value={settings.captionScale}
          /* WebKit has no range-progress pseudo-element, so the travelled
             side of the track is a gradient stop the input carries itself. */
          style={{
            "--fill": `${((settings.captionScale - 0.6) / 0.4) * 100}%`,
          } as CSSVars}
          onChange={(event) => setSettings({
            ...settings,
            captionScale: Number(event.target.value),
          })}
        />
      </label>

      <label className="range-control">
        <span><Sparkles size={15} /> Expression</span>
        <output>{Math.round(settings.motionIntensity * 100)}%</output>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={settings.motionIntensity}
          style={{
            "--fill": `${settings.motionIntensity * 100}%`,
          } as CSSVars}
          onChange={(event) => setSettings({
            ...settings,
            motionIntensity: Number(event.target.value),
          })}
        />
      </label>

      <button
        className="toggle-row"
        aria-pressed={settings.reducedMotion}
        onClick={() => setSettings({
          ...settings,
          reducedMotion: !settings.reducedMotion,
        })}
      >
        <span><Eye size={15} /> Reduce motion</span>
        <i data-on={settings.reducedMotion ? "true" : "false"} />
      </button>

      <button
        className="toggle-row"
        aria-pressed={settings.highContrast}
        onClick={() => setSettings({
          ...settings,
          highContrast: !settings.highContrast,
        })}
      >
        <span><CircleGauge size={15} /> High contrast</span>
        <i data-on={settings.highContrast ? "true" : "false"} />
      </button>

      <button
        className="toggle-row"
        aria-pressed={settings.lightStage}
        onClick={() => setSettings({
          ...settings,
          lightStage: !settings.lightStage,
        })}
      >
        <span><Sun size={15} /> Light stage</span>
        <i data-on={settings.lightStage ? "true" : "false"} />
      </button>

      <button
        className="toggle-row"
        aria-pressed={settings.hangulWave}
        onClick={() => setSettings({
          ...settings,
          hangulWave: !settings.hangulWave,
        })}
      >
        <span><Type size={15} /> Hangul-shaped motion</span>
        <i data-on={settings.hangulWave ? "true" : "false"} />
      </button>

      <button
        className="toggle-row"
        aria-pressed={settings.rollingCaptions}
        onClick={() => setSettings({
          ...settings,
          rollingCaptions: !settings.rollingCaptions,
        })}
      >
        <span><AlignEndHorizontal size={15} /> Rolling captions</span>
        <i data-on={settings.rollingCaptions ? "true" : "false"} />
      </button>

      <button
        className="toggle-row"
        aria-pressed={settings.captionRules}
        onClick={() => setSettings({
          ...settings,
          captionRules: !settings.captionRules,
        })}
      >
        <span><Minus size={15} /> Caption rules, not a box</span>
        <i data-on={settings.captionRules ? "true" : "false"} />
      </button>

      <button
        className="toggle-row"
        aria-pressed={settings.enhancedMotion}
        onClick={() => setSettings({
          ...settings,
          enhancedMotion: !settings.enhancedMotion,
        })}
      >
        <span><Sparkles size={15} /> Enhanced motion</span>
        <i data-on={settings.enhancedMotion ? "true" : "false"} />
      </button>
    </aside>
  );
}

export function LiveStudio() {
  const [settings, setSettings] = useState<SettingsState>({
    captionScale: 1,
    motionIntensity: 1,
    reducedMotion: false,
    highContrast: false,
    /* OFF BY DEFAULT SINCE 2026-08-07, ON THE MEASUREMENT (user: "just dark
       mode"). */
    lightStage: false,
    /* Hangul-structural character motion. */
    hangulWave: true,
    /* THE ENHANCED MOTION SYSTEM (2026-08-11). */
    enhancedMotion: true,
    /* ROLLING CAPTIONS: newest at the bottom, history rising above it and
       receding as it goes. */
    rollingCaptions: true,
    /* THE CAPTION PLATE, AS TWO RULES INSTEAD OF A FILLED BOX. */
    captionRules: true,
  });
  const [view, setView] = useState<ViewMode>("stage");
  const [railOpen, setRailOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const {
    model,
    level,
    waveform,
    connection,
    runtime,
    session,
    languageError,
    selectLanguage,
    scheduleWord,
    clockEpoch,
    playheadMs,
    startedAt,
    lastEventAt,
  } = useCaptionStream({reducedMotion: settings.reducedMotion});
  const elapsed = useElapsed(startedAt);
  const activeLanguage = session.languages.find(
    (language) => language.id === session.language,
  );

  const stageRef = useRef<HTMLDivElement>(null);
  // Which words start a row. Persisted so a later edit to earlier text cannot
  // re-chunk rows the viewer has already read -- see selectStableCaptionStack.
  // Held as lazily-initialised state rather than a ref: the object identity is
  // stable for the life of the component, and reading it during render is what
  // the chunker needs, which a ref is not allowed to provide.
  const [stageMemory] = useState<StageMemory>(createStageMemory);
  const paragraphs = useMemo(
    () => buildCaptionParagraphs(
      model.words,
      model.order,
      runtime.paragraphWordLimit,
    ),
    [model.words, model.order, runtime.paragraphWordLimit],
  );
  /* The whole ordered recording, not the retained rows: deriving pace and
     hold gaps from the rows made layout an input to the motion clock. */
  const timingWords = useMemo(
    () => paragraphs.flatMap((paragraph) => paragraph.words),
    [paragraphs],
  );
  const glyphBaselineEm = useGlyphBaseline(session.language);
  const wideCharEm = useWideCharEm(session.language);
  const stageLayout = useStageLayout(
    stageRef,
    runtime,
    paragraphs.length > 0,
    settings.captionScale,
    view,
    settings.lightStage ? "light" : "dark",
  );
  const stageParagraphs = view === "stage"
    ? selectStableCaptionStack(
      paragraphs,
      stageLayout.rows,
      /* The count is now a CEILING, not the break rule -- the em budget
         below decides. */
      stageLayout.wordsPerRow * 2,
      stageMemory,
      {
        rowEm: stageLayout.rowBudgetEm,
        // Measured through the live studio; see `StageWidthBudget`.
        charEm: 0.4343,
        wordEm: 0.4289,
        /* WHAT `fill` HAS TO COVER, MEASURED (2026-08-06), because a row
           that overruns is CLIPPED and not wrapped. */
        fill: 0.82,
        /* Per-character width for East Asian WIDE scripts, measured off the
           live face (`useWideCharEm`). */
        ...(wideCharEm !== null ? {wideCharEm} : {}),
      },
    )
    : paragraphs;
  /* CWI 2.4.4/2.4.5: sound labels yield to speech. */
  const speechEndMs = useMemo(() => {
    let newest = Number.NEGATIVE_INFINITY;
    for (const paragraph of stageParagraphs) {
      for (const {word} of paragraph.words) {
        const onset = acousticTimeMs(word);
        if (!Number.isFinite(onset)) continue;
        const end = onset
          + Math.max(0, number(word.end) - number(word.start)) * 1000;
        if (end > newest) newest = end;
      }
    }
    return newest;
  }, [stageParagraphs]);
  const speechActive = playheadMs <= speechEndMs + runtime.soundLingerMs;
  const currentParagraph = paragraphs.at(-1);
  // CWI 2.1 assignment (wheel geometry, first-appearance order). The CI palette
  // is built for the black captions box and measures as low as 1.19:1 on the
  // light stage, so the light theme substitutes `palette_light` by INDEX --
  // same hues, darkened to >=4.5:1, same speaker in the same slot. Both arrive
  // from config.yaml via /runtime-config.json; neither is hardcoded here.
  const speakerColors = useSpeakerColors(
    paragraphs,
    runtime,
    settings.lightStage,
  );
  const fallbackColor = (settings.lightStage && runtime.paletteLight.length
    ? runtime.paletteLight
    : runtime.palette)[0];
  const activeColor = currentParagraph
    ? speakerColor(currentParagraph.speaker, speakerColors)
    : fallbackColor;
  // The bearing is read inside `VoiceCompass` now -- the parent no longer
  // renders a number for it, so it no longer needs to know one.
  // Held, not raw: the word alternated on 34% of frames. See
  // `useSettledStatus`. The dB reading beside it stays live.
  const settledStatus = useSettledStatus(String(level.status ?? "idle"));
  const inputGood = settledStatus === "good";
  const studioStyle: CSSVars = {
    "--caption-scale": settings.captionScale,
    "--active-color": activeColor,
    // The CSS width cap is derived from this, so the type size and the row width
    // can never disagree about how many words a row holds.
    "--stack-words": stageLayout.wordsPerRow,
    "--row-budget-em": stageLayout.rowBudgetEm.toFixed(3),
    // CWI 2.2.1: read-ahead is full white at 90% opacity -- which the PDF
    // specifies against 2.4.1's 90%-black box. The light stage has no box, and
    // white measures 1.05:1 on it, so it takes the same-semantic legible value
    // the way palette_light does. Both arrive from config.yaml.
    "--read-ahead-color": settings.lightStage
      ? runtime.readAheadColorLight
      : runtime.readAheadColor,
    "--read-ahead-opacity": runtime.readAheadOpacity,
    "--color-turn-ms": `${runtime.colorTurnMs}ms`,
    // CWI grows a word from its BASELINE and never moves it. This is where that
    // baseline actually is inside the glyph's line box, measured off the live
    // caption face; the CSS fallback covers the frame before it resolves.
    ...(glyphBaselineEm ? {"--glyph-baseline-em": glyphBaselineEm} : {}),
  };

  // The theme lives on the document element, not on the shell: `html`/`body`
  // carry `background: var(--bg)`, so anything the shell does not cover --
  // fullscreen backdrop, overscroll edge -- has to see it too.
  useEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = settings.lightStage ? "light" : "dark";
    root.style.colorScheme = settings.lightStage ? "light" : "dark";
  }, [settings.lightStage]);

  return (
    <main
      className="studio-shell"
      data-rail={railOpen ? "open" : "closed"}
      data-contrast={settings.highContrast ? "high" : "normal"}
      data-theme={settings.lightStage ? "light" : "dark"}
      data-reduced-motion={settings.reducedMotion ? "true" : "false"}
      data-hangul-wave={settings.hangulWave ? "true" : "false"}
      data-motion={settings.enhancedMotion ? "enhanced" : "legacy"}
      data-layout={settings.rollingCaptions ? "rolling" : "stack"}
      data-plate={settings.captionRules ? "rules" : "box"}
      data-language={session.language ?? "pending"}
      lang={session.language === "ko" ? "ko" : "en"}
      style={studioStyle}
    >
      <header className="topbar">
        <div className="brand">
          <div>
            <strong>Weave</strong>
            <span>Expressive caption studio</span>
          </div>
        </div>

        <nav className="view-switcher" aria-label="Studio view">
          <button
            data-active={view === "stage"}
            onClick={() => setView("stage")}
          >
            Stage
          </button>
          <button
            data-active={view === "transcript"}
            onClick={() => setView("transcript")}
          >
            Transcript
          </button>
        </nav>

        <div className="topbar-actions">
          {/* The transport bar under the stage carried this, plus three static
              "Local processing / No cloud / Stream active" badges. The badges said
              nothing that changes; the input level does, so it moves up here and
              the stage gets the 61px back. */}
          <span className="mic-state" data-good={inputGood ? "true" : "false"}>
            <Mic2 size={14} />
            {number(level.rms_db, -72).toFixed(1)} dBFS
          </span>
          {activeLanguage && (
            <span className="language-pill">
              <Globe2 size={13} />
              {activeLanguage.label}
            </span>
          )}
          <span className="session-clock">{elapsed}</span>
          <span className="connection-pill" data-state={connection}>
            <i />
            {connection === "demo" ? "Preview" : connection}
          </span>
          <button
            className="icon-button settings-button"
            onClick={() => setSettingsOpen(true)}
            aria-label="Presentation settings"
          >
            <SlidersHorizontal size={17} />
          </button>
          <button
            className="icon-button rail-toggle"
            onClick={() => setRailOpen((value) => !value)}
            aria-label={railOpen ? "Close signal rail" : "Open signal rail"}
          >
            {railOpen ? <PanelRightClose size={17} /> : <PanelRightOpen size={17} />}
          </button>
        </div>
      </header>

      <section className="workspace">
        <div
          className={`caption-stage ${view === "transcript" ? "is-transcript" : ""}`}
          ref={stageRef}
        >
          {/* THE BRACKETED LABEL LIVES IN THE COMPASS NOW (2026-08-13, at the
              user's direction: "[GASP] 이런게 캡션에서 잘 반영이 안되서 그냥
              콤퍼스에서 처리하는게 좋겠다").
              It sat at the foot of the stage as a second, differently-shaped
              caption -- no speaker, no colour, no motion, and outside the row
              stack that every other caption belongs to. A non-speech sound is
              an event in the ROOM rather than a word in the line, and the dial
              is the instrument that draws the room.
              THIS IS A DEVIATION FROM CWI 2.4.4, which puts sound effects in
              white brackets among the captions. It is a placement change, not
              a removal: the same label, the same brackets, the same white, the
              same vibration -- moved to where it is legible. `autocwi cc`,
              which renders to the design system's own layout, is untouched. */}
          <CaptionFeed
            paragraphs={stageParagraphs}
            timingWords={timingWords}
            speakerColors={speakerColors}
            intensity={settings.motionIntensity}
            runtime={runtime}
            clockEpoch={clockEpoch}
            scheduleWord={scheduleWord}
            playheadMs={playheadMs}
            transcript={view === "transcript"}
            reducedMotion={settings.reducedMotion}
            hangulWave={settings.hangulWave}
            enhancedMotion={settings.enhancedMotion}
          />
          {!stageParagraphs.length && (
            <div className="empty-stage">
              {/* One glyph, one line (2026-09-06, at the user's direction --
                  the circled plate and the "Speech will appear here" sub-line
                  said nothing the empty stage does not). The line is the boot
                  stage while models load, then "Ready for a voice". */}
              <AudioWaveform size={30} strokeWidth={1.25} aria-hidden="true" />
              <p>{lastEventAt ? "Ready for a voice" : model.bootStage}</p>
            </div>
          )}
        </div>
      </section>

      <aside className="signal-rail" aria-label="Live signal intelligence">
        <section className="rail-section compass-section">
          {/* The arrow glyph went with the heading: an icon alone in an
              otherwise empty row is decoration, and the dial already draws a
              bearing. */}
          <header className="rail-heading">
            <h2>Voice compass</h2>
          </header>
          <div className="compass-layout">
            <VoiceCompass level={level} color={activeColor}
                          speakerColors={speakerColors}
                          sound={speechActive ? null : model.sound} />
            {/* THE DIAL IS THE WHOLE READOUT NOW (2026-08-13).
                It has lost, in order: `Direction` (a label on a number that
                already ended in a degree sign), `Voice profile` (the voice
                described in prose the captions carry -- delivery is weight and
                size under CWI 2.3), and finally the number itself. `231°` sat
                beside a dial that was already pointing at it, and with no
                array it read `0°`, which is a bearing nobody measured.
                Twelve ticks put the needle inside 30° at a glance, which is
                the resolution "who is where" actually needs. */}
          </div>
        </section>

        {/* THE LEVEL SITS BELOW THE COMPASS (2026-08-13, at the user's
            direction). In the rolling layout the rail is bottom-aligned and
            the reader's eye is at the bottom of the stage, so the section
            nearest them should be the one they least need to look at -- "is
            the mic working" is a glance, "who is where" is the thing being
            read. */}
        <section className="rail-section signal-section">
          <header className="rail-heading">
            <h2>Voice activity</h2>
          </header>
          <SignalWaveform values={waveform} />
          {/* THE TWO Hz READINGS ARE GONE (2026-08-13). "215 Hz pitch" and
              "2641 Hz color" are engineering telemetry: a viewer cannot act on
              either, and both channels are ALREADY on screen in the form that
              matters -- pitch is the caption's weight and colour-brightness is
              its texture, per CWI 2.3. The rail was restating the captions in
              numbers. Level stays because it is the one reading that says
              whether the mic is working at all. */}
          {/* ONE LINE, NOT TWO CORNERS. The status word sat top-right and the
              level sat bottom-left, and they are the same fact at two
              resolutions: `IDLE` is the summary, `-17.6 dBFS` is the number
              behind it. Read together they answer "is the microphone working"
              once instead of twice. */}
          <div className="level-line" data-good={inputGood ? "true" : "false"}>
            <strong>{settledStatus}</strong>
            <span>{number(level.rms_db, -72).toFixed(1)} dBFS</span>
          </div>
        </section>

        {/* ACTIVE SPEAKERS IS GONE (2026-08-13, at the user's direction).
            Every speaker it listed is already on screen wearing their colour,
            in the captions and on the compass ring -- CWI 2.1 makes colour THE
            speaker signal, so a legend restating it is redundant by the design
            system's own logic. What the section added on top was
            `provisional · 95% confidence`, which is the system talking about
            its own certainty: operator information, and it belongs with the
            operator.
            The `speakers` memo went with it. It fed nothing else --
            `useSpeakerColors` reads `paragraphs` directly, which is what the
            captions and the compass ring have always been coloured from. */}
        {/* A fourth "system" section listed Recognition (a hardcoded string),
            Expression (the settings slider's own value) and Direction (the
            compass readout, again). None of it was a signal. */}
      </aside>

      {settingsOpen && (
        <>
          <button
            className="settings-scrim"
            aria-label="Close settings"
            onClick={() => setSettingsOpen(false)}
          />
          <SettingsPanel
            settings={settings}
            setSettings={setSettings}
            close={() => setSettingsOpen(false)}
          />
        </>
      )}
      {session.state !== "listening" && (
        <LanguageGate
          session={session}
          error={languageError}
          onSelect={selectLanguage}
        />
      )}
    </main>
  );
}
