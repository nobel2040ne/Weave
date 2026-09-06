"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  initialCaptionModel,
  reduceCaptionEvent,
  wordKey,
  type CaptionEvent,
  type CaptionModel,
  type CaptionWord,
  type LevelEvent,
} from "@/lib/caption-store";
import {acousticTimeMs} from "@/lib/motion-timing";
import {
  advanceClock,
  IDLE_CLOCK,
  monotonicTimeForAcousticMs,
  presentationNowMs,
  readAheadMs,
  turnMomentFor,
  type PlayheadClock,
} from "@/lib/caption-clock";

export type ConnectionState = "connecting" | "live" | "reconnecting" | "demo";

/** Everything a word needs in order to animate itself, frozen at discovery. */
export interface WordSchedule {
  turnAtMs: number;
  durationMs: number;
}
export type SessionState =
  | "checking"
  | "selecting"
  | "loading"
  | "listening"
  | "unavailable";

export interface LiveLanguageOption {
  id: "en" | "ko" | string;
  label: string;
  nativeLabel: string;
  description: string;
}

export interface LanguageSession {
  state: SessionState;
  language: string | null;
  languages: LiveLanguageOption[];
}

export interface RuntimeConfig {
  palette: string[];
  /** Speaker colors for the light stage: same hues, darkened to >=4.5:1. */
  paletteLight: string[];
  /** How many trailing entries of `palette` are CWI 2.1.2 supporting colours. */
  paletteSupportCount: number;
  displayMode: string;
  maxWords: number;
  paragraphWordLimit: number;
  stageParagraphHistory: number;
  stageWordsPerBlock: number;
  /** Floor on words per Stage row; below this a row stops reading as a phrase. */
  stageWordsMin: number;
  /** Rows the stack must fit before a shorter row (larger type) is preferred. */
  stageMinRows: number;
  /** CWI 2.2.1. How far the caption playhead runs behind the acoustic clock. */
  readAheadDelayMs: number;
  /** A word may not turn until it has been on screen this long. */
  minReadAheadMs: number;
  /** Stagger between successive floor-clamped (caught-up) words, so a burst
      released at an endpoint ripples instead of popping as one wall. */
  wordRevealCatchupGapMs: number;
  /** 2.2.1: "full white at 90% opacity" -- against 2.4.1's black box. */
  readAheadColor: string;
  /** The boxless light stage measures white at 1.05:1. See config.yaml. */
  readAheadColorLight: string;
  readAheadOpacity: number;
  /** Crossfade for the 2.2.2 turn. It eases with the lift, never a hard cut. */
  colorTurnMs: number;
  wordMotionBaseMs: number;
  wordMotionMaxMs: number;
  wordMotionSpanStretch: number;
  wordMotionMinMs: number;
  wordMotionFollowsSpeech: boolean;
  wordMotionSpeechScale: number;
  wordMotionSpeechFloorMs: number;
  /** Ceiling for the 2.2.3 pop; the crest has its own. */
  wordMotionPopMaxMs: number;
  syncPop: number;
  /** 2.2.3 on the enhanced clock -- measured off the PR film. */
  syncPopEnhanced: number;
  /** The pop is PROPORTIONAL to emphasis; this is what an unemphasised word
     gets, as a fraction of the full pop. A flat step cannot produce the
     1.02-1.15x band that 60% of the reference's words sit in. */
  syncPopFloorEnhanced: number;
  /** How far an ordinary word lifts, em -- the film's default treatment. */
  wordLiftEmEnhanced: number;
  /** The enhanced size cue's whole window, ms, for an unemphasised word. */
  wordMotionEnhancedMs: number;
  /** ...and how much longer it runs at full emphasis, ms. */
  wordMotionEnhancedEmphasisMs: number;
  /** Enhanced: how far the size cue trails the turn, ms. */
  crestLagMs: number;
  /** Enhanced: narrower 2.3 size deadband. */
  voiceScaleDeadbandEnhanced: number;
  /** Enhanced: slope of the continuous loudness ramp. */
  voiceScaleResponseEnhanced: number;
  /** Enhanced: where the size mapping pivots. */
  voiceScalePivotEnhanced: number;
  /** Enhanced: convexity of the loudness ramp. */
  voiceScaleCurveEnhanced: number;
  /** Enhanced: fitted loudness -> crest control points. */
  voiceScalePointsEnhanced?: Array<[number, number]>;
  /** How far the character wave is suppressed as the WORD's own volume departs
   *  from normal: the two scopes trade off. See config.yaml. */
  characterWaveFalloff: number;
  characterWaveFloor: number;
  /** CWI, from the recordings: a word that waits rises, and lands on its turn. */
  holdLiftEm: number;
  holdFullS: number;
  holdMaxS: number;
  holdMinS: number;
  /** Pre-roll: crouch + launch + float, before the landing. See globals.css. */
  holdPreMs: number;
  /** Plateau at full lift, between the turn and the descent. */
  holdHoldMs: number;
  holdLandMs: number;
  /** CWI 2.4.4/2.4.5: sound labels yield to speech; how long after it. */
  soundLingerMs: number;
  deliveryMotionEnabled: boolean;
  /** A drawn-out word holds its 2.2.3 cue slightly longer. */
  deliveryFlowDurationMs: number;
  /** Settled wght band. CWI 2.3.9: low pitch heavy, high pitch light. */
  weightRange: [number, number];
  /** Settled wdth band. CWI 2.3.9/2.3.10: rich harmonics wider. */
  widthRange: [number, number];
  /** Transient voice-size band around the baseline. CWI 2.3.6. */
  voiceScaleRange: [number, number];
  /** How much of 2.3.6's excursion the crest uses, applied about 2.3.5's
     baseline so the anchor stays exact at any value. */
  voiceScaleResponse: number;
  /** The same, below the baseline -- smaller, because shrinking costs legibility. */
  voiceScaleResponseQuiet: number;
  /** Band around the median where size does not move. See caption-motion.ts. */
  voiceScaleDeadband: number;
  /** Weight an emphasised word gains on top of 2.3.9. See caption-motion.ts. */
  weightEmphasis: number;
  deliveryMinConfidence: number;
  languages: LiveLanguageOption[];
  selectedLanguage: string | null;
  languageSelectionRequired: boolean;
}

export const DEFAULT_RUNTIME_CONFIG: RuntimeConfig = {
  palette: ["#e6ff2e", "#56e39f", "#55d7ff", "#c387ff", "#ff667d", "#ffb84d"],
  paletteLight: ["#6d7816", "#31805a", "#307b91", "#895fb4", "#bd4c5d", "#936a2c"],
  paletteSupportCount: 0,
  displayMode: "fast",
  maxWords: 8,
  paragraphWordLimit: 0,
  stageParagraphHistory: 6,
  stageWordsPerBlock: 8,
  stageWordsMin: 3,
  stageMinRows: 16,
  readAheadDelayMs: 2500,
  minReadAheadMs: 420,
  wordRevealCatchupGapMs: 60,
  readAheadColor: "#ffffff",
  readAheadColorLight: "#6e6e73",
  readAheadOpacity: 0.9,
  colorTurnMs: 90,
  wordMotionBaseMs: 420,
  wordMotionMaxMs: 1050,
  wordMotionSpanStretch: 0.42,
  wordMotionMinMs: 320,
  wordMotionFollowsSpeech: true,
  wordMotionSpeechScale: 2.4,
  wordMotionSpeechFloorMs: 160,
  wordMotionPopMaxMs: 700,
  syncPop: 0.15,
  syncPopEnhanced: 0.55,
  syncPopFloorEnhanced: 0.5,
  wordLiftEmEnhanced: 0.045,
  wordMotionEnhancedMs: 500,
  wordMotionEnhancedEmphasisMs: 0,
  crestLagMs: 40,
  voiceScaleDeadbandEnhanced: 0.1,
  voiceScaleResponseEnhanced: 0.5,
  voiceScalePivotEnhanced: 0.1,
  voiceScaleCurveEnhanced: 2.0,
  characterWaveFalloff: 1.0,
  characterWaveFloor: 0.0,
  holdLiftEm: 0.525,
  holdFullS: 0.88,
  holdMaxS: 1.06,
  holdMinS: 0.78,
  holdPreMs: 260,
  holdHoldMs: 420,
  holdLandMs: 290,
  soundLingerMs: 800,
  deliveryMotionEnabled: true,
  deliveryFlowDurationMs: 90,
  weightRange: [340, 760],
  widthRange: [82, 124],
  voiceScaleRange: [0.90, 1.20],
  voiceScaleResponse: 0.25,
  voiceScaleResponseQuiet: 0.55,
  voiceScaleDeadband: 0.34,
  weightEmphasis: 0.55,
  deliveryMinConfidence: 0.38,
  languages: [
    {
      id: "en",
      label: "English",
      nativeLabel: "English",
      description: "English streaming recognition",
    },
    {
      id: "ko",
      label: "Korean",
      nativeLabel: "한국어",
      description: "한국어 실시간 음성 인식",
    },
  ],
  selectedLanguage: null,
  languageSelectionRequired: true,
};

const EMPTY_LEVEL: LevelEvent = {
  type: "level",
  rms_db: -72,
  floor_db: -72,
  gain_db: 0,
  status: "idle",
  speech: false,
  pitch_hz: 0,
  pitch_confidence: 0,
  spectral_centroid_hz: 0,
};

/** How often the playhead is published to React. */
const PLAYHEAD_TICK_MS = 66;

interface StreamOptions {
  reducedMotion: boolean;
}

/** The newest word the browser currently HOLDS -- the far edge of the read-
   ahead. */
function newestAcousticMs(model: CaptionModel): number {
  const newest = model.order.at(-1);
  return newest ? acousticTimeMs(model.words[newest]) : Number.NaN;
}

function demoEvents(): Array<{delay: number; event: CaptionEvent}> {
  const base = {
    utterance: 0,
    loudness: 0.58,
    loudness_db: -27,
    pitch_hz: 172,
    voiced_frac: 0.91,
    delivery_force: 0.58,
    delivery_attack: 0.36,
    delivery_contour: 0,
    delivery_flow: 0.64,
    delivery_texture: 0.25,
    delivery_confidence: 0.88,
    delivery_profile: "steady",
    conf: 0.94,
    final: true,
    speaker: "S1",
    speaker_status: "stable" as const,
    speaker_confidence: 0.92,
  };
  const phrases = [
    ["The", 0.00, 0.22, 0.42, 188],
    ["voice", 0.23, 0.60, 0.72, 154],
    ["isn't", 0.62, 0.91, 0.48, 176],
    ["just", 0.94, 1.18, 0.62, 168],
    ["heard.", 1.20, 1.66, 0.82, 148],
  ] as const;
  const second = [
    ["Now", 1.95, 2.20, 0.54, 224],
    ["you", 2.22, 2.43, 0.46, 236],
    ["can", 2.45, 2.68, 0.58, 228],
    ["see", 2.70, 3.08, 0.74, 244],
    ["it.", 3.10, 3.42, 0.50, 218],
  ] as const;
  const events: Array<{delay: number; event: CaptionEvent}> = [
    {delay: 0, event: {type: "boot", stage: "listening"}},
  ];
  [...phrases, ...second].forEach((item, index) => {
    const [text, start, end, loudness, pitch] = item;
    const secondSpeaker = index >= phrases.length;
    events.push({
      delay: 180 + index * 155,
      event: {
        ...base,
        type: "word",
        word_id: `u0:w${index}`,
        text,
        t: start,
        start,
        end,
        loudness,
        pitch_hz: pitch,
        delivery_force: loudness,
        delivery_attack: index % 4 === 0 ? 0.82 : 0.28,
        delivery_contour: [-0.58, 0.04, 0.62, -0.08, -0.42][index % 5],
        delivery_flow: index % 3 === 1 ? 0.86 : 0.52,
        delivery_texture: index % 4 === 2 ? 0.71 : 0.24,
        delivery_profile: [
          "falling", "sustained", "rising", "steady", "forceful",
        ][index % 5],
        speaker: secondSpeaker ? "S2" : "S1",
        speaker_status: "stable",
        speaker_revision_id: 1,
        text_revision_id: 1,
        timing_revision_id: 1,
      },
    });
  });
  return events;
}

export function useCaptionStream({reducedMotion}: StreamOptions) {
  const [model, setModel] = useState<CaptionModel>(initialCaptionModel);
  const [level, setLevel] = useState<LevelEvent>(EMPTY_LEVEL);
  const [waveform, setWaveform] = useState<number[]>(() => Array(32).fill(-72));
  const [connection, setConnection] =
    useState<ConnectionState>("connecting");
  const [runtime, setRuntime] =
    useState<RuntimeConfig>(DEFAULT_RUNTIME_CONFIG);
  const [session, setSession] = useState<LanguageSession>({
    state: "checking",
    language: null,
    languages: DEFAULT_RUNTIME_CONFIG.languages,
  });
  const [languageError, setLanguageError] = useState<string | null>(null);
  const [playheadMs, setPlayheadMs] = useState(Number.NEGATIVE_INFINITY);
  const [clockEpoch, setClockEpoch] = useState<number | null>(null);
  const [startedAt] = useState(() => Date.now());
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);

  const modelRef = useRef(model);
  const levelRef = useRef(level);
  const scheduledRef = useRef(new Map<string, WordSchedule>());
  // The clock lives in a ref, not in state: it is revised ~15 times a second by
  // level events and must not re-render the studio on every one of them. The
  // coarse playhead tick below is what React actually sees.
  const clockRef = useRef<PlayheadClock>(IDLE_CLOCK);
  const runtimeRef = useRef(runtime);
  // The turn moment of the most recent floor-clamped word, and the epoch it
  // belongs to, so a caught-up burst is staggered rather than popped as a
  // wall. Reset when the capture (epoch) changes; see `scheduleWord`.
  const lastFloorTurnRef = useRef(Number.NEGATIVE_INFINITY);
  const lastFloorEpochRef = useRef<number | null>(null);
  const lateWordsRef = useRef(0);
  const frozenTextRef = useRef(0);
  // id -> the text the word was WEARING when the playhead reached it.
  const settledTextRef = useRef(new Map<string, string>());
  const rearmedWordsRef = useRef(0);
  const minReadAheadRef = useRef(Number.POSITIVE_INFINITY);
  const eventIdRef = useRef(0);
  /* AN ENDPOINT ARRIVES AS A BURST, AND A BURST IS ONE PAINT (2026-08-14).
     The server publishes an endpoint word by word: MEASURED on the PR film,
     74 SSE messages inside 100 ms, 59 of them `word`. Each one was its own
     `setModel`, so each one was its own React commit -- and two layout effects
     in `live-studio.tsx` (the stack FLIP and the row-grow) read
     `getBoundingClientRect` on every row right after every commit, which
     forces a synchronous style recalc + layout of a stage whose font-size,
     font-weight and font-stretch are all driven by animating custom
     properties. Traced: 64 `UpdateLayoutTree` and 65 `Layout` passes in half a
     second, and the frame the burst lands on stretched to 59 ms against an
     8.3 ms median -- the stutter you see when a held utterance turns colour.
     The events are queued here and applied in ONE reducer pass per animation
     frame. Nothing about WHEN a word turns changes: the turn is placed on the
     acoustic clock by `scheduleWord`, which is absolute, and `freezeText`
     below still runs at arrival so the caption invariant is unaffected. */
  const pendingRef = useRef<Array<{event: CaptionEvent; id: number}>>([]);
  const flushFrameRef = useRef(0);

  useEffect(() => {
    modelRef.current = model;
  }, [model]);
  useEffect(() => {
    levelRef.current = level;
  }, [level]);
  useEffect(() => {
    runtimeRef.current = runtime;
  }, [runtime]);

  const backendOrigin =
    process.env.NEXT_PUBLIC_AUTOCWI_ORIGIN?.replace(/\/$/, "") ?? "";

  /* Apply every event queued since the last frame, in arrival order, as one
     state change. Order and ids are exactly what the unbatched path passed, so
     the reducer sees the same sequence it always did. */
  const flushPending = useCallback(() => {
    flushFrameRef.current = 0;
    const pending = pendingRef.current;
    if (pending.length === 0) return;
    pendingRef.current = [];
    setModel((current) => {
      let next = current;
      for (const {event, id} of pending) {
        next = reduceCaptionEvent(next, event, id, new Set<string>());
      }
      return next;
    });
    setLastEventAt(Date.now());
  }, []);

  useEffect(() => () => {
    if (flushFrameRef.current) cancelAnimationFrame(flushFrameRef.current);
  }, []);

  const dispatch = useCallback((event: CaptionEvent, eventId?: number) => {
    const id = eventId ?? ++eventIdRef.current;
    if (event.type === "level") {
      setLastEventAt(Date.now());
      const incoming = event as unknown as LevelEvent;
      // THE CLOCK SOURCE. `level.t` is the capture position in seconds on the
      // same timeline as `word.t` (`stream_base + word.start`), and it arrives
      // every ~64 ms whether or not anyone is speaking, which is exactly what a
      // playhead needs. Word events are NOT used here: they are retained and
      // replayed to each new audience connection, so an old timestamp would
      // look like a capture restart and yank the playhead backwards.
      const seconds = Number(incoming.t);
      if (Number.isFinite(seconds)) {
        clockRef.current = advanceClock(
          clockRef.current,
          seconds * 1000,
          performance.now(),
        );
      }
      setLevel(incoming);
      setWaveform((values) => [...values.slice(-31), Number(incoming.rms_db)]);
      return;
    }
    /* THE CAPTION INVARIANT, ENFORCED RATHER THAN MERELY DOCUMENTED. */
    /** Same letters, differing only in punctuation or case. */
  const sameWordDifferentMarks = (a: string, b: string | undefined) => {
    if (typeof b !== "string") return false;
    const strip = (text: string) =>
      text.normalize("NFKC").toLowerCase()
        .replace(/[^\p{L}\p{N}']/gu, "");
    return strip(a) === strip(b) && strip(a).length > 0;
  };
  const freezeText = <T extends {text?: string}>(word: T): T => {
      const key = wordKey(word as unknown as CaptionWord);
      // Recorded when the playhead passed the word -- NOT read from the live
      // model here. The reducer routinely deletes a non-final word and re-adds
      // it a moment later with the verifier's spelling, and comparing against
      // the model alone let exactly that path through: at the instant of
      // re-insertion there was nothing to compare to. Measured, 14 coloured
      // captions still rewrote themselves ("tab" -> "tab.", "right" ->
      // "Right,") until this was remembered independently.
      const settled = settledTextRef.current.get(key);
      if (settled === undefined || settled === word.text) return word;
      /* ...BUT PUNCTUATION AND CASE ARE NOT A RESPELLING, AND REFUSING THEM
         COST THE CAPTIONS THEIR SENTENCES (2026-08-03). */
      if (sameWordDifferentMarks(settled, word.text)) return word;
      frozenTextRef.current += 1;
      return {...word, text: settled};
    };
    if (Array.isArray(event.words)) {
      event = {...event, words: event.words.map(freezeText)};
    } else if (typeof event.text === "string") {
      // `cue`/`commit`/`word` may carry a single word at the top level rather
      // than in a `words` array. Missing this path let endpoint punctuation
      // ("it" -> "it,", "okay" -> "okay?") still rewrite coloured captions.
      event = freezeText(event as CaptionEvent & {text: string});
    }
    pendingRef.current.push({event, id});
    if (flushFrameRef.current === 0) {
      flushFrameRef.current = requestAnimationFrame(flushPending);
    }
  }, [flushPending]);

  // RE-FETCHED WHEN THE LANGUAGE RESOLVES, not once on mount. The studio opens
  // before the picker is answered, so the first response describes the config
  // BEFORE a profile was chosen -- and a profile can override `display:` keys.
  // `read_ahead_delay_s` is the one that matters: the Korean and bilingual
  // recognizers are slower than English and carry their own read-ahead budget,
  // and reading the pre-language value served Korean a 1750 ms budget it
  // cannot meet, measured as 0 ms of CWI 2.2.1 read-ahead on the stage.
  const sessionLanguage = session.language;
  useEffect(() => {
    let cancelled = false;
    fetch(`${backendOrigin}/runtime-config.json`)
      .then((response) => response.ok ? response.json() : null)
      .then((value: Partial<RuntimeConfig> | null) => {
        if (!cancelled && value) {
          setRuntime({...DEFAULT_RUNTIME_CONFIG, ...value});
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [backendOrigin, sessionLanguage]);

  useEffect(() => {
    let cancelled = false;
    const isDemo = new URLSearchParams(window.location.search).has("demo");
    if (isDemo) {
      const timer = setTimeout(() => setSession({
          state: "listening",
          language: "en",
          languages: DEFAULT_RUNTIME_CONFIG.languages,
        }), 0);
      return () => clearTimeout(timer);
    }
    fetch(`${backendOrigin}/session`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Session service is unavailable.");
        return response.json() as Promise<LanguageSession>;
      })
      .then((value) => {
        if (!cancelled) setSession(value);
      })
      .catch(() => {
        if (!cancelled) {
          setSession((current) => ({...current, state: "unavailable"}));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [backendOrigin]);

  const selectLanguage = useCallback(async (language: string) => {
    setLanguageError(null);
    try {
      const response = await fetch(`${backendOrigin}/session/language`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({language}),
      });
      const value = await response.json() as LanguageSession & {error?: string};
      if (!response.ok) {
        throw new Error(value.error ?? "Could not select that language.");
      }
      setSession(value);
    } catch (error) {
      setLanguageError(
        error instanceof Error ? error.message : "Could not select that language.",
      );
    }
  }, [backendOrigin]);

  useEffect(() => {
    const isDemo = new URLSearchParams(window.location.search).has("demo");
    if (isDemo) {
      const connectionTimer = setTimeout(() => setConnection("demo"), 0);
      const timers: Array<ReturnType<typeof setTimeout>> = [];
      const demoStartedAt = performance.now();
      const levelTimer = setInterval(() => {
        const phase = performance.now() / 420;
        dispatch({
          type: "level",
          // The demo needs a real acoustic clock too, or the playhead never
          // starts and every word stays white. Its words span t=0..3.4 s, so a
          // wall clock from mount plays the read-ahead exactly as live does:
          // the whole block appears white first, then colours word by word.
          t: (performance.now() - demoStartedAt) / 1000,
          rms_db: -32 + Math.sin(phase) * 9,
          floor_db: -61,
          gain_db: 0,
          status: "good",
          speech: true,
          pitch_hz: 196 + Math.sin(phase * 0.72) * 44,
          pitch_confidence: 0.82,
          spectral_centroid_hz: 1900 + Math.cos(phase * 0.44) * 760,
          delivery_force: 0.52 + Math.sin(phase) * 0.24,
          delivery_attack: Math.max(0, Math.sin(phase * 1.8)) * 0.7,
          delivery_contour: Math.sin(phase * 0.72) * 0.72,
          delivery_flow: 0.68 + Math.cos(phase * 0.33) * 0.19,
          delivery_texture: 0.32 + Math.sin(phase * 0.44) * 0.18,
          delivery_confidence: 0.88,
          delivery_profile: Math.sin(phase * 0.72) > 0.25
            ? "rising"
            : Math.sin(phase * 0.72) < -0.25
              ? "falling"
              : "sustained",
        });
      }, 64);
      for (const item of demoEvents()) {
        timers.push(setTimeout(() => dispatch(item.event), item.delay));
      }
      return () => {
        clearTimeout(connectionTimer);
        clearInterval(levelTimer);
        timers.forEach(clearTimeout);
      };
    }

    const source = new EventSource(`${backendOrigin}/events`);
    source.onopen = () => setConnection("live");
    source.onerror = () => setConnection("reconnecting");
    source.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data) as CaptionEvent;
        if (event.type === "boot") {
          setSession((current) => ({
            ...current,
            state: event.stage === "listening" ? "listening" : "loading",
            language: String(event.language ?? current.language ?? "") || null,
          }));
        }
        dispatch(event, Number(message.lastEventId || 0));
      } catch {
        // A malformed diagnostic event must not tear down a live session.
      }
    };
    return () => source.close();
  }, [backendOrigin, dispatch]);

  /* SCHEDULING, IN FULL. A word is placed on the playhead once, by the word
     itself, in its own layout effect (see `MotionWord`). */
  const scheduleWord = useCallback((
    id: string,
    word: CaptionWord,
    durationMs: number,
  ): {turnAtMs: number; epoch: number} | null => {
    const clock = clockRef.current;
    const acousticMs = acousticTimeMs(word);
    // Before the first level event there is no acoustic clock and therefore no
    // honest moment to turn this word. Returning null leaves it in read-ahead
    // type; it is scheduled as soon as the clock starts.
    if (!clock.started || !Number.isFinite(acousticMs)) return null;
    const acousticTurnMs = monotonicTimeForAcousticMs(
      clock,
      acousticMs,
      runtimeRef.current.readAheadDelayMs,
    );
    const previous = scheduledRef.current.get(id);
    if (previous) {
      /* Return the STORED moment, not a recomputed one. */
      rearmedWordsRef.current += 1;
      return {turnAtMs: previous.turnAtMs, epoch: clock.epoch};
    }
    /* A WORD MUST BE READABLE BEFORE IT IS SPOKEN, A TIME DELAY ALONE CANNOT
       GUARANTEE THAT (2026-08-03), AND A BURST OF CAUGHT-UP WORDS MUST NOT POP
       AS ONE WALL (2026-09-04). Both live in `turnMomentFor`; see its note. The
       floor chain resets when the capture (epoch) changes. */
    if (lastFloorEpochRef.current !== clock.epoch) {
      lastFloorEpochRef.current = clock.epoch;
      lastFloorTurnRef.current = Number.NEGATIVE_INFINITY;
    }
    const moment = turnMomentFor(
      acousticTurnMs,
      performance.now(),
      runtimeRef.current.minReadAheadMs,
      lastFloorTurnRef.current,
      runtimeRef.current.wordRevealCatchupGapMs,
    );
    const turnAtMs = moment.turnAtMs;
    if (moment.clamped) lastFloorTurnRef.current = turnAtMs;
    scheduledRef.current.set(id, {turnAtMs, durationMs});
    /* A word delivered past its own onset needs no special case: the
       negative delay leaves the animation finished, so it paints settled. */
    if (turnAtMs < performance.now()) lateWordsRef.current += 1;
    return {turnAtMs, epoch: clock.epoch};
  }, []);

  /* Publish the playhead coarsely. No word's motion depends on this -- CSS
     runs those. */
  useEffect(() => {
    let frame = 0;
    let last = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      const now = performance.now();
      if (now - last < PLAYHEAD_TICK_MS) return;
      last = now;
      const clock = clockRef.current;
      if (!clock.started) return;
      // Publishing the epoch is what starts words scheduling, and what tells
      // words from a previous capture to settle. Both are rare transitions, so
      // this bails out on the common path.
      setClockEpoch((current) => (
        current === clock.epoch ? current : clock.epoch
      ));
      const playhead = presentationNowMs(
        clock,
        now,
        runtimeRef.current.readAheadDelayMs,
      );
      // A word's text is fixed the moment the playhead reaches it. Doing this
      // on the tick rather than on the next revision means a word that is
      // deleted and re-added still comes back wearing what the viewer read.
      for (const [id, schedule] of scheduledRef.current) {
        if (now < schedule.turnAtMs) continue;
        if (settledTextRef.current.has(id)) continue;
        const word = modelRef.current.words[id];
        if (word?.text !== undefined) {
          settledTextRef.current.set(id, word.text);
        }
      }

      const newest = newestAcousticMs(modelRef.current);
      if (Number.isFinite(newest) && scheduledRef.current.size > 0) {
        minReadAheadRef.current = Math.min(
          minReadAheadRef.current,
          readAheadMs(
            clock,
            newest,
            now,
            runtimeRef.current.readAheadDelayMs,
          ),
        );
      }
      setPlayheadMs(playhead);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    window.__cwiStudio = {
      dispatch,
      report: () => {
        const now = performance.now();
        const clock = clockRef.current;
        const delayMs = runtimeRef.current.readAheadDelayMs;
        const playhead = presentationNowMs(clock, now, delayMs);
        const live = new Set(modelRef.current.order);
        for (const id of scheduledRef.current.keys()) {
          if (!live.has(id)) scheduledRef.current.delete(id);
        }
        const scheduled = [...scheduledRef.current.values()];
        return {
          connection,
          words: modelRef.current.order.length,
          // Every recognized word is on screen now: read-ahead words are the
          // white ones. Nothing is withheld, so this must equal `words`.
          visible: modelRef.current.order.filter(
            (id) => Boolean(modelRef.current.words[id]),
          ).length,
          clockStarted: clock.started,
          readAheadDelayMs: delayMs,
          /* THE NUMBER THAT MATTERS. Recognized text sitting on screen ahead
             of the colour, in ms. */
          readAheadMs: readAheadMs(
            clock,
            newestAcousticMs(modelRef.current),
            now,
            delayMs,
          ),
          minReadAheadMs: Number.isFinite(minReadAheadRef.current)
            ? minReadAheadRef.current
            : 0,
          aheadWords: scheduled.filter((item) => item.turnAtMs > now).length,
          activeMotions: scheduled.filter(
            (item) => item.turnAtMs <= now &&
              now - item.turnAtMs < item.durationMs,
          ).length,
          scheduledWords: scheduled.length,
          /* Words that arrived after their own onset had already passed and
             so could never animate. */
          lateWords: lateWordsRef.current,
          // Revisions REJECTED because the playhead had already passed the
          // word. A healthy figure; zero would mean the invariant is unused.
          frozenTextRevisions: frozenTextRef.current,
          rearmedWords: rearmedWordsRef.current,
          playheadMs: Number.isFinite(playhead) ? playhead : null,
          newestAcousticMs: Number.isFinite(newestAcousticMs(
            modelRef.current,
          ))
            ? newestAcousticMs(modelRef.current)
            : null,
          clockEpoch: clock.epoch,
          reducedMotion,
          directionKnown: Number.isFinite(
            Number(levelRef.current.direction_deg ??
              levelRef.current.azimuth_deg),
          ),
        };
      },
    };
    return () => {
      delete window.__cwiStudio;
    };
  }, [connection, dispatch, reducedMotion]);

  // `?probe=1` publishes report() into document.title so headless Chrome can
  // read it with --dump-dom, matching how `scripts/live_render_probe.py` reads
  // the legacy renderer. Without this the studio's metrics are only reachable
  // from a devtools console, so motion could not be verified numerically --
  // --dump-dom alone returns the pre-hydration static shell.
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (new URLSearchParams(window.location.search).get("probe") !== "1") return;
    const publish = () => {
      const report = window.__cwiStudio?.report();
      if (report) document.title = `CWI_STUDIO_PROBE ${JSON.stringify(report)}`;
    };
    publish();
    const timer = window.setInterval(publish, 250);
    return () => window.clearInterval(timer);
  }, []);

  return {
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
    dispatch,
  };
}
