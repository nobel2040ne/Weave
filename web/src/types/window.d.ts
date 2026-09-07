import type {CaptionEvent} from "@/lib/caption-store";

declare global {
  interface Window {
    __cwiStudio?: {
      dispatch: (event: CaptionEvent, eventId?: number) => void;
      report: () => {
        connection: string;
        words: number;
        visible: number;
        clockStarted: boolean;
        readAheadDelayMs: number;
        readAheadMs: number;
        minReadAheadMs: number;
        aheadWords: number;
        activeMotions: number;
        scheduledWords: number;
        lateWords: number;
        rearmedWords: number;
        frozenTextRevisions: number;
        frozenSpeakerRevisions: number;
        playheadMs: number | null;
        newestAcousticMs: number | null;
        clockEpoch: number;
        reducedMotion: boolean;
        directionKnown: boolean;
      };
    };
  }
}

export {};
