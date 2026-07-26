/**
 * Shared, client-safe types and pure helpers for the description insight.
 *
 * Keep this file free of server-only imports (no Sarvam, no env, no fetch) so
 * both the loader (server) and the route component (client) can import it.
 * The actual scoring logic lives in `description-score.server.ts`.
 */

export interface DescriptionScoreInput {
  /** Product title — gives the model context for the rewrite. */
  title: string;
  /** Raw product description. HTML is fine; the model is told to ignore markup. */
  description: string;
  /**
   * Optional merchant instruction to steer the rewrite, e.g. "make it punchier"
   * or "mention it's handmade". Empty/undefined = no extra steering.
   */
  guidance?: string;
}

export interface DescriptionScore {
  /** Overall quality, 0–100. Products under 60 are considered "weak". */
  score: number;
  /** Per-criterion breakdown, each 0–100. */
  breakdown: {
    length: number;
    benefitLanguage: number;
    specificity: number;
    seoKeywords: number;
  };
  /** Short, concrete reasons the description lost points. */
  issues: string[];
  /** A ready-to-use improved description. */
  suggestedRewrite: string;
}

/** Products scoring below this are flagged in the dashboard. */
export const WEAK_DESCRIPTION_THRESHOLD = 60;

export function isWeakDescription(result: DescriptionScore): boolean {
  return result.score < WEAK_DESCRIPTION_THRESHOLD;
}
