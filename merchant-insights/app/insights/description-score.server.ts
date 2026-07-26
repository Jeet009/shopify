/**
 * Insight #1 — Weak product descriptions.
 *
 * Scores a product description 0–100 against a fixed rubric using the Sarvam
 * LLM and returns a suggested rewrite. Uses only product data (no protected
 * customer/order data), so it needs no Shopify protected-data approval.
 */

import { sarvamChat } from "../sarvam.server";
import { extractJsonObject, clampScore } from "./llm-json";
import type {
  DescriptionScore,
  DescriptionScoreInput,
} from "./description-score";

// Re-export the client-safe pieces so existing server-side imports keep working.
export {
  WEAK_DESCRIPTION_THRESHOLD,
  isWeakDescription,
  type DescriptionScore,
  type DescriptionScoreInput,
} from "./description-score";

const RUBRIC_SYSTEM_PROMPT = `You are an e-commerce merchandising expert who audits Shopify product descriptions.

Score the description from 0–100 using these four equally-weighted criteria (each 0–100):
- length: Is it substantial enough to inform a buyer (roughly 50–200 words), without padding?
- benefitLanguage: Does it sell outcomes/benefits, not just list features?
- specificity: Concrete details — materials, dimensions, use cases, what's included — vs vague filler.
- seoKeywords: Natural inclusion of terms a shopper would search for.

The overall "score" is the average of the four criteria, rounded to an integer.
Then write an improved "suggestedRewrite": persuasive, specific, scannable, 50–150 words, plain text (no HTML).

Respond with ONLY a JSON object, no markdown fences, in exactly this shape:
{"score":0,"breakdown":{"length":0,"benefitLanguage":0,"specificity":0,"seoKeywords":0},"issues":["..."],"suggestedRewrite":"..."}`;

/**
 * Score a single product's description. Throws if SARVAM_API_KEY is missing
 * or the API call fails — callers should handle/aggregate per-product errors.
 */
export async function scoreDescription(
  input: DescriptionScoreInput,
): Promise<DescriptionScore> {
  const description = (input.description ?? "").trim();

  // Even when there's no description we still call the model so it can write
  // one from scratch (score the criteria 0, but produce a suggestedRewrite).
  const base = description
    ? `Product title: ${input.title}\n\nDescription:\n${description}`
    : `Product title: ${input.title}\n\nDescription:\n(No description provided. Score every criterion 0, and write a brand-new description for "${input.title}" as the suggestedRewrite.)`;

  const guidance = (input.guidance ?? "").trim();
  const userContent = guidance
    ? `${base}\n\nExtra instruction for the suggestedRewrite (follow this, but keep scoring the ORIGINAL description): ${guidance}`
    : base;

  const raw = await sarvamChat([
    { role: "system", content: RUBRIC_SYSTEM_PROMPT },
    { role: "user", content: userContent },
  ]);

  const parsed = extractJsonObject(raw) as Partial<DescriptionScore> & {
    breakdown?: Partial<DescriptionScore["breakdown"]>;
  };

  const breakdown = {
    length: clampScore(parsed.breakdown?.length),
    benefitLanguage: clampScore(parsed.breakdown?.benefitLanguage),
    specificity: clampScore(parsed.breakdown?.specificity),
    seoKeywords: clampScore(parsed.breakdown?.seoKeywords),
  };

  return {
    score: clampScore(parsed.score),
    breakdown,
    issues: Array.isArray(parsed.issues) ? parsed.issues.map(String) : [],
    suggestedRewrite:
      typeof parsed.suggestedRewrite === "string" ? parsed.suggestedRewrite : "",
  };
}
