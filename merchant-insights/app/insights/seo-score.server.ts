/**
 * AI SEO check — scores a product's SEO title + meta description and suggests
 * optimized replacements. Uses only product data (no protected approval).
 */

import { sarvamChat } from "../sarvam.server";
import { extractJsonObject, clampScore } from "./llm-json";

export interface SeoScoreInput {
  title: string;
  description: string;
  currentSeoTitle: string;
  currentSeoDescription: string;
  /** Optional merchant steering for the suggestions. */
  guidance?: string;
}

export interface SeoScore {
  score: number;
  issues: string[];
  /** Suggested SEO page title (<= 60 chars). */
  suggestedTitle: string;
  /** Suggested meta description (<= 160 chars). */
  suggestedMetaDescription: string;
}

const SEO_SYSTEM_PROMPT = `You are an e-commerce SEO expert auditing a Shopify product's search metadata.

Score 0–100 based on the CURRENT SEO title and meta description:
- Title present, ~50–60 chars, includes the product + a keyword.
- Meta description present, ~140–160 chars, compelling, with keywords and a call to action.
Missing fields score very low.

Then produce optimized replacements:
- suggestedTitle: <= 60 characters.
- suggestedMetaDescription: <= 160 characters.

Respond with ONLY this JSON, no markdown fences:
{"score":0,"issues":["..."],"suggestedTitle":"...","suggestedMetaDescription":"..."}`;

export async function scoreSeo(input: SeoScoreInput): Promise<SeoScore> {
  const guidance = (input.guidance ?? "").trim();
  const userContent =
    `Product title: ${input.title}\n` +
    `Product description: ${input.description || "(none)"}\n\n` +
    `Current SEO title: ${input.currentSeoTitle || "(empty)"}\n` +
    `Current SEO meta description: ${input.currentSeoDescription || "(empty)"}` +
    (guidance ? `\n\nExtra instruction for the suggestions: ${guidance}` : "");

  const raw = await sarvamChat([
    { role: "system", content: SEO_SYSTEM_PROMPT },
    { role: "user", content: userContent },
  ]);

  const parsed = extractJsonObject(raw) as Partial<SeoScore>;
  return {
    score: clampScore(parsed.score),
    issues: Array.isArray(parsed.issues) ? parsed.issues.map(String) : [],
    suggestedTitle:
      typeof parsed.suggestedTitle === "string"
        ? parsed.suggestedTitle.slice(0, 70)
        : "",
    suggestedMetaDescription:
      typeof parsed.suggestedMetaDescription === "string"
        ? parsed.suggestedMetaDescription.slice(0, 200)
        : "",
  };
}
