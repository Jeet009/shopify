/**
 * AI Setup check — suggests a product type and tags from the title/description.
 * The score/issues are computed deterministically (see setupScoreAndIssues);
 * the AI only proposes the type + tags to fill in.
 */

import { sarvamChat } from "../sarvam.server";
import { extractJsonObject } from "./llm-json";

export interface SetupScoreInput {
  title: string;
  description: string;
  currentType: string;
  currentTags: string[];
  guidance?: string;
}

export interface SetupSuggestion {
  suggestedType: string;
  suggestedTags: string[];
}

const SETUP_SYSTEM_PROMPT = `You categorize Shopify products for better organization and discovery.

Given a product, suggest:
- suggestedType: a single concise product type / category (e.g. "Snowboard", "Ski Wax").
- suggestedTags: 5–8 lowercase tags buyers might filter or search by (materials, use case, audience, style).

Respond with ONLY this JSON, no markdown fences:
{"suggestedType":"...","suggestedTags":["...","..."]}`;

export async function suggestSetup(
  input: SetupScoreInput,
): Promise<SetupSuggestion> {
  const guidance = (input.guidance ?? "").trim();
  const userContent =
    `Product title: ${input.title}\n` +
    `Description: ${input.description || "(none)"}\n` +
    `Current type: ${input.currentType || "(none)"}\n` +
    `Current tags: ${input.currentTags.join(", ") || "(none)"}` +
    (guidance ? `\n\nExtra instruction: ${guidance}` : "");

  const raw = await sarvamChat([
    { role: "system", content: SETUP_SYSTEM_PROMPT },
    { role: "user", content: userContent },
  ]);

  const parsed = extractJsonObject(raw) as Partial<SetupSuggestion>;
  return {
    suggestedType:
      typeof parsed.suggestedType === "string" ? parsed.suggestedType : "",
    suggestedTags: Array.isArray(parsed.suggestedTags)
      ? parsed.suggestedTags.map(String).slice(0, 12)
      : [],
  };
}
