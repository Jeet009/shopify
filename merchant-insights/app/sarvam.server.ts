/**
 * Thin server-only client for the Sarvam Chat Completions API.
 *
 * Sarvam exposes an OpenAI-compatible endpoint:
 *   POST https://api.sarvam.ai/v1/chat/completions
 *   Auth: `api-subscription-key: <SARVAM_API_KEY>`  (key format: sk_...)
 *
 * Docs: https://docs.sarvam.ai/api-reference-docs/chat/chat-completions
 */

const SARVAM_BASE_URL = "https://api.sarvam.ai";

/** Largest model available (105B / 128K context). Override via SARVAM_MODEL. */
export const DEFAULT_SARVAM_MODEL = "sarvam-105b";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  /** Abort the request if it runs longer than this (ms). Default 60s. */
  timeoutMs?: number;
}

interface SarvamChatResponse {
  choices?: Array<{
    message?: { content?: string | null; reasoning_content?: string | null };
    finish_reason?: string;
  }>;
}

function getApiKey(): string {
  const key = process.env.SARVAM_API_KEY;
  if (!key) {
    throw new Error(
      "SARVAM_API_KEY is not set. Add it to your .env (get one at https://dashboard.sarvam.ai).",
    );
  }
  return key;
}

/**
 * Send a chat completion request to Sarvam and return the assistant's text.
 * Throws on non-2xx responses or timeouts.
 */
export async function sarvamChat(
  messages: ChatMessage[],
  opts: ChatOptions = {},
): Promise<string> {
  const {
    model = process.env.SARVAM_MODEL || DEFAULT_SARVAM_MODEL,
    temperature = 0.2,
    // sarvam-105b is a reasoning model: it spends tokens on hidden
    // `reasoning_content` before emitting the answer in `content`. Budget
    // generously so the final answer isn't truncated to null.
    maxTokens = 4096,
    timeoutMs = 90_000,
  } = opts;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${SARVAM_BASE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "api-subscription-key": getApiKey(),
      },
      body: JSON.stringify({
        model,
        messages,
        temperature,
        max_tokens: maxTokens,
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(
        `Sarvam API error ${res.status} ${res.statusText}: ${body.slice(0, 500)}`,
      );
    }

    const data = (await res.json()) as SarvamChatResponse;
    const choice = data.choices?.[0];
    const content = choice?.message?.content;
    if (!content) {
      // Most common cause: reasoning consumed the whole token budget before
      // any answer was emitted (finish_reason === "length").
      throw new Error(
        `Sarvam returned no answer content (finish_reason=${choice?.finish_reason ?? "unknown"}). ` +
          `Try a higher max_tokens.`,
      );
    }
    return content;
  } finally {
    clearTimeout(timeout);
  }
}
