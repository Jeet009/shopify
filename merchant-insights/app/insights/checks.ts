/**
 * Shared, client-safe types and helpers for Product Health checks.
 *
 * A "check" inspects one aspect of a product and returns a 0–100 score plus
 * issues. Some checks are AI-powered (description, SEO) and produce a suggested
 * fix; others are rule-based (images, setup, inventory) and just flag problems.
 *
 * Keep this file free of server-only imports so the route components can use it.
 */

export type CheckId = "description" | "seo" | "images" | "setup" | "inventory";
export type CheckStatus = "good" | "warn" | "bad";

export interface CheckResult {
  id: CheckId;
  label: string;
  /** Whether this check needs an AI scan (vs. computed instantly from rules). */
  ai: boolean;
  /** 0–100, or null for an AI check that hasn't been scanned yet. */
  score: number | null;
  issues: string[];
  /** AI checks only: primary suggested value (rewritten description / SEO title). */
  suggestion?: string;
  /** SEO check only: suggested meta description. */
  suggestionMeta?: string;
  /** Whether the suggested fix has been applied back to the product. */
  applied?: boolean;
}

export const CHECK_LABELS: Record<CheckId, string> = {
  description: "Description",
  seo: "SEO",
  images: "Images",
  setup: "Setup",
  inventory: "Inventory",
};

/** Order checks appear in the UI. */
export const CHECK_ORDER: CheckId[] = [
  "description",
  "seo",
  "setup",
  "images",
  "inventory",
];

export function statusFromScore(score: number): CheckStatus {
  if (score >= 80) return "good";
  if (score >= 50) return "warn";
  return "bad";
}

export function checkStatus(check: CheckResult): CheckStatus | "unscanned" {
  if (check.score === null) return "unscanned";
  return statusFromScore(check.score);
}

export function statusTone(
  status: CheckStatus | "unscanned",
): "success" | "warning" | "critical" | undefined {
  switch (status) {
    case "good":
      return "success";
    case "warn":
      return "warning";
    case "bad":
      return "critical";
    default:
      return undefined;
  }
}

/** Overall health = average of all scored checks (unscanned checks ignored). */
export function overallHealth(checks: CheckResult[]): number | null {
  const scored = checks.filter((c) => c.score !== null) as (CheckResult & {
    score: number;
  })[];
  if (!scored.length) return null;
  return Math.round(scored.reduce((s, c) => s + c.score, 0) / scored.length);
}
