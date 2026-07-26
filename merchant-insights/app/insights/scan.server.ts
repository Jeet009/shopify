/**
 * Orchestrates the AI-powered checks (description, SEO): runs them against
 * Sarvam, persists results to the AiCheck cache, and reads them back as
 * client-safe CheckResult objects.
 */

import type { AiCheck } from "@prisma/client";

import prisma from "../db.server";
import type { CheckResult } from "./checks";
import { CHECK_LABELS } from "./checks";
import { scoreDescription } from "./description-score.server";
import { scoreSeo } from "./seo-score.server";
import { suggestSetup } from "./setup-score.server";
import { setupScoreAndIssues, type ProductData } from "./products.server";

export type AiCheckType = "description" | "seo" | "setup";
export const AI_CHECK_TYPES: AiCheckType[] = ["description", "seo", "setup"];

/** Turn a persisted AiCheck row into a client-safe CheckResult. */
export function aiCheckToResult(row: AiCheck): CheckResult {
  const id = row.checkType as AiCheckType;
  return {
    id,
    label: CHECK_LABELS[id],
    ai: true,
    score: row.score,
    issues: JSON.parse(row.issues),
    suggestion: row.suggestion,
    suggestionMeta: row.suggestionMeta ?? undefined,
    applied: Boolean(row.appliedAt),
  };
}

/** A not-yet-scanned AI check, so the UI can show a "Scan" prompt. */
export function unscannedAiCheck(id: AiCheckType): CheckResult {
  return { id, label: CHECK_LABELS[id], ai: true, score: null, issues: [] };
}

/** Load all cached AI checks for a shop, grouped by productId then checkType. */
export async function loadAiChecks(
  shop: string,
): Promise<Map<string, Map<AiCheckType, AiCheck>>> {
  const rows = await prisma.aiCheck.findMany({ where: { shop } });
  const byProduct = new Map<string, Map<AiCheckType, AiCheck>>();
  for (const row of rows) {
    if (!byProduct.has(row.productId)) byProduct.set(row.productId, new Map());
    byProduct.get(row.productId)!.set(row.checkType as AiCheckType, row);
  }
  return byProduct;
}

async function upsert(
  shop: string,
  productId: string,
  checkType: AiCheckType,
  data: { score: number; issues: string[]; suggestion: string; suggestionMeta?: string },
) {
  const payload = {
    score: data.score,
    issues: JSON.stringify(data.issues),
    suggestion: data.suggestion,
    suggestionMeta: data.suggestionMeta ?? null,
  };
  await prisma.aiCheck.upsert({
    where: { shop_productId_checkType: { shop, productId, checkType } },
    create: { shop, productId, checkType, ...payload },
    // A new suggestion invalidates any prior "applied" state.
    update: { ...payload, scoredAt: new Date(), appliedAt: null },
  });
}

export async function runDescriptionScan(
  shop: string,
  p: ProductData,
  guidance?: string,
) {
  const r = await scoreDescription({
    title: p.title,
    description: p.description,
    guidance,
  });
  await upsert(shop, p.id, "description", {
    score: r.score,
    issues: r.issues,
    suggestion: r.suggestedRewrite,
  });
}

export async function runSeoScan(shop: string, p: ProductData, guidance?: string) {
  const r = await scoreSeo({
    title: p.title,
    description: p.description,
    currentSeoTitle: p.seoTitle,
    currentSeoDescription: p.seoDescription,
    guidance,
  });
  await upsert(shop, p.id, "seo", {
    score: r.score,
    issues: r.issues,
    suggestion: r.suggestedTitle,
    suggestionMeta: r.suggestedMetaDescription,
  });
}

export async function runSetupScan(shop: string, p: ProductData, guidance?: string) {
  const { score, issues } = setupScoreAndIssues(p);
  const s = await suggestSetup({
    title: p.title,
    description: p.description,
    currentType: p.productType,
    currentTags: p.tags,
    guidance,
  });
  await upsert(shop, p.id, "setup", {
    score,
    issues,
    suggestion: s.suggestedType,
    suggestionMeta: JSON.stringify(s.suggestedTags),
  });
}

/** Run every AI check for one product. */
export async function scanProduct(shop: string, p: ProductData) {
  await Promise.all([
    runDescriptionScan(shop, p),
    runSeoScan(shop, p),
    runSetupScan(shop, p),
  ]);
}

/** Mark an applied AI check so the UI reflects it. */
export async function markApplied(
  shop: string,
  productId: string,
  checkType: AiCheckType,
) {
  await prisma.aiCheck.update({
    where: { shop_productId_checkType: { shop, productId, checkType } },
    data: { appliedAt: new Date() },
  });
}

export async function getAiCheck(
  shop: string,
  productId: string,
  checkType: AiCheckType,
): Promise<AiCheck | null> {
  return prisma.aiCheck.findUnique({
    where: { shop_productId_checkType: { shop, productId, checkType } },
  });
}
