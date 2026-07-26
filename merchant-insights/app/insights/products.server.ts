/**
 * Server-only helpers for fetching product data from the Admin GraphQL API
 * and computing the rule-based (non-AI) health checks.
 */

import type { CheckResult } from "./checks";
import { statusFromScore } from "./checks";

export interface ProductData {
  id: string;
  title: string;
  handle: string;
  status: string;
  description: string; // plain text
  seoTitle: string;
  seoDescription: string;
  vendor: string;
  productType: string;
  tags: string[];
  imageUrl: string | null;
  imageCount: number;
  imagesMissingAlt: number;
  images: { id: string; url: string; alt: string }[];
  variants: {
    id: string;
    title: string;
    price: string;
    sku: string;
    inventory: number | null;
    inventoryItemId: string | null;
  }[];
}

/** Strip HTML tags / collapse whitespace so the model scores prose, not markup. */
export function toPlainText(html: string): string {
  return html
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Turn plain-text into simple HTML for Shopify's descriptionHtml field. */
export function toDescriptionHtml(text: string): string {
  const esc = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return esc
    .split(/\n{2,}/)
    .map((para) => `<p>${para.replace(/\n/g, "<br>")}</p>`)
    .join("");
}

const PRODUCT_FIELDS = `#graphql
  fragment HealthProductFields on Product {
    id
    title
    handle
    status
    description
    vendor
    productType
    tags
    seo { title description }
    featuredImage { url }
    images(first: 20) { edges { node { id url altText } } }
    variants(first: 100) {
      edges {
        node {
          id
          title
          price
          inventoryQuantity
          inventoryItem { id sku }
        }
      }
    }
  }`;

function mapProduct(node: any): ProductData {
  const imageEdges = node.images?.edges ?? [];
  return {
    id: node.id,
    title: node.title,
    handle: node.handle,
    status: node.status,
    description: toPlainText(node.description ?? ""),
    seoTitle: node.seo?.title ?? "",
    seoDescription: node.seo?.description ?? "",
    vendor: node.vendor ?? "",
    productType: node.productType ?? "",
    tags: node.tags ?? [],
    imageUrl: node.featuredImage?.url ?? null,
    imageCount: imageEdges.length,
    imagesMissingAlt: imageEdges.filter(
      (e: any) => !e.node?.altText || !e.node.altText.trim(),
    ).length,
    images: imageEdges.map((e: any) => ({
      id: e.node.id,
      url: e.node.url,
      alt: e.node.altText ?? "",
    })),
    variants: (node.variants?.edges ?? []).map((e: any) => ({
      id: e.node.id,
      title: e.node.title,
      price: e.node.price,
      sku: e.node.inventoryItem?.sku ?? "",
      inventory: e.node.inventoryQuantity,
      inventoryItemId: e.node.inventoryItem?.id ?? null,
    })),
  };
}

/** First location id — needed to set inventory quantities. */
export async function fetchPrimaryLocationId(
  admin: any,
): Promise<string | null> {
  const response = await admin.graphql(
    `#graphql
      query PrimaryLocation {
        locations(first: 1) { edges { node { id } } }
      }`,
  );
  const json = await response.json();
  return json.data?.locations?.edges?.[0]?.node?.id ?? null;
}

export async function fetchProducts(admin: any): Promise<ProductData[]> {
  const response = await admin.graphql(
    `#graphql
      ${PRODUCT_FIELDS}
      query HealthProducts {
        products(first: 100, sortKey: TITLE) {
          edges { node { ...HealthProductFields } }
        }
      }`,
  );
  const json = await response.json();
  return (json.data?.products?.edges ?? []).map((e: any) => mapProduct(e.node));
}

export async function fetchProduct(
  admin: any,
  id: string,
): Promise<ProductData | null> {
  const response = await admin.graphql(
    `#graphql
      ${PRODUCT_FIELDS}
      query HealthProduct($id: ID!) {
        product(id: $id) { ...HealthProductFields }
      }`,
    { variables: { id } },
  );
  const json = await response.json();
  return json.data?.product ? mapProduct(json.data.product) : null;
}

function ruleCheck(
  id: CheckResult["id"],
  label: string,
  score: number,
  issues: string[],
): CheckResult {
  return { id, label, ai: false, score, issues };
}

/** Images check: has a featured image and alt text on all images. */
export function imagesCheck(p: ProductData): CheckResult {
  const issues: string[] = [];
  let score = 100;
  if (p.imageCount === 0) {
    issues.push("Product has no images.");
    score = 0;
  } else if (p.imagesMissingAlt > 0) {
    issues.push(
      `${p.imagesMissingAlt} of ${p.imageCount} image(s) missing alt text (hurts SEO & accessibility).`,
    );
    score = Math.max(40, 100 - p.imagesMissingAlt * 25);
  }
  return ruleCheck("images", "Images", score, issues);
}

/**
 * Setup scoring (deterministic part). The suggested type/tags come from the AI
 * scorer; this computes the score + issues from what's currently filled in.
 */
export function setupScoreAndIssues(p: ProductData): {
  score: number;
  issues: string[];
} {
  const issues: string[] = [];
  let penalty = 0;
  if (!p.vendor.trim()) {
    issues.push("No vendor set.");
    penalty += 20;
  }
  if (!p.productType.trim()) {
    issues.push("No product type set.");
    penalty += 20;
  }
  if (p.tags.length === 0) {
    issues.push("No tags (hurts filtering & discovery).");
    penalty += 20;
  }
  if (p.status !== "ACTIVE") {
    issues.push(`Status is ${p.status}, not ACTIVE.`);
    penalty += 20;
  }
  return { score: Math.max(0, 100 - penalty), issues };
}

/** Inventory check: variants have prices, SKUs, and stock. */
export function inventoryCheck(p: ProductData): CheckResult {
  const issues: string[] = [];
  let score = 100;
  const noPrice = p.variants.filter((v) => !v.price || Number(v.price) === 0);
  const noSku = p.variants.filter((v) => !v.sku || !v.sku.trim());
  const noStock = p.variants.filter(
    (v) => v.inventory !== null && v.inventory <= 0,
  );
  if (p.variants.length === 0) {
    issues.push("No variants found.");
    score = 0;
  } else {
    if (noPrice.length) {
      issues.push(`${noPrice.length} variant(s) have no price.`);
      score -= 35;
    }
    if (noSku.length) {
      issues.push(`${noSku.length} variant(s) missing a SKU.`);
      score -= 20;
    }
    if (noStock.length) {
      issues.push(`${noStock.length} variant(s) out of stock.`);
      score -= 20;
    }
  }
  return ruleCheck("inventory", "Inventory", Math.max(0, score), issues);
}

/** All rule-based checks for a product (Setup is AI-powered, handled separately). */
export function ruleChecks(p: ProductData): CheckResult[] {
  return [imagesCheck(p), inventoryCheck(p)];
}

// Re-export so callers don't need a second import for tone helpers.
export { statusFromScore };
