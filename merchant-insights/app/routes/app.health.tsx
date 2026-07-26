import { useState } from "react";
import type { ActionFunctionArgs, LoaderFunctionArgs } from "@remix-run/node";
import { Link, useFetcher, useLoaderData } from "@remix-run/react";
import {
  Page,
  Card,
  Text,
  Badge,
  Banner,
  Button,
  BlockStack,
  InlineStack,
  InlineGrid,
  Collapsible,
  Tooltip,
  EmptyState,
  Thumbnail,
  Icon,
  Box,
} from "@shopify/polaris";
import { ImageIcon } from "@shopify/polaris-icons";
import { TitleBar } from "@shopify/app-bridge-react";

import { authenticate } from "../shopify.server";
import { fetchProducts, ruleChecks } from "../insights/products.server";
import {
  AI_CHECK_TYPES,
  aiCheckToResult,
  loadAiChecks,
  scanProduct,
  unscannedAiCheck,
} from "../insights/scan.server";
import {
  CHECK_ORDER,
  overallHealth,
  statusTone,
  checkStatus,
  type CheckResult,
  type CheckId,
} from "../insights/checks";
import { ProductCheckCards } from "../components/check-cards";

interface HealthRow {
  id: string;
  numericId: string;
  title: string;
  imageUrl: string | null;
  currentDescription: string;
  seoTitle: string;
  seoDescription: string;
  currentType: string;
  currentTags: string[];
  vendor: string;
  images: { id: string; url: string; alt: string }[];
  variants: {
    id: string;
    title: string;
    price: string;
    sku: string;
    inventory: number | null;
    inventoryItemId: string | null;
  }[];
  checks: CheckResult[];
  overall: number | null;
  hasUnscanned: boolean;
}

const SHORT_LABEL: Record<CheckId, string> = {
  description: "Desc",
  seo: "SEO",
  images: "Img",
  setup: "Setup",
  inventory: "Inv",
};

const GRID_COLUMNS = "minmax(0, 2fr) 110px minmax(0, 1.6fr) auto";

async function mapPool<T>(
  items: T[],
  concurrency: number,
  fn: (i: T) => Promise<void>,
) {
  const queue = [...items];
  await Promise.all(
    Array.from({ length: Math.min(concurrency, queue.length) }, async () => {
      while (queue.length) await fn(queue.shift()!);
    }),
  );
}

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const { admin, session } = await authenticate.admin(request);
  const shop = session.shop;

  const products = await fetchProducts(admin);
  const aiByProduct = await loadAiChecks(shop);

  const rows: HealthRow[] = products.map((p) => {
    const ai = aiByProduct.get(p.id);
    const aiChecks = AI_CHECK_TYPES.map((t) => {
      const row = ai?.get(t);
      return row ? aiCheckToResult(row) : unscannedAiCheck(t);
    });
    const byId = new Map<CheckId, CheckResult>();
    [...aiChecks, ...ruleChecks(p)].forEach((c) => byId.set(c.id, c));
    const checks = CHECK_ORDER.map((id) => byId.get(id)!).filter(Boolean);

    return {
      id: p.id,
      numericId: p.id.replace("gid://shopify/Product/", ""),
      title: p.title,
      imageUrl: p.imageUrl,
      currentDescription: p.description,
      seoTitle: p.seoTitle,
      seoDescription: p.seoDescription,
      currentType: p.productType,
      currentTags: p.tags,
      vendor: p.vendor,
      images: p.images,
      variants: p.variants,
      checks,
      overall: overallHealth(checks),
      hasUnscanned: checks.some((c) => c.ai && c.score === null),
    };
  });

  rows.sort((a, b) => (a.overall ?? 101) - (b.overall ?? 101));

  const scoredRows = rows.filter((r) => r.overall !== null);
  return {
    rows,
    apiKeyMissing: !process.env.SARVAM_API_KEY,
    stats: {
      total: rows.length,
      needAttention: scoredRows.filter((r) => (r.overall ?? 0) < 60).length,
      healthy: scoredRows.filter((r) => (r.overall ?? 0) >= 80).length,
      unscanned: rows.filter((r) => r.hasUnscanned).length,
    },
  };
};

export const action = async ({ request }: ActionFunctionArgs) => {
  const { admin, session } = await authenticate.admin(request);
  const shop = session.shop;
  if (!process.env.SARVAM_API_KEY) {
    return { ok: false, error: "SARVAM_API_KEY is not set." };
  }
  const form = await request.formData();
  const intent = String(form.get("intent"));

  const products = await fetchProducts(admin);

  if (intent === "scanOne") {
    const productId = String(form.get("productId"));
    const p = products.find((x) => x.id === productId);
    if (p) await scanProduct(shop, p);
    return { ok: true };
  }

  if (intent === "scanAll") {
    const existing = await loadAiChecks(shop);
    const targets = products.filter((p) => {
      const ai = existing.get(p.id);
      return !ai || AI_CHECK_TYPES.some((t) => !ai.has(t));
    });
    await mapPool(targets, 3, (p) => scanProduct(shop, p));
    return { ok: true, scanned: targets.length };
  }

  return { ok: false, error: "Unknown action." };
};

function HealthBadge({ score }: { score: number | null }) {
  if (score === null) return <Badge>Not scanned</Badge>;
  return (
    <Badge
      tone={statusTone(score >= 80 ? "good" : score >= 50 ? "warn" : "bad")}
    >
      {`${score}/100`}
    </Badge>
  );
}

function StatTile({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone?: "critical" | "success";
}) {
  return (
    <Box padding="400" background="bg-surface-secondary" borderRadius="200">
      <BlockStack gap="100">
        <Text as="span" variant="bodySm" tone="subdued">
          {label}
        </Text>
        <Text as="span" variant="headingLg" tone={tone}>
          {String(value)}
        </Text>
      </BlockStack>
    </Box>
  );
}

function CheckChips({ checks }: { checks: CheckResult[] }) {
  return (
    <InlineStack gap="100" wrap>
      {checks.map((c) => {
        const status = checkStatus(c);
        return (
          <Tooltip
            key={c.id}
            content={
              status === "unscanned"
                ? `${c.label}: not scanned yet`
                : `${c.label}: ${c.score}/100${c.issues[0] ? ` — ${c.issues[0]}` : ""}`
            }
          >
            <Badge tone={statusTone(status)} size="small">
              {SHORT_LABEL[c.id]}
            </Badge>
          </Tooltip>
        );
      })}
    </InlineStack>
  );
}

function ProductRow({ row }: { row: HealthRow }) {
  const [open, setOpen] = useState(false);
  const fetcher = useFetcher();
  const scanning = fetcher.state !== "idle";

  return (
    <Box
      paddingBlock="300"
      paddingInline="400"
      borderBlockEndWidth="025"
      borderColor="border"
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: GRID_COLUMNS,
          gap: 16,
          alignItems: "center",
        }}
      >
        <InlineStack gap="300" blockAlign="center" wrap={false}>
          {row.imageUrl ? (
            <Thumbnail source={row.imageUrl} alt={row.title} size="small" />
          ) : (
            <Box
              padding="200"
              background="bg-surface-secondary"
              borderRadius="200"
              borderWidth="025"
              borderColor="border"
            >
              <Icon source={ImageIcon} tone="subdued" />
            </Box>
          )}
          <Link to={`/app/health/${row.numericId}`}>
            <Text as="span" variant="bodyMd" fontWeight="semibold">
              {row.title}
            </Text>
          </Link>
        </InlineStack>
        <HealthBadge score={row.overall} />
        <CheckChips checks={row.checks} />
        <InlineStack gap="200" blockAlign="center" align="end">
          <Button
            disclosure={open ? "up" : "down"}
            onClick={() => setOpen((v) => !v)}
            variant="primary"
            tone="success"
            size="slim"
          >
            {open ? "Hide" : "Fix"}
          </Button>
          <fetcher.Form method="post">
            <input type="hidden" name="intent" value="scanOne" />
            <input type="hidden" name="productId" value={row.id} />
            <Button submit size="slim" loading={scanning}>
              {row.hasUnscanned ? "Scan" : "Re-scan"}
            </Button>
          </fetcher.Form>
        </InlineStack>
      </div>

      <Collapsible
        id={`expand-${row.numericId}`}
        open={open}
        transition={{ duration: "200ms", timingFunction: "ease-in-out" }}
      >
        <Box paddingBlockStart="400" paddingBlockEnd="200">
          <ProductCheckCards
            checks={row.checks}
            product={{
              currentDescription: row.currentDescription,
              seoTitle: row.seoTitle,
              seoDescription: row.seoDescription,
              currentType: row.currentType,
              currentTags: row.currentTags,
              vendor: row.vendor,
              images: row.images,
              variants: row.variants,
            }}
            adminUrl={`shopify:admin/products/${row.numericId}`}
            actionUrl={`/app/health/${row.numericId}`}
          />
        </Box>
      </Collapsible>
    </Box>
  );
}

export default function ProductHealth() {
  const { rows, apiKeyMissing, stats } = useLoaderData<typeof loader>();
  const scanAll = useFetcher();
  const scanning = scanAll.state !== "idle";

  return (
    <Page>
      <TitleBar title="Product Health" />
      <BlockStack gap="400">
        {apiKeyMissing && (
          <Banner tone="warning" title="Sarvam API key not set">
            <p>
              Add <code>SARVAM_API_KEY</code> to your <code>.env</code> and
              restart the dev server to run the AI checks (Description, SEO).
            </p>
          </Banner>
        )}

        <InlineGrid columns={{ xs: 1, sm: 3 }} gap="400">
          <StatTile label="Products" value={stats.total} />
          <StatTile
            label="Need attention"
            value={stats.needAttention}
            tone="critical"
          />
          <StatTile label="Healthy" value={stats.healthy} tone="success" />
        </InlineGrid>

        <Card padding="0">
          <Box padding="400">
            <InlineStack align="space-between" blockAlign="center">
              <BlockStack gap="100">
                <Text as="h2" variant="headingMd">
                  Catalog health
                </Text>
                <Text as="span" variant="bodySm" tone="subdued">
                  Description, SEO & Setup are AI-powered. Images & Inventory are
                  rule-based. Click <b>Fix</b> to resolve issues inline.
                </Text>
              </BlockStack>
              <scanAll.Form method="post">
                <input type="hidden" name="intent" value="scanAll" />
                <Button
                  submit
                  variant="primary"
                  loading={scanning}
                  disabled={apiKeyMissing || stats.unscanned === 0}
                >
                  {stats.unscanned > 0
                    ? `Scan ${stats.unscanned} product(s)`
                    : "All scanned"}
                </Button>
              </scanAll.Form>
            </InlineStack>
          </Box>

          {stats.total === 0 ? (
            <Box padding="400">
              <EmptyState
                heading="No products found"
                image="https://cdn.shopify.com/s/files/1/0262/4071/2726/files/emptystate-files.png"
              >
                <p>Add products to your store to see their health.</p>
              </EmptyState>
            </Box>
          ) : (
            <>
              <Box
                padding="400"
                background="bg-surface-secondary"
                borderBlockEndWidth="025"
                borderColor="border"
              >
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: GRID_COLUMNS,
                    gap: 16,
                  }}
                >
                  <Text as="span" variant="bodySm" tone="subdued" fontWeight="medium">
                    Product
                  </Text>
                  <Text as="span" variant="bodySm" tone="subdued" fontWeight="medium">
                    Health
                  </Text>
                  <Text as="span" variant="bodySm" tone="subdued" fontWeight="medium">
                    Checks
                  </Text>
                  <Text as="span" variant="bodySm" tone="subdued" fontWeight="medium" alignment="end">
                    Actions
                  </Text>
                </div>
              </Box>
              {rows.map((row) => (
                <ProductRow key={row.id} row={row} />
              ))}
            </>
          )}
        </Card>
      </BlockStack>
    </Page>
  );
}
