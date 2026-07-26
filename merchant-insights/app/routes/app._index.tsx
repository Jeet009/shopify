import type { LoaderFunctionArgs } from "@remix-run/node";
import { Link as RemixLink, useLoaderData } from "@remix-run/react";
import {
  Page,
  Card,
  Text,
  Button,
  Icon,
  BlockStack,
  InlineStack,
  InlineGrid,
  Box,
} from "@shopify/polaris";
import {
  ProductIcon,
  HeartIcon,
  AlertTriangleIcon,
  CheckCircleIcon,
} from "@shopify/polaris-icons";
import { TitleBar } from "@shopify/app-bridge-react";

import { authenticate } from "../shopify.server";
import { Logo } from "../components/logo";
import { fetchProducts, ruleChecks } from "../insights/products.server";
import {
  AI_CHECK_TYPES,
  aiCheckToResult,
  loadAiChecks,
  unscannedAiCheck,
} from "../insights/scan.server";
import {
  CHECK_LABELS,
  overallHealth,
  type CheckId,
  type CheckResult,
} from "../insights/checks";

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const { admin, session } = await authenticate.admin(request);
  const shop = session.shop;

  const products = await fetchProducts(admin);
  const aiByProduct = await loadAiChecks(shop);

  const rows = products.map((p) => {
    const ai = aiByProduct.get(p.id);
    const checks: CheckResult[] = [
      ...AI_CHECK_TYPES.map((t) =>
        ai?.get(t) ? aiCheckToResult(ai.get(t)!) : unscannedAiCheck(t),
      ),
      ...ruleChecks(p),
    ];
    return { overall: overallHealth(checks), checks };
  });

  const scored = rows.filter((r) => r.overall !== null);
  const avg = scored.length
    ? Math.round(scored.reduce((s, r) => s + (r.overall ?? 0), 0) / scored.length)
    : null;

  const issueCounts: { id: CheckId; label: string; count: number }[] = (
    Object.keys(CHECK_LABELS) as CheckId[]
  )
    .map((id) => {
      const count = rows.filter((r) => {
        const c = r.checks.find((x) => x.id === id);
        return c && c.score !== null && c.score < 60;
      }).length;
      return { id, label: CHECK_LABELS[id], count };
    })
    .filter((i) => i.count > 0)
    .sort((a, b) => b.count - a.count);

  return {
    stats: {
      total: rows.length,
      avg,
      healthy: scored.filter((r) => (r.overall ?? 0) >= 80).length,
      ok: scored.filter((r) => (r.overall ?? 0) >= 50 && (r.overall ?? 0) < 80)
        .length,
      poor: scored.filter((r) => (r.overall ?? 0) < 50).length,
      scored: scored.length,
      unscanned: rows.length - scored.length,
    },
    issueCounts,
  };
};

const TONE_VAR = {
  success: "var(--p-color-bg-fill-success)",
  warning: "var(--p-color-bg-fill-warning)",
  critical: "var(--p-color-bg-fill-critical)",
} as const;

type Tone = keyof typeof TONE_VAR;

function toneForScore(v: number): Tone {
  return v >= 80 ? "success" : v >= 50 ? "warning" : "critical";
}

/* ---------- Stat tile ---------- */

function StatTile({
  label,
  value,
  icon,
  bg,
  iconTone,
}: {
  label: string;
  value: string;
  icon: React.FunctionComponent<React.SVGProps<SVGSVGElement>>;
  bg: string;
  iconTone: "success" | "critical" | "magic" | "subdued";
}) {
  return (
    <Box
      padding="400"
      background="bg-surface"
      borderRadius="300"
      borderWidth="025"
      borderColor="border"
      shadow="100"
    >
      <InlineStack gap="300" blockAlign="center">
        <Box background={bg as any} borderRadius="200" padding="200">
          <Icon source={icon} tone={iconTone} />
        </Box>
        <BlockStack gap="050">
          <Text as="span" variant="bodySm" tone="subdued">
            {label}
          </Text>
          <Text as="span" variant="heading2xl">
            {value}
          </Text>
        </BlockStack>
      </InlineStack>
    </Box>
  );
}

/* ---------- Radial gauge (average health) ---------- */

function RadialScore({ value }: { value: number | null }) {
  const size = 190;
  const stroke = 16;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = value ?? 0;
  const offset = c * (1 - v / 100);
  const color = value === null ? "var(--p-color-border)" : TONE_VAR[toneForScore(v)];

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--p-color-bg-surface-secondary)"
          strokeWidth={stroke}
        />
        {value !== null && (
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeDasharray={c}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
          />
        )}
      </svg>
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span style={{ fontSize: 44, fontWeight: 700, color: "var(--p-color-text)" }}>
          {value === null ? "—" : value}
        </span>
        <span style={{ fontSize: 13, color: "var(--p-color-text-secondary)" }}>
          out of 100
        </span>
      </div>
    </div>
  );
}

/* ---------- Donut (distribution) ---------- */

function Donut({
  segments,
}: {
  segments: { label: string; value: number; tone: Tone }[];
}) {
  const size = 190;
  const stroke = 26;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const total = Math.max(1, segments.reduce((s, x) => s + x.value, 0));
  let acc = 0;

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--p-color-bg-surface-secondary)"
          strokeWidth={stroke}
        />
        {segments.map((s, i) => {
          const len = (s.value / total) * c;
          const el = (
            <circle
              key={i}
              cx={size / 2}
              cy={size / 2}
              r={r}
              fill="none"
              stroke={TONE_VAR[s.tone]}
              strokeWidth={stroke}
              strokeDasharray={`${len} ${c - len}`}
              strokeDashoffset={-acc}
              transform={`rotate(-90 ${size / 2} ${size / 2})`}
            />
          );
          acc += len;
          return el;
        })}
      </svg>
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span style={{ fontSize: 32, fontWeight: 700, color: "var(--p-color-text)" }}>
          {segments.reduce((s, x) => s + x.value, 0)}
        </span>
        <span style={{ fontSize: 13, color: "var(--p-color-text-secondary)" }}>
          scanned
        </span>
      </div>
    </div>
  );
}

function LegendDot({ tone, label }: { tone: Tone; label: string }) {
  return (
    <InlineStack gap="150" blockAlign="center">
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: 3,
          background: TONE_VAR[tone],
          display: "inline-block",
        }}
      />
      <Text as="span" variant="bodySm">
        {label}
      </Text>
    </InlineStack>
  );
}

/* ---------- Horizontal issue bars ---------- */

function IssueBars({ items }: { items: { label: string; count: number }[] }) {
  const max = Math.max(1, ...items.map((i) => i.count));
  return (
    <BlockStack gap="300">
      {items.map((i) => (
        <BlockStack gap="100" key={i.label}>
          <InlineStack align="space-between">
            <Text as="span" variant="bodySm">{`Weak ${i.label}`}</Text>
            <Text as="span" variant="bodySm" fontWeight="semibold">
              {String(i.count)}
            </Text>
          </InlineStack>
          <div
            style={{
              height: 8,
              borderRadius: 4,
              background: "var(--p-color-bg-surface-secondary)",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${(i.count / max) * 100}%`,
                background: "var(--p-color-bg-fill-critical)",
                borderRadius: 4,
              }}
            />
          </div>
        </BlockStack>
      ))}
    </BlockStack>
  );
}

export default function Dashboard() {
  const { stats, issueCounts } = useLoaderData<typeof loader>();
  const hasScans = stats.scored > 0;

  return (
    <Page>
      <TitleBar title="Merchant Insights" />
      <BlockStack gap="500">
        <Card>
          <InlineStack gap="400" blockAlign="center">
            <Logo size={48} />
            <BlockStack gap="050">
              <Text as="h1" variant="headingLg">
                Merchant Insights
              </Text>
              <Text as="p" tone="subdued" variant="bodyMd">
                AI-powered catalog health, SEO, and inventory insights for your store.
              </Text>
            </BlockStack>
          </InlineStack>
        </Card>

        <InlineGrid columns={{ xs: 1, sm: 2, md: 4 }} gap="400">
          <StatTile
            label="Products"
            value={String(stats.total)}
            icon={ProductIcon}
            bg="bg-surface-info"
            iconTone="subdued"
          />
          <StatTile
            label="Average health"
            value={stats.avg === null ? "—" : `${stats.avg}`}
            icon={HeartIcon}
            bg="bg-surface-magic"
            iconTone="magic"
          />
          <StatTile
            label="Need attention"
            value={String(stats.poor + stats.ok)}
            icon={AlertTriangleIcon}
            bg="bg-surface-critical"
            iconTone="critical"
          />
          <StatTile
            label="Healthy"
            value={String(stats.healthy)}
            icon={CheckCircleIcon}
            bg="bg-surface-success"
            iconTone="success"
          />
        </InlineGrid>

        <InlineGrid columns={{ xs: 1, md: 2 }} gap="400">
          <Card>
            <BlockStack gap="400">
              <Text as="h2" variant="headingMd">
                Catalog health score
              </Text>
              <Box paddingBlock="200">
                <InlineStack align="center">
                  <RadialScore value={stats.avg} />
                </InlineStack>
              </Box>
              {!hasScans ? (
                <InlineStack align="center">
                  <Button url="/app/health" variant="primary">
                    Scan your catalog
                  </Button>
                </InlineStack>
              ) : (
                <Text as="p" tone="subdued" variant="bodySm" alignment="center">
                  Average across {stats.scored} scanned product(s).
                  {stats.unscanned > 0 && ` ${stats.unscanned} not scanned yet.`}
                </Text>
              )}
            </BlockStack>
          </Card>

          <Card>
            <BlockStack gap="400">
              <Text as="h2" variant="headingMd">
                Health distribution
              </Text>
              <InlineStack gap="500" blockAlign="center" align="center">
                <Donut
                  segments={[
                    { label: "Healthy", value: stats.healthy, tone: "success" },
                    { label: "Needs work", value: stats.ok, tone: "warning" },
                    { label: "Poor", value: stats.poor, tone: "critical" },
                  ]}
                />
                <BlockStack gap="200">
                  <LegendDot tone="success" label={`Healthy · ${stats.healthy}`} />
                  <LegendDot tone="warning" label={`Needs work · ${stats.ok}`} />
                  <LegendDot tone="critical" label={`Poor · ${stats.poor}`} />
                </BlockStack>
              </InlineStack>
            </BlockStack>
          </Card>
        </InlineGrid>

        <Card>
          <BlockStack gap="400">
            <InlineStack align="space-between" blockAlign="center">
              <Text as="h2" variant="headingMd">
                Top issues
              </Text>
              <Button url="/app/health">Open Product Health</Button>
            </InlineStack>
            {issueCounts.length === 0 ? (
              <Text as="p" tone="subdued">
                No issues found yet — scan products in Product Health to populate
                this.
              </Text>
            ) : (
              <IssueBars items={issueCounts} />
            )}
          </BlockStack>
        </Card>

        <Card>
          <BlockStack gap="200">
            <Text as="h2" variant="headingMd">
              Quick links
            </Text>
            <InlineStack gap="300">
              <RemixLink to="/app/health">Product Health</RemixLink>
              <RemixLink to="/app/inventory">Inventory &amp; Pricing</RemixLink>
            </InlineStack>
          </BlockStack>
        </Card>
      </BlockStack>
    </Page>
  );
}
