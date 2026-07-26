import type { LoaderFunctionArgs } from "@remix-run/node";
import { Link, useLoaderData } from "@remix-run/react";
import {
  Page,
  Card,
  Text,
  Badge,
  Icon,
  BlockStack,
  InlineStack,
  InlineGrid,
  IndexTable,
  EmptyState,
  Box,
} from "@shopify/polaris";
import {
  AlertTriangleIcon,
  InventoryIcon,
  CashDollarIcon,
  BarcodeIcon,
} from "@shopify/polaris-icons";
import { TitleBar } from "@shopify/app-bridge-react";

import { authenticate } from "../shopify.server";
import { fetchProducts } from "../insights/products.server";

const LOW_STOCK_THRESHOLD = 5;

type IssueTag = "out" | "low" | "noPrice" | "noSku";

interface VariantRow {
  productId: string;
  numericId: string;
  productTitle: string;
  variantTitle: string;
  price: string;
  sku: string;
  inventory: number | null;
  issues: IssueTag[];
}

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const { admin } = await authenticate.admin(request);
  const products = await fetchProducts(admin);

  const rows: VariantRow[] = [];
  for (const p of products) {
    for (const v of p.variants) {
      const issues: IssueTag[] = [];
      if (v.inventory !== null && v.inventory <= 0) issues.push("out");
      else if (v.inventory !== null && v.inventory <= LOW_STOCK_THRESHOLD)
        issues.push("low");
      if (!v.price || Number(v.price) === 0) issues.push("noPrice");
      if (!v.sku.trim()) issues.push("noSku");
      if (issues.length === 0) continue;
      rows.push({
        productId: p.id,
        numericId: p.id.replace("gid://shopify/Product/", ""),
        productTitle: p.title,
        variantTitle: v.title,
        price: v.price,
        sku: v.sku,
        inventory: v.inventory,
        issues,
      });
    }
  }

  // Sort: out of stock first, then low, then pricing/sku gaps.
  const weight = (r: VariantRow) =>
    r.issues.includes("out") ? 0 : r.issues.includes("low") ? 1 : 2;
  rows.sort((a, b) => weight(a) - weight(b));

  const count = (tag: IssueTag) =>
    rows.filter((r) => r.issues.includes(tag)).length;

  return {
    rows,
    stats: {
      outOfStock: count("out"),
      lowStock: count("low"),
      noPrice: count("noPrice"),
      noSku: count("noSku"),
    },
  };
};

function StatTile({
  label,
  value,
  icon,
  bg,
  iconTone,
}: {
  label: string;
  value: number;
  icon: React.FunctionComponent<React.SVGProps<SVGSVGElement>>;
  bg: string;
  iconTone: "critical" | "caution" | "subdued";
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
            {String(value)}
          </Text>
        </BlockStack>
      </InlineStack>
    </Box>
  );
}

const ISSUE_BADGE: Record<
  IssueTag,
  { label: string; tone: "critical" | "warning" | "attention" }
> = {
  out: { label: "Out of stock", tone: "critical" },
  low: { label: "Low stock", tone: "warning" },
  noPrice: { label: "No price", tone: "critical" },
  noSku: { label: "No SKU", tone: "attention" },
};

export default function InventoryInsights() {
  const { rows, stats } = useLoaderData<typeof loader>();

  return (
    <Page>
      <TitleBar title="Inventory & Pricing" />
      <BlockStack gap="400">
        <InlineGrid columns={{ xs: 2, md: 4 }} gap="400">
          <StatTile
            label="Out of stock"
            value={stats.outOfStock}
            icon={AlertTriangleIcon}
            bg="bg-surface-critical"
            iconTone="critical"
          />
          <StatTile
            label="Low stock"
            value={stats.lowStock}
            icon={InventoryIcon}
            bg="bg-surface-caution"
            iconTone="caution"
          />
          <StatTile
            label="No price"
            value={stats.noPrice}
            icon={CashDollarIcon}
            bg="bg-surface-critical"
            iconTone="critical"
          />
          <StatTile
            label="Missing SKU"
            value={stats.noSku}
            icon={BarcodeIcon}
            bg="bg-surface-secondary"
            iconTone="subdued"
          />
        </InlineGrid>

        <Card padding="0">
          <Box padding="400">
            <BlockStack gap="100">
              <Text as="h2" variant="headingMd">
                Variants needing attention
              </Text>
              <Text as="span" variant="bodySm" tone="subdued">
                Out of stock, low stock (≤ {LOW_STOCK_THRESHOLD}), missing price
                or SKU. Click a product to fix it in Product Health.
              </Text>
            </BlockStack>
          </Box>

          {rows.length === 0 ? (
            <Box padding="400">
              <EmptyState
                heading="Everything looks good"
                image="https://cdn.shopify.com/s/files/1/0262/4071/2726/files/emptystate-files.png"
              >
                <p>No stock, price, or SKU issues across your variants.</p>
              </EmptyState>
            </Box>
          ) : (
            <IndexTable
              itemCount={rows.length}
              selectable={false}
              headings={[
                { title: "Product" },
                { title: "Variant" },
                { title: "Price" },
                { title: "Stock" },
                { title: "SKU" },
                { title: "Issues" },
              ]}
            >
              {rows.map((r, index) => (
                <IndexTable.Row
                  id={`${r.productId}-${index}`}
                  key={`${r.productId}-${index}`}
                  position={index}
                >
                  <IndexTable.Cell>
                    <Link to={`/app/health/${r.numericId}`}>
                      <Text as="span" fontWeight="semibold">
                        {r.productTitle}
                      </Text>
                    </Link>
                  </IndexTable.Cell>
                  <IndexTable.Cell>
                    <Text as="span" tone="subdued">
                      {r.variantTitle}
                    </Text>
                  </IndexTable.Cell>
                  <IndexTable.Cell>
                    {!r.price || Number(r.price) === 0 ? (
                      <Text as="span" tone="critical">
                        —
                      </Text>
                    ) : (
                      <Text as="span">{`$${r.price}`}</Text>
                    )}
                  </IndexTable.Cell>
                  <IndexTable.Cell>
                    <Text as="span">
                      {r.inventory === null ? "—" : String(r.inventory)}
                    </Text>
                  </IndexTable.Cell>
                  <IndexTable.Cell>
                    <Text as="span" tone="subdued">
                      {r.sku || "—"}
                    </Text>
                  </IndexTable.Cell>
                  <IndexTable.Cell>
                    <InlineStack gap="100" wrap>
                      {r.issues.map((t) => (
                        <Badge key={t} tone={ISSUE_BADGE[t].tone} size="small">
                          {ISSUE_BADGE[t].label}
                        </Badge>
                      ))}
                    </InlineStack>
                  </IndexTable.Cell>
                </IndexTable.Row>
              ))}
            </IndexTable>
          )}
        </Card>
      </BlockStack>
    </Page>
  );
}
