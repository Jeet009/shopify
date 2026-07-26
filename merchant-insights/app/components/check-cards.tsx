import { useState } from "react";
import { useFetcher } from "@remix-run/react";
import {
  Card,
  Text,
  Badge,
  Button,
  BlockStack,
  InlineStack,
  Box,
  List,
  TextField,
  Thumbnail,
  Link as PolarisLink,
} from "@shopify/polaris";

import { checkStatus, statusTone, type CheckResult } from "../insights/checks";

export function ScoreBadge({ check }: { check: CheckResult }) {
  const status = checkStatus(check);
  if (status === "unscanned") return <Badge>Not scanned</Badge>;
  return <Badge tone={statusTone(status)}>{`${check.score}/100`}</Badge>;
}

function DiffPair({
  label,
  current,
  suggested,
}: {
  label: string;
  current: string;
  suggested: string;
}) {
  return (
    <InlineStack gap="400" align="start" blockAlign="stretch" wrap>
      <div style={{ flex: "1 1 240px", minWidth: 0 }}>
        <BlockStack gap="100">
          <Text as="h4" variant="headingXs" tone="subdued">
            Current {label}
          </Text>
          <Box
            padding="300"
            background="bg-surface-secondary"
            borderWidth="025"
            borderColor="border"
            borderRadius="200"
          >
            <Text as="p" variant="bodyMd" tone="subdued">
              {current || "(empty)"}
            </Text>
          </Box>
        </BlockStack>
      </div>
      <div style={{ flex: "1 1 240px", minWidth: 0 }}>
        <BlockStack gap="100">
          <Text as="h4" variant="headingXs">
            Suggested {label}
          </Text>
          <Box
            padding="300"
            background="bg-surface-success"
            borderWidth="025"
            borderColor="border-success"
            borderRadius="200"
          >
            <Text as="p" variant="bodyMd">
              {suggested || "(none — try regenerating)"}
            </Text>
          </Box>
        </BlockStack>
      </div>
    </InlineStack>
  );
}

/**
 * An AI-powered check (description / SEO): shows issues, a current-vs-suggested
 * diff, Apply, and Regenerate (plain + guided). `actionUrl` lets this post to a
 * different route's action (used when embedded in the list page).
 */
export function AiCheckCard({
  check,
  fields,
  applyIntent,
  regenIntent,
  actionUrl,
}: {
  check: CheckResult;
  fields: { label: string; current: string; suggested: string }[];
  applyIntent: string;
  regenIntent: string;
  actionUrl?: string;
}) {
  const applyFetcher = useFetcher<{ ok: boolean; error?: string }>();
  const regenFetcher = useFetcher();
  const [guidance, setGuidance] = useState("");

  const applying = applyFetcher.state !== "idle";
  const regenerating = regenFetcher.state !== "idle";
  const applied = check.applied || applyFetcher.data?.ok;
  const scanned = check.score !== null;

  const regenerate = (withGuidance: boolean) =>
    regenFetcher.submit(
      { intent: regenIntent, guidance: withGuidance ? guidance : "" },
      { method: "post", action: actionUrl },
    );

  return (
    <Card>
      <BlockStack gap="300">
        <InlineStack align="space-between" blockAlign="center">
          <Text as="h3" variant="headingMd">
            {check.label}
          </Text>
          <ScoreBadge check={check} />
        </InlineStack>

        {!scanned ? (
          <InlineStack gap="200" blockAlign="center">
            <Text as="span" tone="subdued">
              Not scanned yet.
            </Text>
            <Button onClick={() => regenerate(false)} loading={regenerating}>
              Scan now
            </Button>
          </InlineStack>
        ) : (
          <>
            {check.issues.length > 0 && (
              <List type="bullet">
                {check.issues.map((issue, i) => (
                  <List.Item key={i}>{issue}</List.Item>
                ))}
              </List>
            )}

            {fields.map((f) => (
              <DiffPair key={f.label} {...f} />
            ))}

            <TextField
              label="Regenerate with an instruction (optional)"
              labelHidden
              autoComplete="off"
              value={guidance}
              onChange={setGuidance}
              placeholder="e.g. make it punchier, mention free shipping, target beginners"
              disabled={regenerating}
              connectedRight={
                <Button
                  onClick={() => regenerate(true)}
                  loading={regenerating}
                  disabled={!guidance.trim()}
                >
                  Regenerate with prompt
                </Button>
              }
            />

            <InlineStack gap="200" blockAlign="center">
              <applyFetcher.Form method="post" action={actionUrl}>
                <input type="hidden" name="intent" value={applyIntent} />
                <Button
                  submit
                  variant="primary"
                  loading={applying}
                  disabled={Boolean(applied)}
                >
                  {applied ? "Applied ✓" : "Apply to product"}
                </Button>
              </applyFetcher.Form>
              <Button onClick={() => regenerate(false)} loading={regenerating}>
                Regenerate
              </Button>
              {applied && (
                <Text as="span" tone="success" variant="bodySm">
                  Saved to Shopify.
                </Text>
              )}
              {applyFetcher.data &&
                !applyFetcher.data.ok &&
                applyFetcher.data.error && (
                  <Text as="span" tone="critical" variant="bodySm">
                    {applyFetcher.data.error}
                  </Text>
                )}
            </InlineStack>
          </>
        )}
      </BlockStack>
    </Card>
  );
}

/** A rule-based check (images / setup / inventory): issues + admin link. */
export function RuleCheckCard({
  check,
  adminUrl,
}: {
  check: CheckResult;
  adminUrl: string;
}) {
  return (
    <Card>
      <BlockStack gap="300">
        <InlineStack align="space-between" blockAlign="center">
          <Text as="h3" variant="headingMd">
            {check.label}
          </Text>
          <ScoreBadge check={check} />
        </InlineStack>
        {check.issues.length > 0 ? (
          <List type="bullet">
            {check.issues.map((issue, i) => (
              <List.Item key={i}>{issue}</List.Item>
            ))}
          </List>
        ) : (
          <Text as="p" tone="subdued">
            No issues found.
          </Text>
        )}
        <InlineStack>
          <PolarisLink url={adminUrl} target="_blank">
            Fix in Shopify admin
          </PolarisLink>
        </InlineStack>
      </BlockStack>
    </Card>
  );
}

/** Renders the full set of check cards for one product (used inline & on detail). */
/** Card header used by the editable (manual) checks. */
function CardHeader({
  title,
  check,
}: {
  title: string;
  check?: CheckResult;
}) {
  return (
    <InlineStack align="space-between" blockAlign="center">
      <Text as="h3" variant="headingMd">
        {title}
      </Text>
      {check && <ScoreBadge check={check} />}
    </InlineStack>
  );
}

/** Vendor — a single editable text field. */
export function VendorCard({
  currentVendor,
  actionUrl,
}: {
  currentVendor: string;
  actionUrl?: string;
}) {
  const fetcher = useFetcher<{ ok: boolean; error?: string }>();
  const [vendor, setVendor] = useState(currentVendor);
  const saving = fetcher.state !== "idle";
  const saved = fetcher.data?.ok;

  return (
    <Card>
      <BlockStack gap="300">
        <CardHeader title="Vendor / Brand" />
        <fetcher.Form method="post" action={actionUrl}>
          <input type="hidden" name="intent" value="applyVendor" />
          <TextField
            label="Vendor"
            labelHidden
            name="vendor"
            autoComplete="off"
            value={vendor}
            onChange={setVendor}
            placeholder="e.g. Acme Co."
            connectedRight={
              <Button submit loading={saving} disabled={vendor === currentVendor}>
                Save
              </Button>
            }
          />
        </fetcher.Form>
        {saved && (
          <Text as="span" tone="success" variant="bodySm">
            Saved to Shopify.
          </Text>
        )}
        {fetcher.data && !fetcher.data.ok && fetcher.data.error && (
          <Text as="span" tone="critical" variant="bodySm">
            {fetcher.data.error}
          </Text>
        )}
      </BlockStack>
    </Card>
  );
}

interface VariantData {
  id: string;
  title: string;
  price: string;
  sku: string;
  inventory: number | null;
  inventoryItemId: string | null;
}

function VariantRow({
  variant,
  actionUrl,
}: {
  variant: VariantData;
  actionUrl?: string;
}) {
  const fetcher = useFetcher<{ ok: boolean; error?: string }>();
  const [price, setPrice] = useState(variant.price ?? "");
  const [sku, setSku] = useState(variant.sku ?? "");
  const [qty, setQty] = useState(
    variant.inventory === null ? "" : String(variant.inventory),
  );
  const saving = fetcher.state !== "idle";

  return (
    <Box
      padding="300"
      background="bg-surface-secondary"
      borderRadius="200"
      borderWidth="025"
      borderColor="border"
    >
      <BlockStack gap="200">
        <Text as="span" variant="bodySm" fontWeight="medium">
          {variant.title}
        </Text>
        <fetcher.Form method="post" action={actionUrl}>
          <input type="hidden" name="intent" value="applyVariant" />
          <input type="hidden" name="variantId" value={variant.id} />
          <input
            type="hidden"
            name="inventoryItemId"
            value={variant.inventoryItemId ?? ""}
          />
          <InlineStack gap="300" blockAlign="end" wrap>
            <div style={{ width: 120 }}>
              <TextField
                label="Price"
                name="price"
                type="number"
                prefix="$"
                autoComplete="off"
                value={price}
                onChange={setPrice}
              />
            </div>
            <div style={{ width: 160 }}>
              <TextField
                label="SKU"
                name="sku"
                autoComplete="off"
                value={sku}
                onChange={setSku}
              />
            </div>
            <div style={{ width: 120 }}>
              <TextField
                label="Stock"
                name="quantity"
                type="number"
                autoComplete="off"
                value={qty}
                onChange={setQty}
              />
            </div>
            <Button submit loading={saving}>
              Save
            </Button>
          </InlineStack>
        </fetcher.Form>
        {fetcher.data?.ok && (
          <Text as="span" tone="success" variant="bodySm">
            Saved.
          </Text>
        )}
        {fetcher.data && !fetcher.data.ok && fetcher.data.error && (
          <Text as="span" tone="critical" variant="bodySm">
            {fetcher.data.error}
          </Text>
        )}
      </BlockStack>
    </Box>
  );
}

/** Inventory — editable price / SKU / stock per variant. */
export function InventoryCard({
  check,
  variants,
  actionUrl,
}: {
  check: CheckResult;
  variants: VariantData[];
  actionUrl?: string;
}) {
  return (
    <Card>
      <BlockStack gap="300">
        <CardHeader title="Inventory" check={check} />
        {check.issues.length > 0 && (
          <List type="bullet">
            {check.issues.map((issue, i) => (
              <List.Item key={i}>{issue}</List.Item>
            ))}
          </List>
        )}
        {variants.map((v) => (
          <VariantRow key={v.id} variant={v} actionUrl={actionUrl} />
        ))}
      </BlockStack>
    </Card>
  );
}

interface ImageData {
  id: string;
  url: string;
  alt: string;
}

function ImageAltRow({
  image,
  actionUrl,
}: {
  image: ImageData;
  actionUrl?: string;
}) {
  const fetcher = useFetcher<{ ok: boolean; error?: string }>();
  const [alt, setAlt] = useState(image.alt ?? "");
  const saving = fetcher.state !== "idle";

  return (
    <InlineStack gap="300" blockAlign="center" wrap={false}>
      <Thumbnail source={image.url} alt={image.alt} size="small" />
      <div style={{ flex: 1, minWidth: 0 }}>
        <fetcher.Form method="post" action={actionUrl}>
          <input type="hidden" name="intent" value="applyImageAlt" />
          <input type="hidden" name="imageId" value={image.id} />
          <TextField
            label="Alt text"
            labelHidden
            name="alt"
            autoComplete="off"
            value={alt}
            onChange={setAlt}
            placeholder="Describe this image for SEO & accessibility"
            connectedRight={
              <Button submit loading={saving} disabled={alt === image.alt}>
                {fetcher.data?.ok ? "Saved ✓" : "Save"}
              </Button>
            }
          />
        </fetcher.Form>
      </div>
    </InlineStack>
  );
}

/** Images — edit alt text on existing images and upload a new one. */
export function ImagesCard({
  check,
  images,
  actionUrl,
}: {
  check: CheckResult;
  images: ImageData[];
  actionUrl?: string;
}) {
  const uploader = useFetcher<{ ok: boolean; error?: string }>();
  const uploading = uploader.state !== "idle";

  return (
    <Card>
      <BlockStack gap="300">
        <CardHeader title="Images" check={check} />
        {check.issues.length > 0 && (
          <List type="bullet">
            {check.issues.map((issue, i) => (
              <List.Item key={i}>{issue}</List.Item>
            ))}
          </List>
        )}

        {images.map((img) => (
          <ImageAltRow key={img.id} image={img} actionUrl={actionUrl} />
        ))}

        <Box
          padding="300"
          background="bg-surface-secondary"
          borderRadius="200"
          borderWidth="025"
          borderColor="border"
        >
          <BlockStack gap="200">
            <Text as="h4" variant="headingXs">
              Upload a new image
            </Text>
            <uploader.Form
              method="post"
              action={actionUrl}
              encType="multipart/form-data"
            >
              <input type="hidden" name="intent" value="uploadImage" />
              <InlineStack gap="300" blockAlign="center" wrap>
                <input type="file" name="image" accept="image/*" />
                <Button submit variant="primary" loading={uploading}>
                  Upload
                </Button>
              </InlineStack>
            </uploader.Form>
            {uploader.data?.ok && (
              <Text as="span" tone="success" variant="bodySm">
                Image uploaded.
              </Text>
            )}
            {uploader.data && !uploader.data.ok && uploader.data.error && (
              <Text as="span" tone="critical" variant="bodySm">
                {uploader.data.error}
              </Text>
            )}
          </BlockStack>
        </Box>
      </BlockStack>
    </Card>
  );
}

function parseTags(json: string | undefined): string {
  if (!json) return "";
  try {
    const arr = JSON.parse(json);
    return Array.isArray(arr) ? arr.join(", ") : "";
  } catch {
    return "";
  }
}

export function ProductCheckCards({
  checks,
  product,
  adminUrl,
  actionUrl,
}: {
  checks: CheckResult[];
  product: {
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
  };
  adminUrl: string;
  actionUrl?: string;
}) {
  const get = (id: CheckResult["id"]) => checks.find((c) => c.id === id)!;
  const description = get("description");
  const seo = get("seo");
  const setup = get("setup");

  return (
    <BlockStack gap="400">
      <AiCheckCard
        check={description}
        applyIntent="applyDescription"
        regenIntent="regenDescription"
        actionUrl={actionUrl}
        fields={[
          {
            label: "description",
            current: product.currentDescription,
            suggested: description.suggestion ?? "",
          },
        ]}
      />
      <AiCheckCard
        check={seo}
        applyIntent="applySeo"
        regenIntent="regenSeo"
        actionUrl={actionUrl}
        fields={[
          {
            label: "SEO title",
            current: product.seoTitle,
            suggested: seo.suggestion ?? "",
          },
          {
            label: "meta description",
            current: product.seoDescription,
            suggested: seo.suggestionMeta ?? "",
          },
        ]}
      />
      <AiCheckCard
        check={setup}
        applyIntent="applySetup"
        regenIntent="regenSetup"
        actionUrl={actionUrl}
        fields={[
          {
            label: "product type",
            current: product.currentType,
            suggested: setup.suggestion ?? "",
          },
          {
            label: "tags",
            current: product.currentTags.join(", "),
            suggested: parseTags(setup.suggestionMeta),
          },
        ]}
      />
      <VendorCard currentVendor={product.vendor} actionUrl={actionUrl} />
      <ImagesCard
        check={get("images")}
        images={product.images}
        actionUrl={actionUrl}
      />
      <InventoryCard
        check={get("inventory")}
        variants={product.variants}
        actionUrl={actionUrl}
      />
    </BlockStack>
  );
}
