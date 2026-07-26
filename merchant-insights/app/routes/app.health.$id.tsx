import type { ActionFunctionArgs, LoaderFunctionArgs } from "@remix-run/node";
import { useFetcher, useLoaderData } from "@remix-run/react";
import { Page, Badge, Banner, BlockStack } from "@shopify/polaris";
import { TitleBar } from "@shopify/app-bridge-react";

import { authenticate } from "../shopify.server";
import {
  fetchProduct,
  fetchPrimaryLocationId,
  ruleChecks,
  toDescriptionHtml,
} from "../insights/products.server";
import {
  AI_CHECK_TYPES,
  aiCheckToResult,
  getAiCheck,
  markApplied,
  runDescriptionScan,
  runSeoScan,
  runSetupScan,
  scanProduct,
  unscannedAiCheck,
} from "../insights/scan.server";
import {
  CHECK_ORDER,
  overallHealth,
  statusTone,
  type CheckResult,
  type CheckId,
} from "../insights/checks";
import { ProductCheckCards } from "../components/check-cards";
import prisma from "../db.server";

function gidFromParam(id: string): string {
  return `gid://shopify/Product/${id}`;
}

export const loader = async ({ request, params }: LoaderFunctionArgs) => {
  const { admin, session } = await authenticate.admin(request);
  const shop = session.shop;
  const gid = gidFromParam(params.id!);

  const product = await fetchProduct(admin, gid);
  if (!product) {
    throw new Response("Product not found", { status: 404 });
  }

  const aiRows = await prisma.aiCheck.findMany({
    where: { shop, productId: gid },
  });
  const aiById = new Map(aiRows.map((r) => [r.checkType, r]));
  const aiChecks = AI_CHECK_TYPES.map((t) => {
    const row = aiById.get(t);
    return row ? aiCheckToResult(row) : unscannedAiCheck(t);
  });

  const byId = new Map<CheckId, CheckResult>();
  [...aiChecks, ...ruleChecks(product)].forEach((c) => byId.set(c.id, c));
  const checks = CHECK_ORDER.map((id) => byId.get(id)!).filter(Boolean);

  return {
    apiKeyMissing: !process.env.SARVAM_API_KEY,
    numericId: params.id!,
    product: {
      title: product.title,
      currentDescription: product.description,
      seoTitle: product.seoTitle,
      seoDescription: product.seoDescription,
      currentType: product.productType,
      currentTags: product.tags,
      vendor: product.vendor,
      images: product.images,
      variants: product.variants,
    },
    checks,
    overall: overallHealth(checks),
  };
};

const PRODUCT_UPDATE_MUTATION = `#graphql
  mutation UpdateProduct($product: ProductUpdateInput!) {
    productUpdate(product: $product) {
      product { id }
      userErrors { field message }
    }
  }`;

const VARIANTS_BULK_UPDATE = `#graphql
  mutation BulkUpdateVariants($productId: ID!, $variants: [ProductVariantsBulkInput!]!) {
    productVariantsBulkUpdate(productId: $productId, variants: $variants) {
      productVariants { id }
      userErrors { field message }
    }
  }`;

const INVENTORY_SET = `#graphql
  mutation SetInventory($input: InventorySetQuantitiesInput!) {
    inventorySetQuantities(input: $input) {
      userErrors { field message }
    }
  }`;

const FILE_UPDATE = `#graphql
  mutation UpdateFileAlt($files: [FileUpdateInput!]!) {
    fileUpdate(files: $files) {
      userErrors { field message }
    }
  }`;

const STAGED_UPLOADS_CREATE = `#graphql
  mutation StagedUploads($input: [StagedUploadInput!]!) {
    stagedUploadsCreate(input: $input) {
      stagedTargets { url resourceUrl parameters { name value } }
      userErrors { field message }
    }
  }`;

const PRODUCT_CREATE_MEDIA = `#graphql
  mutation CreateMedia($productId: ID!, $media: [CreateMediaInput!]!) {
    productCreateMedia(productId: $productId, media: $media) {
      media { id }
      mediaUserErrors { field message }
    }
  }`;

function errMsg(errs: any[]): string {
  return errs.map((e: any) => e.message).join("; ");
}

export const action = async ({ request, params }: ActionFunctionArgs) => {
  const { admin, session } = await authenticate.admin(request);
  const shop = session.shop;
  const gid = gidFromParam(params.id!);
  const form = await request.formData();
  const intent = String(form.get("intent"));
  const guidance = String(form.get("guidance") ?? "");

  // Apply actions only need the cached suggestion + the product id.
  if (intent === "applyDescription") {
    const row = await getAiCheck(shop, gid, "description");
    if (!row?.suggestion) return { ok: false, error: "Nothing to apply." };
    const res = await admin.graphql(PRODUCT_UPDATE_MUTATION, {
      variables: {
        product: { id: gid, descriptionHtml: toDescriptionHtml(row.suggestion) },
      },
    });
    const j = await res.json();
    const errs = j.data?.productUpdate?.userErrors ?? [];
    if (errs.length)
      return { ok: false, error: errs.map((e: any) => e.message).join("; ") };
    await markApplied(shop, gid, "description");
    return { ok: true };
  }

  if (intent === "applySeo") {
    const row = await getAiCheck(shop, gid, "seo");
    if (!row?.suggestion) return { ok: false, error: "Nothing to apply." };
    const res = await admin.graphql(PRODUCT_UPDATE_MUTATION, {
      variables: {
        product: {
          id: gid,
          seo: { title: row.suggestion, description: row.suggestionMeta ?? "" },
        },
      },
    });
    const j = await res.json();
    const errs = j.data?.productUpdate?.userErrors ?? [];
    if (errs.length)
      return { ok: false, error: errs.map((e: any) => e.message).join("; ") };
    await markApplied(shop, gid, "seo");
    return { ok: true };
  }

  if (intent === "applySetup") {
    const row = await getAiCheck(shop, gid, "setup");
    if (!row?.suggestion) return { ok: false, error: "Nothing to apply." };
    let tags: string[] = [];
    try {
      const parsed = JSON.parse(row.suggestionMeta ?? "[]");
      if (Array.isArray(parsed)) tags = parsed.map(String);
    } catch {
      tags = [];
    }
    const res = await admin.graphql(PRODUCT_UPDATE_MUTATION, {
      variables: {
        product: { id: gid, productType: row.suggestion, tags },
      },
    });
    const j = await res.json();
    const errs = j.data?.productUpdate?.userErrors ?? [];
    if (errs.length)
      return { ok: false, error: errs.map((e: any) => e.message).join("; ") };
    await markApplied(shop, gid, "setup");
    return { ok: true };
  }

  // ---- Manual edits (no AI) ----
  if (intent === "applyVendor") {
    const vendor = String(form.get("vendor") ?? "").trim();
    const res = await admin.graphql(PRODUCT_UPDATE_MUTATION, {
      variables: { product: { id: gid, vendor } },
    });
    const j = await res.json();
    const errs = j.data?.productUpdate?.userErrors ?? [];
    if (errs.length) return { ok: false, error: errMsg(errs) };
    return { ok: true };
  }

  if (intent === "applyVariant") {
    const variantId = String(form.get("variantId"));
    const inventoryItemId = String(form.get("inventoryItemId") ?? "");
    const price = String(form.get("price") ?? "").trim();
    const sku = String(form.get("sku") ?? "").trim();
    const quantityRaw = String(form.get("quantity") ?? "").trim();

    // Price + SKU via bulk update.
    const variantInput: any = { id: variantId };
    if (price) variantInput.price = price;
    variantInput.inventoryItem = { sku };
    const res = await admin.graphql(VARIANTS_BULK_UPDATE, {
      variables: { productId: gid, variants: [variantInput] },
    });
    const j = await res.json();
    const errs = j.data?.productVariantsBulkUpdate?.userErrors ?? [];
    if (errs.length) return { ok: false, error: errMsg(errs) };

    // Stock quantity via inventory API (needs a location).
    if (quantityRaw !== "" && inventoryItemId) {
      const quantity = Number(quantityRaw);
      if (Number.isFinite(quantity)) {
        const locationId = await fetchPrimaryLocationId(admin);
        if (!locationId) return { ok: false, error: "No location found for inventory." };
        const invRes = await admin.graphql(INVENTORY_SET, {
          variables: {
            input: {
              name: "available",
              reason: "correction",
              ignoreCompareQuantity: true,
              quantities: [{ inventoryItemId, locationId, quantity }],
            },
          },
        });
        const invJson = await invRes.json();
        const invErrs = invJson.data?.inventorySetQuantities?.userErrors ?? [];
        if (invErrs.length) return { ok: false, error: errMsg(invErrs) };
      }
    }
    return { ok: true };
  }

  if (intent === "applyImageAlt") {
    const imageId = String(form.get("imageId"));
    const alt = String(form.get("alt") ?? "");
    const res = await admin.graphql(FILE_UPDATE, {
      variables: { files: [{ id: imageId, alt }] },
    });
    const j = await res.json();
    const errs = j.data?.fileUpdate?.userErrors ?? [];
    if (errs.length) return { ok: false, error: errMsg(errs) };
    return { ok: true };
  }

  if (intent === "uploadImage") {
    const file = form.get("image");
    if (!(file instanceof File) || file.size === 0) {
      return { ok: false, error: "No image selected." };
    }
    // 1) Reserve a staged upload target.
    const stagedRes = await admin.graphql(STAGED_UPLOADS_CREATE, {
      variables: {
        input: [
          {
            filename: file.name,
            mimeType: file.type || "image/jpeg",
            resource: "PRODUCT_IMAGE",
            httpMethod: "POST",
          },
        ],
      },
    });
    const stagedJson = await stagedRes.json();
    const stagedErrs = stagedJson.data?.stagedUploadsCreate?.userErrors ?? [];
    if (stagedErrs.length) return { ok: false, error: errMsg(stagedErrs) };
    const target = stagedJson.data?.stagedUploadsCreate?.stagedTargets?.[0];
    if (!target) return { ok: false, error: "Could not start upload." };

    // 2) POST the file bytes to the staged target.
    const uploadForm = new FormData();
    for (const param of target.parameters as { name: string; value: string }[]) {
      uploadForm.append(param.name, param.value);
    }
    uploadForm.append("file", file, file.name);
    const uploadResp = await fetch(target.url, { method: "POST", body: uploadForm });
    if (!uploadResp.ok) {
      return { ok: false, error: `Upload failed (${uploadResp.status}).` };
    }

    // 3) Attach the uploaded file to the product.
    const mediaRes = await admin.graphql(PRODUCT_CREATE_MEDIA, {
      variables: {
        productId: gid,
        media: [
          {
            originalSource: target.resourceUrl,
            mediaContentType: "IMAGE",
            alt: String(form.get("alt") ?? ""),
          },
        ],
      },
    });
    const mediaJson = await mediaRes.json();
    const mediaErrs = mediaJson.data?.productCreateMedia?.mediaUserErrors ?? [];
    if (mediaErrs.length) return { ok: false, error: errMsg(mediaErrs) };
    return { ok: true };
  }

  // Scan / regenerate need fresh product data.
  if (!process.env.SARVAM_API_KEY) {
    return { ok: false, error: "SARVAM_API_KEY is not set." };
  }
  const product = await fetchProduct(admin, gid);
  if (!product) return { ok: false, error: "Product not found." };

  if (intent === "scan") {
    await scanProduct(shop, product);
    return { ok: true };
  }
  if (intent === "regenDescription") {
    await runDescriptionScan(shop, product, guidance);
    return { ok: true };
  }
  if (intent === "regenSeo") {
    await runSeoScan(shop, product, guidance);
    return { ok: true };
  }
  if (intent === "regenSetup") {
    await runSetupScan(shop, product, guidance);
    return { ok: true };
  }
  return { ok: false, error: "Unknown action." };
};

export default function ProductHealthDetail() {
  const { product, checks, overall, numericId, apiKeyMissing } =
    useLoaderData<typeof loader>();
  const scan = useFetcher();
  const scanning = scan.state !== "idle";

  return (
    <Page
      backAction={{ content: "Product Health", url: "/app/health" }}
      title={product.title}
      titleMetadata={
        overall !== null ? (
          <Badge
            tone={statusTone(
              overall >= 80 ? "good" : overall >= 50 ? "warn" : "bad",
            )}
          >
            {`Health ${overall}/100`}
          </Badge>
        ) : undefined
      }
      primaryAction={{
        content: "Re-scan all",
        loading: scanning,
        disabled: apiKeyMissing,
        onAction: () => scan.submit({ intent: "scan" }, { method: "post" }),
      }}
    >
      <TitleBar title={product.title} />
      <BlockStack gap="400">
        {apiKeyMissing && (
          <Banner tone="warning" title="Sarvam API key not set">
            <p>Add SARVAM_API_KEY to .env and restart to run AI checks.</p>
          </Banner>
        )}
        <ProductCheckCards
          checks={checks}
          product={product}
          adminUrl={`shopify:admin/products/${numericId}`}
        />
      </BlockStack>
    </Page>
  );
}
