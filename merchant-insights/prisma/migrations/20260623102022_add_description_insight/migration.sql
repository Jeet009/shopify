-- CreateTable
CREATE TABLE "DescriptionInsight" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "shop" TEXT NOT NULL,
    "productId" TEXT NOT NULL,
    "score" INTEGER NOT NULL,
    "breakdown" TEXT NOT NULL,
    "issues" TEXT NOT NULL,
    "suggestedRewrite" TEXT NOT NULL,
    "appliedAt" DATETIME,
    "scoredAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- CreateIndex
CREATE UNIQUE INDEX "DescriptionInsight_shop_productId_key" ON "DescriptionInsight"("shop", "productId");
