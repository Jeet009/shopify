/*
  Warnings:

  - You are about to drop the `DescriptionInsight` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "DescriptionInsight";
PRAGMA foreign_keys=on;

-- CreateTable
CREATE TABLE "AiCheck" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "shop" TEXT NOT NULL,
    "productId" TEXT NOT NULL,
    "checkType" TEXT NOT NULL,
    "score" INTEGER NOT NULL,
    "issues" TEXT NOT NULL,
    "suggestion" TEXT NOT NULL,
    "suggestionMeta" TEXT,
    "appliedAt" DATETIME,
    "scoredAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- CreateIndex
CREATE UNIQUE INDEX "AiCheck_shop_productId_checkType_key" ON "AiCheck"("shop", "productId", "checkType");
