/*
  Warnings:

  - You are about to alter the column `rate` on the `recomendations` table. The data in that column could be lost. The data in that column will be cast from `Decimal(15,0)` to `Decimal(15,5)`.

*/
-- AlterTable
ALTER TABLE `recomendations` MODIFY `rate` DECIMAL(15, 5) NOT NULL;
