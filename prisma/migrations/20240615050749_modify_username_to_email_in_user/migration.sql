/*
  Warnings:

  - You are about to drop the column `username` on the `users` table. All the data in the column will be lost.
  - Added the required column `email` to the `users` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE `users` DROP COLUMN `username`,
    ADD COLUMN `email` VARCHAR(255) NOT NULL AFTER `name`;

-- AlterTable
ALTER TABLE `users` MODIFY `profile_url` VARCHAR(255) NULL;
