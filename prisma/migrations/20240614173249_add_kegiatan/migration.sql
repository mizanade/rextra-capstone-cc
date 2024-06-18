-- CreateTable
CREATE TABLE `kegiatan` (
    `id` INTEGER NOT NULL AUTO_INCREMENT,
    `kategori` VARCHAR(255) NOT NULL,
    `sub_kategori` VARCHAR(255) NOT NULL,
    `penyelenggara` VARCHAR(255) NOT NULL,
    `posisi` VARCHAR(255) NOT NULL,
    `deskripsi` VARCHAR(255) NOT NULL,
    `durasi` VARCHAR(255) NOT NULL,
    `tanggal_pendaftaran` VARCHAR(255) NOT NULL,
    `link` VARCHAR(255) NOT NULL,
    `cp` VARCHAR(255) NOT NULL,
    `jenis_kontak` VARCHAR(255) NOT NULL,
    `lokasi` VARCHAR(255) NOT NULL,
    `persyaratan` VARCHAR(255) NOT NULL,
    `skill` VARCHAR(255) NOT NULL,
    `created_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    `updated_at` DATETIME(3) NOT NULL,

    PRIMARY KEY (`id`)
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
