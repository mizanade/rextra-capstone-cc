const { PrismaClient } = require('@prisma/client')
const prisma = new PrismaClient()
const kegiatanData = require('../public/data/Dataset-new.json')

async function main() {
    await prisma.kegiatan.createMany({
        data: kegiatanData.map((item) => {
            const {
                id,
                kategori,
                subKategori,
                penyelenggara,
                posisi = 'Default Posisi',
                deskripsi,
                durasi,
                tanggalPendaftaran,
                link,
                cp,
                jenisKontak,
                lokasi,
                persyaratan,
                skill,
            } = item

            return {
                id,
                kategori,
                subKategori,
                penyelenggara,
                posisi,
                deskripsi,
                durasi,
                tanggalPendaftaran,
                link,
                cp: String(cp),
                jenisKontak,
                lokasi,
                persyaratan,
                skill,
            }
        }),
    })
}

main()
    .then(async () => {
        await prisma.$disconnect()
    })
    .catch(async (e) => {
        console.error(e)
        await prisma.$disconnect()
        process.exit(1)
    })
