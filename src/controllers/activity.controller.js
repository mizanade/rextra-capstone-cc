const { equal } = require('joi')
const prisma = require('../config/prisma.config')

module.exports = {
    getActivityCategories: async (req, res, next) => {
        try {
            const data = await prisma.kegiatan.findMany({
                distinct: 'subKategori',
                select: { subKategori: true },
            })

            return res.status(200).json({
                success: true,
                statusCode: 200,
                message: 'Activity category data retrieved successfully.',
                data,
            })
        } catch (error) {
            next(error)
        }
    },
    getAllActivity: async (req, res, next) => {
        const { tipe, jenis, lokasi, nama } = req.query
        try {
            let filter = {}
            if (tipe || jenis || lokasi)
                filter = {
                    AND: [
                        {
                            kategori: {
                                equals: tipe,
                            },
                        },
                        {
                            subKategori: {
                                equals: jenis,
                            },
                        },
                        {
                            lokasi: {
                                equals: lokasi,
                            },
                        },
                    ],
                }

            if (nama)
                filter = {
                    posisi: {
                        contains: nama,
                    },
                }
            const data = await prisma.kegiatan.findMany({
                where: {
                    ...filter,
                },
                select: { id: true, kategori: true, posisi: true },
            })

            return res.status(200).json({
                success: true,
                statusCode: 200,
                message: 'Activity data retrieved successfully.',
                data,
            })
        } catch (error) {
            next(error)
        }
    },
    getActivityById: async (req, res, next) => {
        const { idActivity } = req.query
        try {
            const data = await prisma.kegiatan.findFirst({
                where: { id: idActivity },
            })
            if (!data) throw new Err(404, 'Activity data is not found.')

            if (data.persyaratan) {
                data.persyaratan = data.persyaratan
                    .split('\n')
                    .map((item) => item.trim())
            }

            if (data.skill) {
                data.skill = data.skill.split('\n').map((item) => item.trim())
            }
            return res.status(200).json({
                success: true,
                statusCode: 200,
                message: 'Activity data retrieved successfully.',
                data,
            })
        } catch (error) {
            next(error)
        }
    },
}
