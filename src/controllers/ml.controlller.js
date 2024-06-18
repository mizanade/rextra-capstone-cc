const prisma = require('../config/prisma.config')
const tf = require('@tensorflow/tfjs')
const axios = require('axios')
const FormData = require('form-data')

module.exports = {
    getRecomendations: async (req, res, next) => {
        try {
            const ML_BASED_URL = process.env.ML_BASED_URL
            const { id } = req.loggedUser

            const foundUser = await prisma.user.findFirst({ where: { id } })
            if (!foundUser) throw new Err(404, 'User Not Found')

            const {
                penyelenggara,
                preference,
                user_preference,
                skills,
                past_activities,
                durasi,
            } = req.body
            const formData = new FormData()
            formData.append('penyelenggara', penyelenggara)
            formData.append('preference', preference)
            formData.append('user_preference', user_preference)
            formData.append('skills', skills)
            formData.append('past_activities', past_activities)
            formData.append('durasi', durasi)

            if (req.file) {
                formData.append('cv', req.file.buffer, req.file.originalname)
            }

            const response = await axios.post(
                `${ML_BASED_URL}/recommend`,
                formData,
                {
                    headers: {
                        ...formData.getHeaders(),
                    },
                }
            )

            console.log(response)
            const data = response.data

            const savedRecommendations = await Promise.all(
                data.data.map(async (item) => {
                    const existingRecommendation =
                        await prisma.recomendation.findFirst({
                            where: {
                                userId: parseInt(id, 10),
                                kegiatanId: item.id,
                            },
                        })

                    // Save the recommendation only if it doesn't already exist
                    if (!existingRecommendation) {
                        return await prisma.recomendation.create({
                            data: {
                                userId: parseInt(id, 10),
                                kegiatanId: item.id,
                                rate: item.similarity_score,
                            },
                        })
                    }
                    // Return null if recommendation already exists
                    return null
                })
            )

            const detailedRecommendations = await Promise.all(
                data.data.map(async (rec) => {
                    return await prisma.recomendation.findUnique({
                        where: {
                            kegiatanId: rec.id,
                            userId: id,
                        },
                        select: {
                            kegiatan: {
                                select: {
                                    id: true,
                                    kategori: true,
                                    subKategori: true,
                                    posisi: true,
                                },
                            },
                        },
                    })
                })
            )
            console.log(detailedRecommendations)

            return res.status(200).json({
                success: true,
                statusCode: 200,
                message: 'Recomendations retrieved successfully.',
                data: data.data,
            })
        } catch (error) {
            console.log(error)
            next(error)
        }
    },

    getAllRecomendations: async (req, res, next) => {
        try {
            const { id } = req.loggedUser

            const foundUser = await prisma.user.findFirst({ where: { id } })
            if (!foundUser) throw new Err(404, 'User Not Found')

            const userRecommendations = await prisma.recomendation.findMany({
                where: {
                    userId: id,
                },
                select: {
                    kegiatan: {
                        select: {
                            id: true,
                            kategori: true,
                            subKategori: true,
                            posisi: true,
                        },
                    },
                },
            })

            return res.status(200).json({
                success: true,
                statusCode: 200,
                message: 'User data retrieved successfully.',
                data: userRecommendations,
            })
        } catch (error) {
            next(error)
        }
    },
}
