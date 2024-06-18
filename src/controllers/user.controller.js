const prisma = require('../config/prisma.config')
const { Err } = require('../middlewares/err.middleware')
const uploadService = require('../services/upload.service')

module.exports = {
    getUserProfile: async (req, res, next) => {
        try {
            const { id } = req.loggedUser

            const foundUser = await prisma.user.findFirst({
                where: { id },
                select: {
                    id: true,
                    name: true,
                    username: true,
                    email: true,
                    profileUrl: true,
                    createdAt: true,
                    updatedAt: true,
                },
            })
            if (!foundUser) throw new Err(404, 'User Not Found')

            return res.status(200).json({
                success: true,
                statusCode: 200,
                message: 'User data retrieved successfully.',
                data: foundUser,
            })
        } catch (error) {
            next(error)
        }
    },
    editUserProfile: async (req, res, next) => {
        try {
            const { id } = req.loggedUser
            const { name, email, username } = req.body

            const foundUser = await prisma.user.findFirst({ where: { id } })
            if (!foundUser) throw new Err(404, 'User Not Found')

            const foundEmail = await prisma.user.findFirst({ where: { email } })
            if (foundEmail && foundUser.email != email)
                throw new Err(404, 'Email Has Been Used')
            console.log('tes')

            const foundUsername = await prisma.user.findFirst({
                where: { username },
            })
            if (foundUsername && foundUser.username != username)
                throw new Err(404, 'Username Has Been Used')
            console.log('tes2')

            const updatedUser = await prisma.user.update({
                where: { id },
                data: {
                    name,
                    email: email.toLowerCase(),
                    username: username.toLowerCase(),
                },
            })

            return res.status(200).json({
                success: true,
                statusCode: 200,
                message: 'User data updated successfully.',
            })
        } catch (error) {
            next(error)
        }
    },

    updateProfilePhoto: async (req, res, next) => {
        try {
            const { id } = req.loggedUser

            const file = req.file
            if (!file) throw new Err(400, 'Please Insert Image')

            const foundUser = await prisma.user.findFirst({ where: { id } })
            if (!foundUser) throw new Err(404, 'User Not Found')

            const { publicUrl } = await uploadService.storageBucket(
                file,
                foundUser.profileUrl
            )

            const updatedImage = await prisma.user.update({
                where: { id },
                data: {
                    profileUrl: publicUrl,
                },
            })

            return res.status(200).json({
                success: true,
                statusCode: 200,
                message: 'User data retrieved successfully.',
                data: updatedImage,
            })
        } catch (error) {
            next(error)
        }
    },
}
