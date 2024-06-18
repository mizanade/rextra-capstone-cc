const prisma = require('../config/prisma.config')
const { Err } = require('../middlewares/err.middleware')
const bcryptService = require('../services/bcrypt.service')
const jwtServices = require('../services/jwt.services')

module.exports = {
    userLogin: async (req, res, next) => {
        try {
            const { email, password } = req.body

            const foundUser = await prisma.user.findFirst({ where: { email } })
            if (!foundUser) throw new Err(404, 'Email is not found')

            const passwordValid = await bcryptService.comparePassword(
                password,
                foundUser.password
            )
            if (!passwordValid) throw new Err(400, 'Password is invalid.')

            const token = jwtServices.signToken(foundUser.id)

            return res.status(200).json({
                success: true,
                statusCode: 200,
                message: 'User login successfully.',
                token,
            })
        } catch (error) {
            next(error)
        }
    },
    userRegister: async (req, res, next) => {
        try {
            const { email, name, password, confirmPassword } = req.body

            const foundEmail = await prisma.user.findFirst({ where: { email } })
            if (foundEmail) throw new Err(409, 'Email is already exist.')

            let payload = { email, name, profileUrl: null }
            if (password === confirmPassword) {
                const hashPassword =
                    await bcryptService.hashPassword(confirmPassword)
                payload = { ...payload, password: hashPassword }
            } else {
                throw new Err(403, 'Password is not match.')
            }

            await prisma.user.create({ data: payload })

            return res.status(201).json({
                success: true,
                statusCode: 201,
                message: 'User account created successfully.',
            })
        } catch (error) {
            next(error)
        }
    },
}
