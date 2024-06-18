const prisma = require('../config/prisma.config')
const jwtServices = require('../services/jwt.services')
const { Err } = require('./err.middleware')

module.exports = {
    authentication: async (req, res, next) => {
        try {
            const authToken = req.headers.authorization
            const token = authToken && authToken.split(' ')[1]

            if (!token) throw new Err(400, 'Unauthorized user.')

            const decoded = jwtServices.verifyToken(token)
            const foundUser = await prisma.user.findFirst({
                where: { id: decoded.userId },
            })
            req.loggedUser = {
                id: foundUser.id,
            }

            next()
        } catch (error) {
            console.log(error)
            next(error)
        }
    },
}
