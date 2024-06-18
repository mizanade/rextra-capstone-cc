const jwt = require('jsonwebtoken')
const privateKey = process.env.JWT_KEY

module.exports = {
    signToken: (userId) => {
        return jwt.sign({ userId: userId }, privateKey, { expiresIn: '1h' })
    },
    verifyToken: (token) => {
        return jwt.verify(token, privateKey)
    },
}
