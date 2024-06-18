const bcrypt = require('bcrypt')
const saltRounds = process.env.SALT_ROUNDS

module.exports = {
    hashPassword: async (password) => {
        return await bcrypt.hash(password, +saltRounds)
    },
    comparePassword: async (password, hashedPassword) => {
        return await bcrypt.compare(password, hashedPassword)
    },
}
