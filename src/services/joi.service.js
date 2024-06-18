const Joi = require('joi')

module.exports = {
    authSchema: {
        login: Joi.object({
            email: Joi.string().email().required(),
            password: Joi.string().required(),
        }),
        register: Joi.object({
            email: Joi.string().email().required(),
            name: Joi.string().required(),
            password: Joi.string().required(),
            confirmPassword: Joi.string().required(),
        }),
    },
}
