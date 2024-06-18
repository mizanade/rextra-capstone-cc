const authController = require('../controllers/auth.controller')
const validateMiddleware = require('../middlewares/validate.middleware')
const { authSchema } = require('../services/joi.service')
const router = require('express').Router()

router.post(
    '/auth/register',
    validateMiddleware.validate(authSchema.register),
    authController.userRegister
)
router.post(
    '/auth/login',
    validateMiddleware.validate(authSchema.login),
    authController.userLogin
)

module.exports = router
