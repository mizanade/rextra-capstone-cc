const mlControlller = require('../controllers/ml.controlller')
const authMiddleware = require('../middlewares/auth.middleware')
const uploadMiddleware = require('../middlewares/upload.middleware')

const router = require('express').Router()

router.post(
    '/recommendations',
    authMiddleware.authentication,
    uploadMiddleware.ml.single('cv'),
    mlControlller.getRecomendations
)
router.get(
    '/recommendations',
    authMiddleware.authentication,
    mlControlller.getAllRecomendations
)

module.exports = router
