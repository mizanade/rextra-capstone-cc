const userController = require('../controllers/user.controller')
const authMiddleware = require('../middlewares/auth.middleware')
const uploadMiddleware = require('../middlewares/upload.middleware')

const router = require('express').Router()

router.get(
    '/users',
    authMiddleware.authentication,
    userController.getUserProfile
)
router.put(
    '/users',
    authMiddleware.authentication,
    userController.editUserProfile
)
router.post(
    '/users/profile-image',
    authMiddleware.authentication,
    uploadMiddleware.profileImage.single('image'),
    userController.updateProfilePhoto
)

module.exports = router
