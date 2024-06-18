const activityController = require('../controllers/activity.controller')
const authMiddleware = require('../middlewares/auth.middleware')
const router = require('express').Router()

router.get(
    '/activities/categories',
    authMiddleware.authentication,
    activityController.getActivityCategories
)
router.get(
    '/activities',
    authMiddleware.authentication,
    activityController.getAllActivity
)
router.get(
    '/activities/:idActivity',
    authMiddleware.authentication,
    activityController.getActivityById
)

module.exports = router
