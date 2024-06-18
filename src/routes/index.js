const router = require('express').Router()
const authRoute = require('./auth.route')
const activityRoute = require('./activity.route')
const userRoute = require('./user.route')
const mlRoute = require('./ml.route')

router.use(authRoute)
router.use(activityRoute)
router.use(userRoute)
router.use(mlRoute)

module.exports = router
