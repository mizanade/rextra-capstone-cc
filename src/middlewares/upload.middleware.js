const { Storage } = require('@google-cloud/storage')
const multer = require('multer')

module.exports = {
    profileImage: multer({
        storage: multer.memoryStorage(), // Store the file in memory before uploading to the cloud storage
    }),
    ml: multer({
        storage: multer.memoryStorage(), // Store the file in memory before uploading to the cloud storage
    }),
}
