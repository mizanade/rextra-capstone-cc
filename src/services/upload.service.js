const { Storage } = require('@google-cloud/storage')
const { Err } = require('../middlewares/err.middleware')

module.exports = {
    storageBucket: async (file, oldProfileUrl) => {
        const storage = new Storage({
            projectId: process.env.GCP_PROJECT_ID,
            keyFilename: process.env.GCP_KEYFILE_PATH,
        })

        const bucketName = process.env.GCP_BUCKET_NAME
        const bucket = storage.bucket(bucketName)

        try {
            // delete old image
            const oldFileName = oldProfileUrl
                ? oldProfileUrl.split(`${bucketName}/`)[1]
                : null
            console.log(oldFileName)
            // Delete the old file if it exists
            if (oldFileName) {
                const oldFile = bucket.file(oldFileName)
                try {
                    await oldFile.delete()
                    console.log(`Successfully deleted old file: ${oldFileName}`)
                } catch (error) {
                    // Handle errors (file not found, etc.)
                    console.error(
                        `Failed to delete old file: ${oldFileName}`,
                        error
                    )
                }
            }

            const fileName =
                `profile/${Date.now()}-${file.originalname}`.replace(/\s/g, '')

            const cloudFile = bucket.file(fileName)
            await cloudFile.createWriteStream().end(file.buffer)
            // Get the public URL of the uploaded file
            const publicUrl = `https://storage.googleapis.com/${bucketName}/${fileName}`

            return { fileName, publicUrl }
        } catch (error) {
            throw new Err(500, 'User Not Found')
        }
    },
}
