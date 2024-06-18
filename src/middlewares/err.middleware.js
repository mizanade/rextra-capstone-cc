class Err extends Error {
    statusCode = 0
    constructor(statusCode, message) {
        super(message)
        this.statusCode = statusCode
    }
}

module.exports = {
    handleError: (err, _req, res, _) => {
        if (err instanceof Err) {
            res.status(err.statusCode).json({
                success: false,
                statusCode: err.statusCode,
                message: err.message,
            })
        } else {
            res.status(500).json({
                success: false,
                statusCode: 500,
                message: err.message,
            })
        }
    },
    Err,
}
