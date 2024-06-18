module.exports = {
    validate: (schema) => {
        return function (req, _, next) {
            const { error } = schema.validate(req.body)
            if (error)
                throw new ErrHandler(409, error.message.replace(/"/g, ''))

            next()
        }
    },
}
