require('dotenv').config()
const express = require('express')
const morgan = require('morgan')
const cors = require('cors')
const routes = require('./routes/index')
const errMiddleware = require('./middlewares/err.middleware')
const app = express()
const dotenv = require('dotenv')

dotenv.config()

app.use(
    morgan(function (tokens, req, res) {
        return [
            tokens.method(req, res),
            tokens.url(req, res),
            tokens.status(req, res),
            tokens.res(req, res, 'content-length'),
            '-',
            tokens['response-time'](req, res),
            'ms',
        ].join(' ')
    })
)
app.use(cors())
app.use(express.json())
app.use(express.urlencoded({ extended: true }))

app.use('/api/v1', routes)
app.use(errMiddleware.handleError)

app.listen(3000, () => {
    console.log(`listening add port 3000`)
})
