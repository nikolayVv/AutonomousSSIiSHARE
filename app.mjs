import dotenv from "dotenv";

import express from "express";
import expressSanitizer from "express-sanitizer";
import path from "path";
import logger from "morgan";
import bodyParser from "body-parser";
import https from "https";
import { fileURLToPath } from 'url';

import("./app_api/models/ssiAgent.mjs");
import ssiAgent from "./app_api/models/ssiAgent.mjs";
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0'
import indexApi from "./app_api/routes/index.mjs";

const app = express();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({path: `${__dirname}/.env`});

app.use(logger('dev'));
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));
app.use(expressSanitizer());
app.use(express.static(path.join(__dirname, 'app_public', 'build')));

app.use("/api", (req, res, next) => {
    res.header("Cache-Control", "no-store");
    res.header("Pragma", "no-cache");
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, PUT, POST, DELETE");
    res.header(
        "Access-Control-Allow-Headers",
        "Origin, X-Requested-With, Content-Type, Accept, Authorization, Do-Not-Sign, application"
    );

    // if (!req.headers["content-type"] || !("application/didcomm")) {
    //     return res.status(400).json({ error: "Missing or incorrect Content-Type header. Expected application/didcomm" });
    // }

    next();
});
app.use("/api", indexApi);

app.get("*", (req, res) => {
   res.sendFile(path.join(__dirname, "app_public", "build", "index.html"));
});

// Large headers
app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ limit: "50mb", extended: true, parameterLimit: 5000000 }));

function normalizePort(val) {
    const port = parseInt(val, 10);

    if (isNaN(port)) {
        return val;
    }

    if (port >= 0) {
        return port
    }

    return false
}


function onError(error) {
    if (error.syscall !== 'listen') {
        throw error;
    }

    const bind = typeof port === 'string'
        ? 'Pipe ' + port
        : 'Port ' + port;

    switch (error.code) {
        case 'EACCES':
            console.error(bind + ' requires elevated privileges');
            process.exit(1);
            break;
        case 'EADDRINUSE':
            console.error(bind + ' is already in use');
            process.exit(1);
            break;
        default:
            throw error;
    }
}


function onListening() {
    const addr = server.address();
    const bind = typeof addr === 'string'
        ? 'pipe ' + addr
        : 'port ' + addr.port;
    const agent = process.env.AGENT || "Test";

    console.log(`Running agent "${agent}" on ` + bind);
}

// Get port from environment
const port = normalizePort(process.env.PORT || '3000');
app.set("port", port);

const options = {
    key: process.env.PRIVATE_KEY.replace(/\\n/g, '\n'),
    cert: process.env.CERT.replace(/\\n/g, '\n')
}

// Create HTTP server
const server = https.createServer(options, app);

// Listen on the provided port
server.listen(port);
server.on("error", onError);
server.on("listening", onListening);