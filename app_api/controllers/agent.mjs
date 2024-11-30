import * as tokenService from "../services/tokenService.mjs";
import * as certificateService from "../services/certificateService.mjs";
import * as ssiService from "../services/ssiService.mjs";
import JSONModel from "../models/model.mjs";
import ssiAgent from "../models/ssiAgent.mjs";
const jsonFile = new JSONModel(`../data/agent${process.env.PORT}.json`);

const Agent = {
    async getInformation(req, res) {
        res.status(200).json({ agent: {
            name: process.env.AGENT,
            eori: process.env.EORI,
            role: process.env.ROLE
        }});
    },

    async generateClientAssertion(req, res) {
        const reqBody = await req.body;
        const eori_receiver = reqBody.eori_receiver
        if(!eori_receiver) {
            return res.status(400).json({ message: 'Request has no eori_receiver.' });
        }
        const x5c = certificateService.convertFromCertToX5C(process.env.CERT);

        const client_assertion = tokenService.generateClientAssertion(process.env.EORI, x5c, eori_receiver, process.env.PRIVATE_KEY);

        res.status(200).json({
            client_assertion: client_assertion,
            expires_in: 30,
            token_type: 'Identification'
        });
    },

    async generateClientAssertionSSI(req, res) {
        const reqBody = await req.body;
        const eori_receiver = reqBody.eori_receiver
        if(!eori_receiver) {
            return res.status(400).json({ message: 'Request has no eori_receiver.' });
        }

        // Check if connection already exists

        const identifier = await ssiService.getOrGenerateIdentifier(eori_receiver);
        const x5c = certificateService.convertFromCertToX5C(process.env.CERT);
        const client_assertion = tokenService.generateClientAssertionSSI(process.env.EORI, x5c, eori_receiver, process.env.PRIVATE_KEY, identifier.did);

        res.status(200).json({
            client_assertion: client_assertion,
            expires_in: 30,
            token_type: 'Identification'
        });
    },

    async generateToken(req, res) {
        const reqBody = await req.body;
        const client_id = reqBody.client_id
        if(!client_id) {
            return res.status(400).json({ message: 'Request has no client_id.' });
        }

        const x5c = certificateService.convertFromCertToX5C(process.env.CERT);
        const access_token = tokenService.generateAccessToken(process.env.EORI, x5c, client_id, process.env.PRIVATE_KEY);

        res.status(200).json({
            access_token: access_token,
            expires_in: 3600,
            token_type: 'Bearer'
        });
    },

    async getIdentifiers(req, res) {
        const identifiers = await ssiService.getIdentifiers();

        res.status(200).json({ count: identifiers.length, data: identifiers });
    },

    async getConnections(req, res) {
        const result = await jsonFile.load();
        if (result.error) {
            return res.status(500).json({ message: result.error });
        }

        const connections = result.data.did_connections || [];
        res.status(200).json({ count: connections.length, data: connections });
    }
}

export default Agent;