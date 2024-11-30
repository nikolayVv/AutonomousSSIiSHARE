import JSONModel from "../models/model.mjs";
const jsonFile = new JSONModel(`../data/agent${process.env.PORT}.json`);

import * as tokenService from "../services/tokenService.mjs";
import * as certificateService from "../services/certificateService.mjs";

const Capabilities = {
    async getCapabilities(req, res) {
        // Load the capabilities
        try {
            const result = await jsonFile.load();
            if (result.error) {
                return res.status(500).json({ message: result.error });
            }
            const capabilities = result.data.capabilities;

            const headers = await req.headers;
            if (!headers["do-not-sign"]) {
                const x5c = certificateService.convertFromCertToX5C(process.env.CERT);
                const capabilities_token = tokenService.generateCapabilitiesToken(process.env.EORI, x5c, capabilities, process.env.PRIVATE_KEY);

                res.status(200).json({ capabilities_token: capabilities_token });
            } else {
                res.status(200).json({ capabilities: capabilities });
            }
        } catch (e) {
            res.status(500).json({ message: "Internal Server Error." });
        }
    }
}

export default Capabilities;