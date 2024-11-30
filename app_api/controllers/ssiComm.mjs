import * as ssiService from "../services/ssiService.mjs";
import ssiAgent from "../models/ssiAgent.mjs";

const keyMapping = {
    Secp256k1: 'EcdsaSecp256k1VerificationKey2019',
    Ed25519: 'Ed25519VerificationKey2018',
    X25519: 'X25519KeyAgreementKey2019',
}

const ssiComm = {
    async getDDO(req, res) {
        let identifier = undefined;

        try {
            identifier = await ssiService.getIdentifier(`did:web:${encodeURIComponent(req.get('host') || req.hostname)}:api:${req.params[0]}`);

            if (identifier) {
                const allKeys = identifier.keys.map((key) => ({
                    id: identifier.did + "#" + key.kid,
                    type: keyMapping[key.type],
                    controller: identifier.did,
                    publicKeyHex: key.publicKeyHex
                }));

                const keyAgreementKeyIds = allKeys
                    .filter((key) => ['Ed25519VerificationKey2018', 'X25519KeyAgreementKey2019'].includes(key.type))
                    .map((key) => key.id)
                const signingKeyIds = allKeys
                    .filter((key) => key.type !== 'X25519KeyAgreementKey2019')
                    .map((key) => key.id)


                res.status(200).json({
                    '@context': 'https://w3id.org/did/v1',
                    id: identifier.did,
                    verificationMethod: allKeys,
                    authentication: signingKeyIds,
                    assertionMethod: signingKeyIds,
                    keyAgreement: keyAgreementKeyIds,
                    service: identifier.services
                });
            } else {
                return res.status(400).json({ message: "Couldn't find identifier "});
            }
        } catch (e) {
            res.status(404).json({ message: "Invalid did" })
        }
    },

    async resolve(req, res) {
        const ddo = await ssiService.resolveIdentifier(`did:web:${process.env.DOMAIN}%3A${process.env.PORT}:api:did`);
        res.status(200).json(ddo);
    },

    async sendMessage(req, res) {
        const reqBody = await req.body;
        if (!reqBody.receiver) {
            return res.status(400).json({ message: "Receiver's did is missing." });
        }
        if (!reqBody.type) {
            return res.status(400).json({ message: "Message's type is missing." });
        }
        if (!reqBody.data) {
            return res.status(400).json({ message: "Message's data is missing." });
        }

        let data = reqBody.data

        switch (reqBody.type) {
            case "connection":
                if (!reqBody.data.eori) {
                    return res.status(400).json({ message: `The 'data.eori' attribute is missing for the DIDComm message of type '${reqBody.type}'.` });
                }
                if (!reqBody.data.response_requested) {
                    return res.status(400).json({ message: `The 'data.response_requested' attribute is missing for the DIDComm message of type '${reqBody.type}'.` });
                }
                break

            case "message":
                if (!reqBody.data.message) {
                    return res.status(400).json({ message: `The 'data.message' attribute is missing for the DIDComm message of type '${reqBody.type}'.` });
                }
                break

            case "credential-v1":
            case "credential-v2":
                if (!reqBody.data.credential_type) {
                    return res.status(400).json({ message: `The 'data.credential_type' attribute is missing for the DIDComm message of type '${reqBody.type}'.` });
                }

                if (reqBody.data.credentials) {
                    const presentation = await ssiService.handlePresentation(reqBody.data.credentials, `did:web:${process.env.DOMAIN}%3A${process.env.PORT}:api:did`, reqBody.receiver);
                    if (!presentation) {
                        return res.status(400).json({ message: "Unsuccessful presentation generation." });
                    } else {
                        data.presentation = presentation;
                    }
                }
        }

        const returnMessage = await ssiService.sendMessage(
            reqBody.type,
            `did:web:${process.env.DOMAIN}%3A${process.env.PORT}:api:did`,
            reqBody.receiver,
            data
        );

        if (returnMessage !== "Connection established." && returnMessage !== "Message received.") {
            return res.status(401).json({ message: returnMessage });
        }

        res.status(200).json({ message: returnMessage });
    },

    async handleMessage(req, res) {
        const reqBody = await req.body;
        const currentTime = new Date().getTime();

        const rawMessage = reqBody.message;
        if (!rawMessage) {
            return res.status(400).json({ message: "Missing message." })
        }

        try {
            const didCommMessageType = await ssiService.getMessageType(rawMessage);
            if (!didCommMessageType) {
                return res.status(401).json({ message: `Invalid DIDComm type.`});
            }
        } catch (e) {
            res.status(401).json({ message: `Could not parse message as DIDComm v2: ${e}`});
        }

        let message = undefined;
        try {
            message = await ssiService.unpackMessage(rawMessage);
            if (!message) {
                return res.status(400).json({ message: "Couldn't unpack the DIDComm message." })
            }
        } catch (e) {
            res.status(500).json({ message: "DIDComm validation error." })
        }

        const { id, expiresAt, type, data, from, to } = message;
        if (!from) {
            return res.status(400).json({ message: "Missing from attribute." });
        }
        if (!to) {
            return res.status(400).json({ message: "Missing to attribute." });
        }
        if (currentTime > expiresAt) {
            return res.status(400).json({ message: 'DIDComm message has expired.' });
        }

        let responseMessage = "";
        const date = new Date(currentTime)

        switch (type) {
            case "connection":
                // { eori: ..., response_requested: ... }
                responseMessage = await ssiService.handleConnection(data, `did:web:${process.env.DOMAIN}%3A${process.env.PORT}:api:did`, from, id, process.env.EORI);

                if (responseMessage !== "Connection established.") {
                    return res.status(400).json({ message: `Failed to establish connection: ${responseMessage}` });
                }

                return res.status(200).json({ message: responseMessage });
            case "message":
                // { message: ... }
                responseMessage = await ssiService.handleMessage(data, from, id, date.toLocaleString())
                if (responseMessage !== "Message received.") {
                    return res.status(400).json({ message: responseMessage });
                }

                return res.status(200).json({ message: responseMessage });
            case "credential-v1":
            case "credential-v2":
                // { credential_type: ..., presentation: ..., owned_credentials: ... }
                if (!data.presentation) {
                    responseMessage = await ssiService.handleSDR(data, `did:web:${process.env.DOMAIN}%3A${process.env.PORT}:api:did`, from)

                    if (responseMessage !== "Message received.") {
                        return res.status(400).json({ message: responseMessage });
                    }

                    return res.status(200).json({ message: responseMessage });
                }

                responseMessage = await ssiService.handleCredential(data, `did:web:${process.env.DOMAIN}%3A${process.env.PORT}:api:did`, from);

                if (type == "credential-v2" && responseMessage == "Invalid presentation.") {
                    if (!data.owned_credentials) {
                        return res.status(400).json({ message: "Couldn't generate alternative SDR. Missing owned credentials message." })
                    }

                    unmet_claims
                    responseMessage = await ssiService.handleAlternativeSDR(
                        data,
                        `did:web:${process.env.DOMAIN}%3A${process.env.PORT}:api:did`,
                        from,
                        unmet_claims,
                        process.env.OPENAI_KEY
                    )

                    if (responseMessage !== "Message received.") {
                        return res.status(400).json({ message: responseMessage });
                    }

                    return res.status(200).json({ message: responseMessage });

                } else if (responseMessage !== "Message received.") {
                    return res.status(400).json({ message: responseMessage });
                }

                return res.status(200).json({ message: responseMessage });
            default:
                res.status(401).json({ message: `Invalid DIDComm message type. Valid types are "connection", "message", "sdr", "credential", "presentation".`});
        }
    }
};

export default ssiComm;

