import jwt from "jsonwebtoken";
import forge from "node-forge";
import request from "request-promise";
import JSONModel from "../models/model.mjs";
const jsonFile = new JSONModel(`../data/agent${process.env.PORT}.json`);

import * as certificateService from "../services/certificateService.mjs";
import * as satelliteService from "../services/satelliteService.mjs";
import {stat} from "fs";

const Verification = {
    async verifyClientAssertion(req, res, next) {
        const reqBody = await req.body;
        const epoch = Math.floor(new Date() / 1000);

        // Check the request attributes
        const grant_type = reqBody.grant_type;
        if (!grant_type) {
            return res.status(400).json({ message: 'Request has no client_credentials.' });
        }
        if (grant_type !== "client_credentials") {
            return res.status(400).json({ message: 'Request has invalid client_assertion.' });
        }

        const scope = reqBody.scope;
        if (!scope) {
            return res.status(400).json({ message: 'Request has no scope.' });
        }
        if (scope !== "iSHARE") {
            return res.status(400).json({ message: 'Request has invalid scope.' });
        }

        const client_assertion_type = reqBody.client_assertion_type;
        if (!client_assertion_type) {
            return res.status(400).json({ message: 'Request has no client_assertion_type.' });
        }
        if (client_assertion_type !== "urn:ietf:params:oauth:client-assertion-type:jwt-bearer") {
            return res.status(400).json({ message: 'Request has invalid client_assertion_type.' });
        }

        const client_id = reqBody.client_id;
        if (!client_id) {
            return res.status(400).json({ message: 'Request has no client_id.' });
        }

        const ca = reqBody.client_assertion;
        if (!ca) {
            return res.status(400).json({ message: 'Request has no client_assertion.' });
        }
        const client_assertion = ca.replace(/\s/g,'');
        const decodedClientAssertion = jwt.decode(client_assertion, { complete: true });


        // Check the Client Assertion's header
        const x5c = decodedClientAssertion.header.x5c;
        if (!x5c) {
            return res.status(400).json({message: 'Client assertion\'s header has no x5c value.'});
        }

        const alg = decodedClientAssertion.header.alg;
        if (!alg) {
            return res.status(400).json({ message: 'Client assertion\'s header has no alg value.' });
        }
        if (alg.substring(0,2) !== 'RS' || alg.substring(2,5) < 256) {
            return res.status(400).json({ message: 'Client assertion\'s header has invalid alg value.' });
        }

        const typ = decodedClientAssertion.header.typ;
        if (!typ) {
            return res.status(400).json({ message: 'Client assertion\'s header has no typ value.' });
        }
        if (typ !== 'JWT') {
            return res.status(400).json({ message: 'Client assertion\'s header has invalid typ value.' });
        }


        // Check the Client Assertion's payload
        const iss = decodedClientAssertion.payload.iss;
        if (!iss) {
            return res.status(400).json({ message: 'Client assertion\'s payload has no iss value.' });
        }
        if (iss !== client_id) {
            return res.status(400).json({ message: 'Client assertion\'s payload has different iss and client_id values.' });
        }

        const sub = decodedClientAssertion.payload.sub;
        if (!sub) {
            return res.status(400).json({ message: 'Client assertion\'s payload has no sub value.' });
        }
        if (sub !== iss) {
            return res.status(400).json({ message: 'Client assertion\'s payload has different sub and iss values.' });
        }

        const jti = decodedClientAssertion.payload.jti;
        if (!jti) {
            return res.status(400).json({ message: 'Client assertion\'s payload has no jti value.' });
        }

        const iat = decodedClientAssertion.payload.iat;
        if (!iat) {
            return res.status(400).json({ message: 'Client assertion\'s payload has no iat value.' });
        }
        if (iat > epoch + 5) {
            return res.status(400).json({ message: 'Client assertion\'s payload iat value is after the current time.' });
        }

        const nbf = decodedClientAssertion.payload.nbf;
        if (!nbf) {
            return res.status(400).json({ message: 'Client assertion\'s payload has no nbf value.' });
        }
        if (nbf !== iat) {
            return res.status(400).json({ message: 'Client assertion\'s payload has different nbf and iat values.' });
        }

        const exp = decodedClientAssertion.payload.exp;
        if (!exp) {
            return res.status(400).json({ message: 'Client assertion\'s payload has no exp value.' });
        }
        if (exp !== iat + 30) {
            return res.status(400).json({ message: 'Client assertion\'s payload has invalid exp value (should be 30s.).' });
        }
        if (exp < epoch + 5) {
            return res.status(400).json({ message: 'Client assertion has expired.' });
        }

        const aud = decodedClientAssertion.payload.aud;
        if (!aud) {
            return res.status(400).json({ message: 'Client assertion\'s payload has no aud value.' });
        }
        if (aud !== process.env.EORI) {
            return res.status(400).json({ message: 'Client assertion\'s payload has different aud value than the EORI of the agent.' });
        }

        next();
    },

    async verifyCertificate(req, res, next) {
        const reqBody = await req.body;
        const client_assertion = reqBody.client_assertion.replace(/\s/g,'');
        const decodedClientAssertion = jwt.decode(client_assertion, { complete: true });
        const x5c = decodedClientAssertion.header.x5c;
        const signature = reqBody.client_assertion.split('.').slice(2).join('.');
        if (!signature) {
            return res.status(400).json({ message: 'Client assertion JWT signature is missing.' });
        }

        const pem_cert = certificateService.convertFromX5CToCert(x5c);
        try {
            const derKey = forge.util.decode64(x5c[0]);
            const asnObj = forge.asn1.fromDer(derKey);
            const asn1Cert = forge.pki.certificateFromAsn1(asnObj);
            const certificate = forge.pki.certificateFromPem(pem_cert);
            const length =  certificate.publicKey.n.bitLength();

            if(length < 2048){
                return res.status(400).json({ message: 'The certificate has length smaller than 2048.' });
            }

            const result = await jsonFile.load();
            if (result.error) {
                return res.status(500).json({ message: result.error });
            }
            const is_satellite = jsonFile.isSpecificRole("SchemeOwner");
            let trusted_list = null;

            if (is_satellite) {
                trusted_list = result.data.trusted_list;
            } else {
                if (jsonFile.connectionIsValid(result.data.satellite_id)) {
                    // TODO -> Use the SSI Connection
                } else {
                    const x5c = certificateService.convertFromCertToX5C(process.env.CERT);
                    const token = jsonFile.findAccessTokenByEori(result.data.satellite_id)
                    let access_token = undefined;
                    if (token) {
                        access_token = token.access_token;
                    }
                    trusted_list = await satelliteService.classicTrustedListExtraction(
                        process.env.EORI,
                        x5c,
                        result.data.satellite_id,
                        process.env.PRIVATE_KEY,
                        process.env.SATELLITE_URL,
                        access_token
                    );
                }
            }
            if (!trusted_list) {
                return res.status(500).json({ message: 'An internal error occurred.' });
            }

            let valid= false;
            for (let i = 0; i < trusted_list.length; i++) {
                try {
                    const caStore = forge.pki.createCaStore([ trusted_list[i].crt ]);
                    forge.pki.verifyCertificateChain(caStore, [asn1Cert]);
                    if (trusted_list[i].validity === "valid") {
                        valid = true;
                        break;
                    }
                } catch (e) {
                    console.log(trusted_list[i].name + " is invalid.");
                }
            }
            if (!valid) {
                return res.status(401).json({ message: 'Failed to verify the certificate on the chain.' });
            }
            next();
        } catch (err) {
            console.log(err)
            if (err === 'TokenExpiredError: jwt expired') {
                return res.status(401).json({ message: 'JWT is expired.' });
            } else {
                return res.status(401).json({ message: 'Client assertion JWT header "x5c" contains invalid certificate.' });
            }
        }
    },

    async verifyOrigin(req, res, next) {
        const origin = req.headers.referer;
        if (origin.includes(`:${process.env.PORT}/`)) {
            next();
        } else {
            return res.status(403).json({ message: 'Forbidden: Request origin not allowed.' });
        }
    },

    async verifyNetworkStatus(req, res, next) {
        const client_id = await req.body.client_id;
        const result = await jsonFile.load();
        if (result.error) {
            return res.status(500).json({ message: result.error });
        }

        const is_satellite = jsonFile.isSpecificRole("SchemeOwner");
        const x5c = certificateService.convertFromCertToX5C(process.env.CERT);
        let status = undefined;
        let party = undefined;
        if (is_satellite) {
            party = jsonFile.findPartyById(client_id);
            if (!party) {
                party = await satelliteService.findPartyInOtherSatellites(
                    process.env.EORI,
                    x5c,
                    client_id,
                    process.env.PRIVATE_KEY,
                    result.data.satellites,
                    result.data.access_tokens || []
                );
            } else {
                status = party.status;
            }
        } else {
            if (jsonFile.connectionIsValid(result.data.satellite_id)) {
                // TODO -> Use the SSI Connection to get the party
            } else {
                party = await satelliteService.classicPartyExtraction(
                    process.env.EORI,
                    x5c,
                    result.data.satellite_id,
                    client_id,
                    process.env.PRIVATE_KEY,
                    process.env.SATELLITE_URL,
                    result.data.access_token
                );
            }
        }

        if (!party) {
            return res.status(401).json({ message: "Couldn't find party with the corresponding EORI in the network." });
        } else {
            if (!status) {
                status = party.adherence.status;
            }

            if (status !== 'Active') {
                return res.status(401).json({ message: 'The participant is not active in the network.' });
            }
            next();
        }
    },

    async verifyAccessToken(req, res, next) {
        const headers = await req.headers;
        if (!headers.authorization) {
            return res.status(403).json({ message: 'Forbidden: You need access token to perform this action.' });
        }
        if (headers.authorization.substring(0,6) !== "Bearer") {
            return res.status(400).json({ message: 'Invalid token.' });
        }

        const x5c = certificateService.convertFromCertToX5C(process.env.CERT);
        const token = headers.authorization.substring(7);
        const begin_cert = "-----BEGIN CERTIFICATE-----\n";
        const end_cert = "\n-----END CERTIFICATE-----"
        const inn_pem_cert = begin_cert.concat(x5c, end_cert);

        try {
            const decoded_token = jwt.verify(token, inn_pem_cert, { algorithms: 'RS256' });
            if (decoded_token.iss !== process.env.EORI) {
                return res.status(401).json({ message: 'The access token is not provided by this agent.' });
            }
            req.client_id = decoded_token.aud;
            next();
        } catch (error) {
            if (error === 'TokenExpiredError: jwt expired') {
                return res.status(401).json({ message: 'The access token is expired.' });
            } else {
                return res.status(401).json({ message: 'Invalid signature.' });
            }
        }
    },

    async verifySatellite(req, res, next) {
        const result = await jsonFile.load();
        if (result.error) {
            return res.status(500).json({ message: result.error });
        }

        const valid = jsonFile.isSpecificRole("SchemeOwner");
        if (valid) {
            next();
        } else {
            res.status(403).json({ message: 'Forbidden: Cannot perform the following action since the agent is not a Satellite.' });
        }
    },


    async verifyProvider(req, res, next) {
        const result = await jsonFile.load();
        if (result.error) {
            return res.status(500).json({ message: result.error });
        }

        const valid = jsonFile.isSpecificRole("ServiceProvider");

        if (valid) {
            next();
        } else {
            res.status(403).json({ message: 'Forbidden: Cannot perform the following action since the agent is not a Service Provider.' });
        }
    },

    async verifyAR(req, res, next) {
        const result = await jsonFile.load();
        if (result.error) {
            return res.status(500).json({ message: result.error });
        }

        const valid = jsonFile.isSpecificRole("AuthorizationRegistry");

        if (valid) {
            next();
        } else {
            res.status(403).json({ message: 'Forbidden: Cannot perform the following action since the agent is not an Authorization Registry.' });
        }
    },

    async verifyDelegation(req, res, next) {
        // Verify signatures of the delegations, verify if the actions are permitted for the chosen data.
    }
};

export default Verification;