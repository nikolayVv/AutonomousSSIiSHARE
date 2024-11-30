import crypto from "crypto";
import jwt from "jsonwebtoken";


const generateClientAssertion = (eori, x5c_value, eori_receiver, private_key) => {
    private_key = private_key.replace(/\\n/g, '\n');
    const iat = Math.floor(new Date() / 1000);
    const header = {
        "alg": "RS256",
        "typ": "JWT",
        "x5c": [x5c_value]
    };
    const payload = {
        "iss": eori,
        "sub": eori,
        "jti": crypto.randomBytes(16).toString('hex'),
        "iat": iat,
        "nbf": iat,
        "exp": iat + 30,
        "aud": eori_receiver
    };

    return jwt.sign(payload, private_key, { algorithm: 'RS256', header: header });
};

const generateClientAssertionSSI = (eori, x5c_value, eori_receiver, private_key, did) => {
    private_key = private_key.replace(/\\n/g, '\n');
    const iat = Math.floor(new Date() / 1000);
    const header = {
        "alg": "RS256",
        "typ": "JWT",
        "x5c": [x5c_value]
    };
    const payload = {
        "iss": eori,
        "sub": eori,
        "jti": crypto.randomBytes(16).toString('hex'),
        "iat": iat,
        "nbf": iat,
        "exp": iat + 30,
        "aud": eori_receiver,
        "did": did
    };

    return jwt.sign(payload, private_key, { algorithm: 'RS256', header: header });
};

const generateAccessToken = (eori, x5c_value, client_id, private_key) => {
    private_key = private_key.replace(/\\n/g, '\n');
    const iat = Math.floor(new Date() / 1000);
    const header = {
        "alg": "RS256",
        "typ": "JWT",
        "x5c": [x5c_value]
    };
    const payload = {
        "iss": eori,
        "sub": eori,
        "jti": crypto.randomBytes(16).toString('hex'),
        "iat": iat,
        "nbf": iat,
        "exp": iat + 3600,
        "aud": client_id
    };

    return jwt.sign(payload, private_key, { algorithm: 'RS256', noTimestamp: true, header: header });
};

const generateAccessTokenSSI = (eori, x5c_value, client_id, private_key, did) => {
    private_key = private_key.replace(/\\n/g, '\n');
    const iat = Math.floor(new Date() / 1000);
    const header = {
        "alg": "RS256",
        "typ": "JWT",
        "x5c": [x5c_value]
    };
    const payload = {
        "iss": eori,
        "sub": eori,
        "jti": crypto.randomBytes(16).toString('hex'),
        "iat": iat,
        "nbf": iat,
        "exp": iat + 3600,
        "aud": client_id,
        "did": did
    };

    return jwt.sign(payload, private_key, { algorithm: 'RS256', noTimestamp: true, header: header });
};

const generateDelegationToken = (eori, x5c_value, client_id, delegation_evidence, private_key) => {
    private_key = private_key.replace(/\\n/g, '\n');
    const iat = Math.floor(new Date() / 1000);
    const header = {
        "alg": "RS256",
        "typ": "JWT",
        "x5c": [x5c_value]
    };
    const payload = {
        "iss": eori,
        "sub": eori,
        "jti": crypto.randomBytes(16).toString('hex'),
        "iat": iat,
        "nbf": iat,
        "exp": iat + 30,
        "aud": client_id,
        "delegationEvidence": delegation_evidence
    };

    return jwt.sign(payload, private_key, { algorithm: 'RS256', noTimestamp: true, header: header });
};

const generateCapabilitiesToken = (eori, x5c_value, capabilities, private_key) => {
    private_key = private_key.replace(/\\n/g, '\n');
    const iat = Math.floor(new Date() / 1000);
    const header = {
        "alg": "RS256",
        "typ": "JWT",
        "x5c": [x5c_value]
    };
    const payload = {
        "iss": eori,
        "sub": eori,
        "jti": crypto.randomBytes(16).toString('hex'),
        "iat": iat,
        "nbf": iat,
        "exp": iat + 30,
        "capabilities": capabilities
    };

    return jwt.sign(payload, private_key, { algorithm: 'RS256', noTimestamp: true, header: header });
};

export {
    generateClientAssertion,
    generateClientAssertionSSI,
    generateAccessToken,
    generateAccessTokenSSI,
    generateDelegationToken,
    generateCapabilitiesToken
};