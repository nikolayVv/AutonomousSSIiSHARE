import * as certificateService from "./certificateService.mjs";
import * as tokenService from "./tokenService.mjs";
import JSONModel from "../models/model.mjs";
const jsonFile = new JSONModel(`../data/agent${process.env.PORT}.json`);

import querystring from "querystring";
import request from "request-promise";
import jwt from "jsonwebtoken";

const requestAccessToken = async (eori, x5c, receiver_eori, private_key, url, access_token) => {
    if (access_token) {
        const begin_cert = "-----BEGIN CERTIFICATE-----\n";
        const end_cert = "\n-----END CERTIFICATE-----"
        const inn_pem_cert = begin_cert.concat(x5c, end_cert);

        try {
            const decoded_token = jwt.verify(access_token, inn_pem_cert, { algorithms: 'RS256' });
            return access_token;
        } catch (error) {
            if (error === 'TokenExpiredError: jwt expired') {
                console.log("Expired token");
            } else {
                console.log("Invalid signature");
            }
        }
    }

    const reqBody = querystring.stringify({
        'grant_type': 'client_credentials',
        'scope': 'iSHARE',
        'client_assertion_type': 'urn:ietf:params:oauth:client-assertion-type:jwt-bearer',
        'client_assertion': tokenService.generateClientAssertion(eori, x5c, receiver_eori, private_key),
        'client_id': eori
    });

    url = `${url}/connect/token`;

    const response = await request.post({
        headers: {
            'Content-Length': Buffer.byteLength(reqBody),
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        url: url,
        body: reqBody,
        resolveWithFullResponse: true,
        rejectUnauthorized: false,
        strictSSL: false
    });
    let resBody = await response.body;
    let parsedData = JSON.parse(resBody);
    if (!parsedData.access_token) {
        return null;
    }
    //const result = await jsonFile.storeAccessToken(parsedData.access_token, receiver_eori);

    return parsedData.access_token;
};

const classicTrustedListExtraction = async (eori, x5c, receiver_eori, private_key, url, access_token) => {
    access_token = await requestAccessToken(eori, x5c, receiver_eori, private_key, url, access_token);
    if(!access_token) {
        return null;
    }
    const authorizationHeader = 'Bearer ' + access_token;
    let response = await request({
        headers: {
            'Authorization': authorizationHeader,
        },
        url: `${url}/trusted_list`,
        method: 'GET',
        resolveWithFullResponse: true,
        rejectUnauthorized: false,
        strictSSL: false
    });
    let resBody = await response.body;
    let parsedData = JSON.parse(resBody);
    if (!parsedData.data) {
        return null;
    }

    return parsedData.data;
};

const classicPartyExtraction = async (eori, x5c, receiver_eori, client_id, private_key, url, access_token) => {
    access_token = await requestAccessToken(eori, x5c, receiver_eori, private_key, url, access_token);
    if(!access_token) {
        return null;
    }
    const authorizationHeader = 'Bearer ' + access_token;

    let response = await request({
        headers: {
            'Authorization': authorizationHeader,
        },
        url: `${url}/parties?eori=${client_id}`,
        method: 'GET',
        resolveWithFullResponse: true,
        rejectUnauthorized: false,
        strictSSL: false
    });
    let resBody = await response.body;
    let parsedData = JSON.parse(resBody);
    if (!parsedData.data) {
        return parsedData;
    }

    return parsedData.data[0];
};

const findPartyInOtherSatellites = async (eori, x5c, client_id, private_key, satellites, access_tokens) => {
    const result = await jsonFile.load();

    for (let i = 0; i < satellites.length; i++) {
        let curr_satellite = satellites[i];
        let party = undefined;
        let connectionIsValid = false;
        if (!result.error) {
            if (jsonFile.connectionIsValid(curr_satellite.id)) {
                connectionIsValid = true;
            }
        }

        if (connectionIsValid) {
            // TODO -> Use the SSI Connection to get the party
        } else {
            const token = access_tokens.find(token => token.eori === curr_satellite.id);
            let access_token = undefined;
            if (token) {
                access_token = token.access_token;
            }
            party = await classicPartyExtraction(
                eori,
                x5c,
                curr_satellite.id,
                client_id,
                private_key,
                curr_satellite.url,
                access_token
            );
        }

        if (!party) {
            continue;
        }
        return party;
    }

    return null;
}

export {
    classicTrustedListExtraction,
    classicPartyExtraction,
    findPartyInOtherSatellites,
    requestAccessToken
}