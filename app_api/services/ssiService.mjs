import ssiAgent from "../models/ssiAgent.mjs";
import crypto from "crypto";
import * as cheerio from 'cheerio';
import JSONModel from "../models/model.mjs";
import request from "request-promise";
import querystring from "querystring";
import * as tokenService from "./tokenService.mjs";
import agent from "../controllers/agent.mjs";
const jsonFile = new JSONModel(`../data/agent${process.env.PORT}.json`);


const generateRandomString = async (length) => {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    return result;
}


const generateMessageId = async (content) => {
    const startRandomString = generateRandomString(6);
    const endRandomString = generateRandomString(6);
    const combinedString = startRandomString + content + endRandomString;

    // Calculate hash of the combined string
    const encoder = new TextEncoder();
    const data = encoder.encode(combinedString);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashHex = Array.from(new Uint8Array(hashBuffer)).map(byte => byte.toString(16).padStart(2, '0')).join('');

    return hashHex;
}


const generateCredential = async (issuer, holder, credential_type) => {
    let credentialData = null;
    const credential_context = `https://schema.org/${credential_type}`;

    try {
        const html = await request(credential_context);
        const $ = cheerio.load(html, { xmlMode: false });

        const jsonLd = $('script[type="application/ld+json"]').html();

        if (jsonLd) {
            credentialData = JSON.parse(jsonLd);
        } else {
            console.error('No JSON-LD data found for schema:', credential_type);
            return null;
        }
    } catch (error) {
        console.error('Error fetching schema example data:', error);
        return null;
    }

    const { "@context": context, "@type": type, ...credentialSubject } = credentialData;
    credentialSubject.id = holder;

    return await ssiAgent.createVerifiableCredential({
        issuer: { id: issuer },
        '@context': [
            'https://www.w3.org/ns/credentials/v2',
            credential_context
        ],
        type: [
            'VerifiableCredential',
            credential_type
        ],
        issuanceDate: new Date().toISOString(),
        credentialSubject: credentialSubject,
        proofFormat: 'lds',
    });
}


const generatePresentation = async (holder, verifiers, credentials) => {
    return await ssiAgent.createVerifiablePresentation({
        presentation: {
            '@context': ['https://www.w3.org/ns/credentials/v2'],
            type: ['VerifiablePresentation'],
            issuanceDate: new Date().toISOString(),
            holder: holder,
            verifiers: verifiers,
            verifiableCredential: credentials,
        },
        challenge: 'VERAMO',
        proofFormat: 'jwt'
    });
}


function generateDataPrompt(data, label) {
    return data.map(item => `${label}: ${item.name || item}`).join("\n");
}


const generateModelPrompt = (sourceClasses, sourceProperties, targetClasses, targetProperties) => {
    const baseInstructions = `
        You are an expert schema alignment AI tasked with mapping properties from one or more source class to one or more properties from one or more target classes based on similarity. Consider the following:

        - *Semantic Similarity*: Compare names and descriptions for related meanings.
        - *Contextual Similarity*: Consider how properties are used in their classes and their broader context.
        - *Parent Relations*: Account for similarities between parent classes and properties for hierarchical alignment.
        - *Multi-Attribute Mapping*: A source property may map to one or more target properties and vice versa if relevant.

        I will provide the source properties with descriptions, classes, and related attributes. Then, I will provide the target properties and schemas in a similar way. Parent properties/classes information may also be included for broader context.
    `;

    const sourceInformation = `
        Map the following ${sourceProperties.length} source properties:
        ${generateDataPrompt(sourceProperties, 'Source property')}
        The source properties are part of the following ${sourceClasses.length} classes:
        ${generateDataPrompt(sourceClasses, 'Source class')}
    `;

    const targetInformation = `
        Map the source properties to one or more of these ${targetProperties.length} target properties:
        ${generateDataPrompt(targetProperties, 'Target property')}
        The target properties are part of the following ${targetClasses.length} classes:
        ${generateDataPrompt(targetClasses, 'Target class')}
    `;

    const taskDescription = `
        Your task is to map each source property to valid target properties to solve a critical problem. Let's work this out step by step to ensure accuracy.
        Return a JSON that contains the source properties as keys and lists of mapping strings as the values. Each mapping string should follow the format 'target_property_1 (corresponding_class), ..., target_property_N (corresponding_class)', where none of the target properties should be the same as the source properties, otherwise the corresponding mapping is not valid. The mappings should be also ordered by confidence, highest to lowest. If for some key (source property) no mappings are valid and/or none of them have at least 80% confidence, return an empty list ([]) as the corresponding value.
    `;

    return {
        system: baseInstructions,
        user: `${sourceInformation}\n\n${targetInformation}\n\n${taskDescription}`,
    };
}


const generateSDR = async (issuer, credential_type) => {
    const result = await jsonFile.load();
    if (result.error) {
        return null;
    }
    if (!result.data.policies) {
        return null
    }

    const claims = result.data.policies[credential_type];
    if (!claims) {
        console.error(`No claims found for credential type: ${credential_type}`);
        return null;
    }

    return {
        issuer: issuer,
        tag: `SDR for generation of credential of type '${credential_type}'`,
        claims: claims,
    }
}


const parseMappings = (response) => {
    const mappings = response.split(',').map(mapping => {
        const [targetProperty, targetClass] = mapping.split('(');
        return {
            targetProperty: targetProperty.trim(),
            targetClass: targetClass ? targetClass.replace(')', '').trim() : '',
        };
    });
    return mappings.filter(mapping => mapping.targetProperty && mapping.targetClass);
}


const generateAlternativeSDR = async (issuer, credential_type, old_sdr, unmet_claims, targetClasses, targetProperties, openai_key) => {
    const new_claims = [];

    for (const unmet_claim of unmet_claims) {
        const sourceProperty = unmet_claim.claimType;
        const sourceClass = unmet_claim.credentialType;

        const prompt = generateModelPrompt(
            [sourceClass],
            [sourceProperty],
            targetClasses,
            targetProperties
        );

        const reqBody = new URLSearchParams({
            model: 'ft:gpt-4o-mini-2024-07-18:personal:ishare:AQPypjCF',
            prompt: prompt,
        }).toString();

        try {
            const response = await request.post({
                uri: 'https://api.openai.com/v1/completions',
                headers: {
                    'Authorization': `Bearer ${openai_key}`,
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: reqBody,
                resolveWithFullResponse: true,
            });

            const data = JSON.parse(response.body);
            const modelResponse = data.choices && data.choices[0].text.trim();

            if (modelResponse) {
                const mappings = parseMappings(modelResponse);
                if (mappings.length > 0) {
                    const matchedClaim = old_sdr.claims.find(claim => claim.property === sourceProperty);

                    if (matchedClaim) {
                        mappings.forEach(mapping => {
                            const newClaim = { ...matchedClaim };

                            newClaim.claimType = mapping.targetProperty;
                            newClaim.credentialType = mapping.targetClass;

                            new_claims.push(newClaim);
                        });
                    }
                } else {
                    console.log(`No valid mappings found for ${sourceProperty}.`);
                    return null;
                }
            } else {
                console.log(`No response from model for ${sourceProperty}. Exiting...`);
                return null;
            }
        } catch (error) {
            console.error(`Error processing claim ${sourceProperty}:`, error);
            return null;
        }
    }

    return {
        issuer: issuer,
        tag: `alternative SDR for generation of credential of type '${credential_type}'`,
        claims: new_claims,
    }
}


const getOrGenerateIdentifier = async (alias) => {
    return await ssiAgent.didManagerGetOrCreate({
        provider: 'did:web',
        alias: alias
    });
}


const getIdentifier = async (did) => {
    return await ssiAgent.didManagerGet({
        did: did
    });
}


const storeIdentifier = async (identifier) => {
    return await ssiAgent.didManagerImport(identifier);
}


const deleteIdentifierByAlias = async (eori) => {
    await ssiAgent.didManagerDelete({
        alias: eori
    });
}


const resolveIdentifier = async (did) => {
    return await ssiAgent.resolveDid({ didUrl: did });
}


const getIdentifiers = async() => {
    return await ssiAgent.didManagerFind();
}


const unpackMessage = async(message) => {
    return await ssiAgent.handleMessage({
        raw: message
    });
}


const getMessageType = async(message) => {
    return await ssiAgent.getDIDCommMessageMediaType({ message: message })
}


const sendMessage = async (type, sender, receiver, body) => {
    const currentTime = new Date();
    const expirationTime = new Date(currentTime.getTime() + (5 * 60 * 1000));

    const id = await generateMessageId(`${currentTime}-${sender}-${JSON.stringify(body)}-${receiver}-${expirationTime}`)
    const message = {
        id: id,
        type: type,
        to: receiver,
        from: sender,
        created_time: currentTime.getTime(),
        expires_time: expirationTime.getTime(),
        body: body,
    };

    const packedMessage = await ssiAgent.packDIDCommMessage({
       packing: 'authcrypt',
       message
    });

    const identifier = await resolveIdentifier(receiver);
    let url = undefined;
    if (identifier.didDocument.service) {
        const services = identifier.didDocument.service
        for (let i = 0; i < services.length; i++) {
            const service = services[i];

            if (service.type === "DIDCommMessaging") {
                url = service.serviceEndpoint;
                break;
            }
        }
    }

    if (url) {
        const reqBody = querystring.stringify(packedMessage);
        try {
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
            const resBody = await response.body;
            const parsedData = JSON.parse(resBody);

            if (parsedData.message === "Presentation generated.") {
                return parsedData;
            }
            return parsedData.message;
        } catch (error) {
            if (error.body) {
                return `Unsuccessful DIDComm message sending: ${error.message}.`
            }
            return `Unsuccessful DIDComm message sending: ${error}.`
        }
    }

    return "Missing DIDComm service."
}


const handleConnection = async(data, sender, receiver, id, eori) => {
    let answer = "Connection established.";

    if (!data.eori) {
        return "Missing EORI attribute.";
    }

    if (data.response_requested) {
        answer = await sendMessage("connection", sender, receiver, {
            eori: eori,
            response_requested: false
        });
    }

    if (answer === "Connection established.") {
        const result = await jsonFile.load();
        if (result.error) {
            return "Error when accessing the list of existing connections";
        }

        if (!jsonFile.connectionExists(receiver)) {
            const didResult = await jsonFile.addDidConnection(receiver, data.eori)
            if (didResult.error) {
                return "Error when adding the connection.";
            }
        } else if (!jsonFile.connectionIsValid(receiver)) {
            return "Inactive connection."
        }
    }

    return answer;
}


const handleMessage = async(data, from, id, currentTime) => {
    const result = await jsonFile.load();
    if (result.error) {
        return "Error when accessing the list of existing messages";
    }

    if (!jsonFile.connectionExists(from)) {
        return "Connection not established! Send a DIDComm message with type 'connection' first."
    } else if (!jsonFile.connectionIsValid(from)) {
        return "Inactive connection."
    }

    if (!data.message) {
        return "Missing message attribute";
    }

    const messageResult = await jsonFile.addMessage(currentTime, id, data.message)
    if (messageResult.error) {
        return "Error when adding the message.";
    }

    return "Message received.";
}


const handleSDR = async (data, sender, receiver) => {
    if (!jsonFile.connectionExists(receiver)) {
        return "Connection not established! Send a DIDComm message with type 'connection' first."
    } else if (!jsonFile.connectionIsValid(receiver)) {
        return "Inactive connection."
    }

    if (!data.credential_type) {
        return "The credential type is not specified.";
    }

    const result = await jsonFile.load();
    if (result.error) {
        return "Error when accessing the list of existing messages";
    }

    const sdr = await generateSDR(sender, data.credential_type);
    const answer = await sendMessage("message", sender, receiver, { message: sdr });

    if (answer !== "Message received.") {
        return "Unsuccessful issuing of SDR.";
    }

    return answer
}


// TODO: Generation of alternative SDR
// Call the model, get the alternative mappings, create the SDR manually
const handleAlternativeSDR = async (data, sender, receiver, unmet_claims, openai_key) => {
    if (!jsonFile.connectionExists(receiver)) {
        return "Connection not established! Send a DIDComm message with type 'connection' first."
    } else if (!jsonFile.connectionIsValid(receiver)) {
        return "Inactive connection."
    }

    if (!data.credential_type) {
        return "The credential type is not specified.";
    }

    const result = await jsonFile.load();
    if (result.error) {
        return "Error when accessing the list of existing messages";
    }

    //generate the target classes and properties

    const sdr = generateAlternativeSDR(
        sender,
        data.credential_type,
        await generateSDR(sender, data.credential_type),
        unmet_claims,
        targetClasses,
        targetProperties,
        openai_key
    );
    const answer = await sendMessage("message", sender, receiver, { message: sdr });

    if (answer !== "Message received.") {
        return "Unsuccessful issuing of SDR.";
    }

    return answer
}


const handleCredential = async (data, sender, receiver) => {
    if (!jsonFile.connectionExists(receiver)) {
        return "Connection not established! Send a DIDComm message with type 'connection' first."
    } else if (!jsonFile.connectionIsValid(receiver)) {
        return "Inactive connection."
    }

    if (!data.credential_type) {
        return "Missing credential type.";
    }

    try {
        await ssiAgent.validatePresentationAgainstSdr({
            presentation: data.presentation,
            sdr: await generateSDR(sender, data.credential_type),
        })
    } catch (e) {
        console.error(e.details)
        return "Invalid presentation.";
    }

    const credential = await generateCredential(sender, receiver, data.credential_type);

    try {
        await ssiAgent.verifyCredential({
            credential: credential,
            proofFormat: 'lds'
        });

        const answer = await sendMessage("message", sender, receiver, { message: credential });

        if (answer !== "Message received.") {
            return "Unsuccessful issuing of credential.";
        }

        return answer;
    } catch (e) {
        return "Invalid Credential";
    }
}


const handlePresentation = async(credentials, sender, receiver) => {
    if (!jsonFile.connectionExists(receiver)) {
        return null;
    } else if (!jsonFile.connectionIsValid(receiver)) {
        return null;
    }

    const result = await jsonFile.load();
    if (result.error) {
        return null;
    }

    try {
        return await generatePresentation(sender, [receiver], credentials);
    } catch (e) {
        console.error(e)
        return null
    }
}


export {
    getOrGenerateIdentifier,
    getIdentifier,
    storeIdentifier,
    resolveIdentifier,
    getIdentifiers,
    deleteIdentifierByAlias,
    unpackMessage,
    getMessageType,
    sendMessage,
    handleConnection,
    handleMessage,
    handleSDR,
    handleAlternativeSDR,
    handleCredential,
    handlePresentation
}