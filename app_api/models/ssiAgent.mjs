import {
    createAgent
} from "@veramo/core";
import {
    DataStore,
    DataStoreDiscoveryProvider,
    DataStoreORM,
    DIDStore,
    Entities,
    KeyStore,
    migrations,
    PrivateKeyStore
} from "@veramo/data-store";
import {
    CredentialIssuerLD,
    LdDefaultContexts,
    VeramoEcdsaSecp256k1RecoverySignature2020,
    VeramoEd25519Signature2018,
    VeramoEd25519Signature2020,
    VeramoJsonWebSignature2020
} from "@veramo/credential-ld";

import { contexts as credential_contexts } from "@transmute/credentials-context";
import { CredentialPlugin, W3cMessageHandler } from "@veramo/credential-w3c";
import { SdrMessageHandler, SelectiveDisclosure } from "@veramo/selective-disclosure";
import { DIDComm, DIDCommHttpTransport, DIDCommMessageHandler } from "@veramo/did-comm";
import { DIDDiscovery } from "@veramo/did-discovery";

import { DataSource } from "typeorm";
import { DIDResolverPlugin } from "@veramo/did-resolver";
import { getResolver as webDidResolver } from "web-did-resolver";
import { getDidKeyResolver, KeyDIDProvider } from "@veramo/did-provider-key";
import { KeyManagementSystem, SecretBox } from "@veramo/kms-local";
import { WebDIDProvider } from "@veramo/did-provider-web";
import { MessageHandler } from "@veramo/message-handler";
import { JwtMessageHandler } from "@veramo/did-jwt";
import { PeerDIDProvider, getResolver as peerDidResolver } from "@veramo/did-provider-peer";
import { KeyManager } from "@veramo/key-manager";
import { AliasDiscoveryProvider, DIDManager } from "@veramo/did-manager";

let agentName = process.env.AGENT || "Agent";
agentName = agentName.replace(/\s/g, '');

const dbConnection = new DataSource({
    name: agentName || 'test',
    type: 'sqlite',
    database: `${agentName}.sqlite`,
    synchronize: false,
    migrations,
    migrationsRun: true,
    logging: false,
    entities: Entities
}).initialize();

const defaultKms = 'local';

const pemPrivateKey = process.env.PRIVATE_KEY || "b6a60ffc46e84840158b4ec3b1510a8b803380b4792af73ce7f6a69cbde7bbf6";
const rsaPrivateKey = pemPrivateKey.replace('-----BEGIN RSA PRIVATE KEY-----', '')
    .replace('-----END RSA PRIVATE KEY-----', '')
    .replace(/\n/g, '');

// Convert RSA private key to Buffer
const rsaPrivateKeyBuffer = Buffer.from(rsaPrivateKey, 'base64');

// Convert RSA private key buffer to hexadecimal string
const rsaPrivateKeyHex = rsaPrivateKeyBuffer.toString('hex');

const ssiAgent = createAgent({
    plugins: [
        new DIDResolverPlugin({
            ...webDidResolver(),
            ...getDidKeyResolver(),
            ...peerDidResolver()
        }),
        new KeyManager({
            store: new KeyStore(dbConnection),
            kms: {
                [defaultKms]: new KeyManagementSystem(new PrivateKeyStore(dbConnection, new SecretBox(rsaPrivateKeyHex)))
            }
        }),
        new DIDManager({
            store: new DIDStore(dbConnection),
            defaultProvider: 'did:web',
            providers: {
                'did:web': new WebDIDProvider({ defaultKms }),
                'did:key': new KeyDIDProvider({ defaultKms }),
                'did:peer': new PeerDIDProvider({ defaultKms })
            }
        }),
        new DataStore(dbConnection),
        new DataStoreORM(dbConnection),
        new MessageHandler({
            messageHandlers: [
                new DIDCommMessageHandler(),
                new JwtMessageHandler(),
                new W3cMessageHandler(),
                new SdrMessageHandler()
            ]
        }),
        new DIDComm({ transports: [new DIDCommHttpTransport()]}),
        new CredentialPlugin(),
        new CredentialIssuerLD({
            contextMaps: [LdDefaultContexts, credential_contexts],
            suites: [
                new VeramoEcdsaSecp256k1RecoverySignature2020(),
                new VeramoEd25519Signature2018(),
                new VeramoJsonWebSignature2020(),
                new VeramoEd25519Signature2020(),
            ],
        }),
        new SelectiveDisclosure(),
        new DIDDiscovery({
            providers: [new AliasDiscoveryProvider(), new DataStoreDiscoveryProvider()]
        })
    ]
});

const identifier = await ssiAgent.didManagerGetOrCreate({
    provider: 'did:web',
    alias: `${process.env.DOMAIN || "localhost"}%3A${process.env.PORT || "3000"}:api:did`,
    options: {
        keyType: 'Ed25519'
    }
});

// const key = await ssiAgent.keyManagerCreate({
//     type: 'Ed25519',
//     kms: 'local'
// });

// await ssiAgent.didManagerAddKey({
//     did: identifier.did,
//     key: key
// });


// await ssiAgent.didManagerAddService({
//     did: ddo.did,
//     service: {
//         id: `${process.env.EORI}-DIDCommEstablishMain`,
//         type: 'DIDCommEstablish',
//         serviceEndpoint: `http://localhost:${process.env.PORT}/api/ssi/connect`,
//         description: 'Endpoint for establishing a DIDComm.'
//     }
// });

await ssiAgent.didManagerAddService({
    did: identifier.did,
    service: {
        id: `${identifier.did}#didcomm-1`,
        type: 'DIDCommMessaging',
        serviceEndpoint: `https://${process.env.DOMAIN || "localhost"}:${process.env.PORT || "3000"}/api/did/message`,
        description: 'Handles incoming DIDComm messages'
    }
});
// Generate membership credential

export default ssiAgent;