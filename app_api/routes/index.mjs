import express from "express";
const router = express.Router();

import ctrlTokens from "../controllers/tokens.mjs";
import ctrlAgent from "../controllers/agent.mjs";
import ctrlNetwork from "../controllers/network.mjs";
import ctrlCapabilities from "../controllers/capabilities.mjs";
import ctrlDelegations from "../controllers/delegations.mjs";
import ctrlVerification from "../controllers/verification.mjs";
import ctrlWarehouse from "../controllers/warehouse.mjs";
import ctrlDIDCommV1 from "../controllers/ssiComm.mjs";

// CLASSIC ISHARE //
// Participants routes (functionalities available for all participants)
router.post("/connect/token", ctrlVerification.verifyClientAssertion, ctrlVerification.verifyCertificate, ctrlVerification.verifyNetworkStatus, ctrlTokens.getToken);
router.get("/capabilities", ctrlVerification.verifyAccessToken, ctrlCapabilities.getCapabilities);

// Agent routes
router.get("/agent/info", ctrlVerification.verifyOrigin, ctrlAgent.getInformation);
router.post("/agent/assertion", ctrlVerification.verifyOrigin, ctrlAgent.generateClientAssertion);
router.post("/agent/token", ctrlVerification.verifyOrigin, ctrlAgent.generateToken);

// Network routes
router.get("/trusted_list", ctrlVerification.verifySatellite, ctrlVerification.verifyAccessToken, ctrlNetwork.getTrustedList);
router.get("/parties", ctrlVerification.verifySatellite, ctrlVerification.verifyAccessToken, ctrlNetwork.getParties);

// Authorization Registry routes
router.post("/delegation", ctrlVerification.verifyAR, ctrlVerification.verifyAccessToken, ctrlDelegations.getDelegationEvidence);

// Service Provider routes
router.post("/warehouse/SLOAir", ctrlVerification.verifyProvider, ctrlVerification.verifyAccessToken, ctrlVerification.verifyDelegation, ctrlWarehouse.getAirplanesData); // TODO LAST 2


// SSI ISHARE //
// Participants routes (functionalities available for all participants)
router.get(/^\/(.+)\/did.json$/, ctrlDIDCommV1.getDDO);
router.get('/resolve', ctrlDIDCommV1.resolve);
router.post('/didcomm', ctrlVerification.verifyOrigin, ctrlDIDCommV1.sendMessage);
router.post('/did/message', ctrlDIDCommV1.handleMessage);

// Agent routes
router.get('/agent/connections/ssi', ctrlVerification.verifyOrigin, ctrlAgent.getConnections);
router.get('/agent/didDocs/ssi', ctrlVerification.verifyOrigin, ctrlAgent.getIdentifiers);
router.post("/agent/assertion/ssi", ctrlVerification.verifyOrigin, ctrlAgent.generateClientAssertionSSI);

export default router;