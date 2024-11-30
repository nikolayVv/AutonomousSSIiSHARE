import JSONModel from "../models/model.mjs";
const jsonFile = new JSONModel(`../data/agent${process.env.PORT}.json`);

import * as tokenService from "../services/tokenService.mjs";
import * as certificateService from "../services/certificateService.mjs";

const Delegations = {
    async getDelegationEvidence(req, res) {
        const reqBody = await req.body;
        const client_id = await req.client_id;
        const epoch = Math.floor(new Date() / 1000);

        // Check the request attributes
        const delegationRequest = reqBody.delegationRequest;
        if (!delegationRequest) {
            return res.status(400).json({message: 'Request has no delegationRequest.'});
        }
        const policyIssuer = delegationRequest.policyIssuer;
        if (!policyIssuer) {
            return res.status(400).json({ message: 'Delegation request has no policyIssuer.' });
        }
        const target = delegationRequest.target;
        if (!target) {
            return res.status(400).json({ message: 'Delegation request has no target.' });
        }
        const accessSubject = target.accessSubject;
        if (!accessSubject) {
            return res.status(400).json({ message: 'Delegation request has no accessSubject.' });
        }
        if (accessSubject !== client_id) {
            return res.status(400).json({ message: 'Delegation request has invalid accessSubject.' });
        }
        const requested_policySets = delegationRequest.policySets;
        if (!requested_policySets || requested_policySets.length === 0) {
            return res.status(400).json({ message: 'Delegation request has no policySets.' });
        }


        const result = await jsonFile.load();
        if (result.error) {
            return res.status(500).json({ message: result.error });
        }

        const delegations = result.data.delegations || [];
        const delegation = delegations.find(del => del.policyIssuer === policyIssuer && del.target.accessSubject === accessSubject);
        let delegationEvidence = {
            "notBefore": delegation.notBefore,
            "notOnOrAfter": delegation.notOnOrAfter,
            "policyIssuer": delegation.policyIssuer,
            "target": delegation.target,
            "policySets": [],
        };

        for (let i = 0; i < requested_policySets.length; i++) {
            const requested_policySet = requested_policySets[i];

            for (let j = 0; j < delegation.policySets.length; j++) {
                const policySet = delegation.policySets[j];
                const evidence_policies = [];
                for (let k = 0; k < requested_policySet.policies.length; k++) {
                    const requested_policy = requested_policySet.policies[k];

                    if (!requested_policy.target) {
                        return res.status(400).json({message: 'Requested policy has no target.'});
                    }
                    const resource = requested_policy.target.resource;
                    if (!resource) {
                        return res.status(400).json({message: 'Requested policy has no resource.'});
                    }
                    const type = resource.type;
                    if (!type) {
                        return res.status(400).json({message: 'Requested policy has no type.'});
                    }
                    const identifiers = resource.identifiers;
                    if (!identifiers || identifiers.length === 0) {
                        return res.status(400).json({message: 'Requested policy has no identifiers.'});
                    }
                    const policy = policySet.policies.find(pol => pol.target.resource.type === type);
                    if (policy) {
                        let valid_identifiers = true;
                        if (!policy.target.resource.identifiers.includes("*")) {
                            for (const identifier of identifiers) {
                                if (!policy.target.resource.identifiers.includes(identifier)) {
                                    valid_identifiers = false;
                                    break;
                                }
                            }
                        }

                        const attributes = resource.attributes;
                        let valid_attributes = true;
                        if (!attributes) {
                            requested_policy.target.resource.attributes = policy.target.resource.attributes;
                        } else if (valid_identifiers) {
                            if (!policy.target.resource.attributes.includes("*")) {
                                for (const attribute of attributes) {
                                    if (!policy.target.resource.attributes.includes(attribute)) {
                                        valid_attributes = false;
                                        break;
                                    }
                                }
                            }
                        }

                        const actions = requested_policy.target.actions;
                        let valid_actions = true;
                        if (!actions) {
                            requested_policy.target.actions = policy.target.actions;
                        } else if (valid_identifiers && valid_attributes) {
                            if (!policy.target.actions.includes("*")) {
                                for (const action of actions) {
                                    if (!policy.target.actions.includes(action)) {
                                        valid_actions = false;
                                        break;
                                    }
                                }
                            }
                        }

                        const environment = requested_policy.target.environment;
                        let valid_environment = true;
                        if (!environment || !environment.serviceProviders) {
                            requested_policy.target.environment = policy.target.environment;
                        } else if (valid_identifiers && valid_attributes && valid_actions) {
                            if (!policy.target.environment.serviceProviders.includes("*")) {
                                for (const serviceProvider of environment.serviceProviders) {
                                    if (!policy.target.environment.serviceProviders.includes(serviceProvider)) {
                                        valid_environment = false;
                                        break;
                                    }
                                }
                            }
                        }

                        let valid_type = true;
                        if (valid_identifiers && valid_attributes && valid_actions && valid_environment) {
                            for (const rule of policy.rules) {
                                if (rule.effect === "Deny") {
                                    if (rule.target.resource.type === requested_policy.target.resource.type) {
                                        valid_type = false;
                                        break;
                                    }

                                    if (rule.target.resource.identifiers.includes("*")) {
                                        valid_identifiers = false;
                                        break;
                                    } else {
                                        for (const identifier of identifiers) {
                                            if (rule.target.resource.identifiers.includes(identifier)) {
                                                valid_identifiers = false;
                                                break;
                                            }
                                        }
                                        if (!valid_identifiers) {
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        let evidence_policy = requested_policy;
                        if (valid_identifiers && valid_attributes && valid_actions && valid_environment && valid_type) {
                            evidence_policy.rules = [
                                {
                                    "effect": "Permit"
                                }
                            ]
                        } else {
                            evidence_policy.rules = [
                                {
                                    "effect": "Deny"
                                }
                            ]
                        }
                        evidence_policies.push({
                            ...evidence_policy
                        });
                    }
                }
                delegationEvidence.policySets.push({
                    "maxDelegationDepth": policySet.maxDelegationDepth,
                    "target": policySet.target,
                    "policies": evidence_policies
                });
            }
        }

        if (delegationEvidence.policySets.length === 0) {
            res.status(400).json({ message: "Couldn't find a corresponding delegation evidence." });
        } else {
            const headers = await req.headers;
            if (!headers["do-not-sign"]) {
                const x5c = certificateService.convertFromCertToX5C(process.env.CERT);
                const delegation_token = tokenService.generateDelegationToken(process.env.EORI, x5c, accessSubject, delegationEvidence, process.env.PRIVATE_KEY);

                res.status(200).json({ delegation_token: delegation_token });
            } else {
                res.status(200).json({ delegationEvidence: delegationEvidence });
            }

        }
    }
}

export default Delegations;