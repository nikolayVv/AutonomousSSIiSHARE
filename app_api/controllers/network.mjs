import JSONModel from "../models/model.mjs";
const jsonFile = new JSONModel(`../data/agent${process.env.PORT}.json`);

import * as certificateService from "../services/certificateService.mjs";
import * as satelliteService from "../services/satelliteService.mjs";

const Network = {
    async getTrustedList(req, res){
        const result = await jsonFile.load();
        if (result.error) {
            return res.status(500).json({ message: result.error });
        }
        const trusted_list = result.data.trusted_list;

        res.status(200).json({ count: trusted_list.length, data: trusted_list });
    },

    async getParties(req, res) {
        const eori = await req.query.eori;
        if (!eori) {
            return res.status(400).json({message: 'Request has no eori parameter.'});
        }

        const result = await jsonFile.load();
        if (result.error) {
            return res.status(500).json({ message: result.error });
        }

        const parties = result.data.parties;
        let party = null;
        let satellites = [];
        for (let i = 0; i < parties.length; i++) {
            if (parties[i].id === eori) {
                party = parties[i];
                break;
            }
            if (parties[i].certifications) {
                for (let j = 0; j < parties[i].roles.length; j++) {
                    if (parties[i].roles[j].role === "SchemeOwner") {
                        satellites.push(parties[i])
                    }
                }
            }
        }

        if (party) {
            const status = party.status
            if (status !== 'Active') {
                return res.status(401).json({ message: 'The participant is not active in the network.' });
            }
            let certifications = [];
            if (party.certifications) {
                certifications = party.certifications;
            }
            party = {
                "count": 1,
                "data": [
                    {
                        "party_id": party.id,
                        "party_name": party.name,
                        "url": party.url,
                        "adherence": {
                            "status": party.status,
                            "start_date": party.start_date,
                            "end_date": party.end_date
                        },
                        "certifications": certifications,
                        "roles": party.roles
                    }
                ]
            };
        } else {
            const x5c = certificateService.convertFromCertToX5C(process.env.CERT);
            party = await satelliteService.findPartyInOtherSatellites(
                process.env.EORI,
                x5c,
                eori,
                process.env.PRIVATE_KEY,
                satellites,
                result.data.access_tokens || []
            );
        }

        if (!party) {
            res.status(401).json({ message: "Couldn't find party with the corresponding EORI in the network." });
        } else {
            res.status(200).json(party);
        }
    }
};

export default Network;