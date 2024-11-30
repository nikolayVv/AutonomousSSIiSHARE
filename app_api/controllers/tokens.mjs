import * as tokenService from "../services/tokenService.mjs";
import * as certificateService from "../services/certificateService.mjs";
import * as satelliteService from "../services/satelliteService.mjs";
import JSONModel from "../models/model.mjs";
const jsonFile = new JSONModel(`../data/agent${process.env.PORT}.json`);

import jwt from "jsonwebtoken";
import querystring from "querystring";
import request from "request-promise";

const Tokens = {
    async getToken(req, res){
       const reqBody = await req.body;
       const x5c = certificateService.convertFromCertToX5C(process.env.CERT);
       const access_token = tokenService.generateAccessToken(process.env.EORI, x5c, reqBody.client_id, process.env.PRIVATE_KEY);

       res.status(200).json({
          access_token: access_token,
          expires_in: 3600,
          token_type: 'Bearer'
       });
    }
};

export default Tokens;