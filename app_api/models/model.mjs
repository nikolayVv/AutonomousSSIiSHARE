import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";

class JSONModel {
    constructor(filePath) {
        const __dirname = path.dirname(fileURLToPath(import.meta.url));
        this.filePath = path.join(__dirname, filePath);
        this.data = [];
    }

    async load() {
        try {
            const jsonData = await fs.readFile(this.filePath, 'utf8');
            this.data = JSON.parse(jsonData);
            return { error: null, data: this.data };
        } catch (error) {
            console.error(error.message);
            return { error: 'Error loading the data.', data: null };
        }
    }

    async save() {
        try {
            await fs.writeFile(this.filePath, JSON.stringify(this.data, null, 2), 'utf8');
            return { error: null, data: 'Data saved successfully.'}
        } catch (error) {
            console.error(error.message)
            return { error: 'Error saving the data.', data: null };
        }
    }

    connectionExists(did) {
        if (this.data.did_connections && this.data.did_connections.length > 0) {
            return this.data.did_connections.some(existingConnection => existingConnection.did === did);
        }

        return false;
    }

    connectionIsValid(did) {
        if (this.data.did_connections && this.data.did_connections.length > 0) {
            return this.data.did_connections.some(existingConnection => existingConnection.did === did && existingConnection.status === "Active");
        }

        return false;
    }

    async addDidConnection(did, eori) {
        if (!this.data.did_connections) {
            this.data.did_connections = [];
        }
        const connection = {
            did: did,
            eori: eori,
            status: "Active"
        };

        let error = null;
        if (!this.connectionExists(connection.did)) {
            this.data.did_connections.push(connection);
            const result = await this.save();
            error = result.error;
        } else {
            console.log("Connection already exists");
        }


        return { error: error, data: connection };
    }

    async addCredential(credential) {
        if (!this.data.credentials) {
            this.data.credentials = [];
        }

        let error = null;

        this.data.credentials.push(credential);
        const result = await this.save();
        error = result.error;

        return { error: error, data: credential };
    }

    messageExists(id) {
        if (this.data.didcomm_messages && this.data.didcomm_messages.length > 0) {
            return this.data.didcomm_messages.some(existingMessage => existingMessage.id === id);
        }

        return false;
    }

    async addMessage(received_at, id, message) {
        if (!this.data.didcomm_messages) {
            this.data.didcomm_messages = [];
        }
        const didcomm_message = {
            received_at: received_at,
            id: id,
            message: message
        };

        let error = null;
        if (!this.messageExists(id)) {
            this.data.didcomm_messages.push(didcomm_message);
            const result = await this.save();
            error = result.error;
        } else {
            console.log("Message already exists");
        }


        return { error: error, data: message };
    }

    async storeAccessToken(access_token, eori) {
        if (!this.data) {
            this.load();
        }
        if (!this.data.access_tokens) {
            this.data.access_tokens = [];
        }
        const token = {
            eori: eori,
            access_token: access_token
        }

        const index = this.data.access_tokens.findIndex(token => token.eori === eori);
        if (index !== -1) {
            this.data.access_tokens[index] = token;
        } else {
            this.data.access_tokens.push(token);
        }
        const result = await this.save();

        return { error: result.error, data: token };
    }

    findAccessTokenByEori(eori) {
        if (this.data.access_tokens) {
            return this.data.access_tokens.find(token => token.eori === eori);
        }
        return null;
    }

    findPartyById(id) {
        for (let i = 0; i < this.data.parties.length; i++) {
            if (this.data.parties[i].id === id) {
                return this.data.parties[i];
            }
        }

        return null;
    }

    isSpecificRole(definedRole) {
        for (let i = 0; i < this.data.roles.length; i++) {
            if (this.data.roles[i].role === definedRole) {
                return true;
            }
        }

        return false;
    }

    async create(item) {
        this.data.push(item);
        await this.save();

        return { error: null, data: item };
    }
}

export default JSONModel;