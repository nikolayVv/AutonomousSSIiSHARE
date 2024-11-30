import forge from "node-forge";

const convertFromCertToX5C = (cert) => {
    cert = cert.replace(/\\n/g, '\n');
    return forge.util.encode64(
        forge.asn1
            .toDer(forge.pki.certificateToAsn1(forge.pki.certificateFromPem(cert)))
    .getBytes()
    )
}

const convertFromX5CToCert = (x5c) => {
    const x5cStringified = JSON.stringify(x5c);
    const begin_cert = '-----BEGIN CERTIFICATE-----\n';
    const end_cert = '\n-----END CERTIFICATE-----';
    const result = x5cStringified.substring(2, x5cStringified.length - 2);


    return begin_cert.concat(result, end_cert);
}

const generatePrivateKeyHex = (private_key) => {
    const rsaPrivateKey = private_key.replace('-----BEGIN RSA PRIVATE KEY-----', '')
        .replace('-----END RSA PRIVATE KEY-----', '')
        .replace(/\n/g, '');

    const rsaPrivateKeyBuffer = Buffer.from(rsaPrivateKey, 'base64');

    return  rsaPrivateKeyBuffer.toString('hex');
}

export {
    convertFromCertToX5C,
    convertFromX5CToCert,
    generatePrivateKeyHex
}