FROM node:18-alpine

ARG NODE_ENV=development
ENV NODE_ENV=${NODE_ENV}

WORKDIR /usr/src/app

COPY package*.json ./
RUN npm install

COPY .. .

EXPOSE 3000 3030 3333 8000 8080 8081 8082 8083 8888

CMD [ "npm", "run", "start" ]