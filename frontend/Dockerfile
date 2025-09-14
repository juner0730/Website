FROM node:18-alpine

WORKDIR /App

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

CMD ["npm", "run", "dev", "--", "--host"]