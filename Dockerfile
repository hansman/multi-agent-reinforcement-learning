FROM node:10.15.3

# Install app
COPY . /app/
WORKDIR /app

RUN npm install

ENTRYPOINT ["npm", "start"]
