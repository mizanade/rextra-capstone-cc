# Use the official Node.js image as the base image
FROM node:20

# Set the working directory to /app
WORKDIR /app

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install dependencies
RUN yarn install

# Copy the application code to the working directory
COPY . .

# Expose the port that your application will run on
EXPOSE 3000

# Command to run your application
CMD ["node", "src/index.js"]