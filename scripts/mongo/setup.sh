#!/bin/bash

set -euo pipefail

# Install MongoDB
echo "Installing MongoDB..."
sudo apt-get install -y gnupg curl
curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg \
   --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.2.list
sudo apt-get update
sudo apt-get install -y mongodb-org
# Check if MongoDB is installed successfully
command -v mongod >/dev/null 2>&1 || { echo "mongod not found"; exit 1; }
command -v mongosh >/dev/null 2>&1 || { echo "mongosh not found"; exit 1; }
echo "MongoDB installed successfully!"

# Start MongoDB
echo "Starting MongoDB..."
mkdir -p /tmp/rs0
mongod --replSet rs0 --port 27017 --dbpath /tmp/rs0 --bind_ip 127.0.0.1 --fork --logpath /tmp/rs0/mongod.log

# simple wait until it's up
for i in {1..30}; do
  if mongosh --port 27017 --eval 'db.runCommand({ ping: 1 })' >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "MongoDB started successfully!"

# Initialize MongoDB replica set
echo "Initializing MongoDB replica set..."
mongosh --port 27017 --eval 'rs.initiate({_id: "rs0", members: [{_id: 0, host: "localhost:27017"}]})'
echo "MongoDB replica set initialized successfully!"
