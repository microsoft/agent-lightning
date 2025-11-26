// Copyright (c) Microsoft. All rights reserved.

// MongoDB replica set initialization script.
// Use this if you are accessing MongoDB from the **host**.

rs.initiate({
  _id: "rs0",
  members: [{ _id: 0, host: "localhost:27017" }],
});
