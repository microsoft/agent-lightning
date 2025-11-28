// Copyright (c) Microsoft. All rights reserved.

// MongoDB replica set initialization script.
// Use this if you are accessing MongoDB from another **container**.
// `mongodb_init_rs_host.js` is the counterpart if accessing from the host.

rs.initiate({
  _id: "rs0",
  members: [{ _id: 0, host: "mongo:27017" }],
});

db.setProfilingLevel(2);
