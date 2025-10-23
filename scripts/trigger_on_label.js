// Copyright (c) Microsoft. All rights reserved.

module.exports = function triggerOnLabel({ core, context, labelName }) {
  if (!labelName) {
    throw new Error("labelName is required");
  }

  if (context.eventName !== "pull_request_target") {
    core.setOutput("should-run", "true");
    core.notice("Triggering this workflow because event is not a pull request");
    return;
  }

  core.notice(`This workflow is triggered by ${context.eventName}`);

  const labels = (context.payload.pull_request?.labels ?? []).map(
    (label) => label.name
  );
  core.notice(`Pull request labels: ${labels.join(", ")}`);

  if (labels.includes(labelName)) {
    core.setOutput("should-run", "true");
    core.notice(
      `Triggering this workflow because pull request has the '${labelName}' label.`
    );
  } else if (labels.includes("ci-all")) {
    core.setOutput("should-run", "true");
    core.notice(
      `Triggering this workflow because pull request has the 'ci-all' label.`
    );
  } else {
    core.setOutput("should-run", "false");
    core.notice(
      `Skipping because pull request is missing the '${labelName}' label.`
    );
  }
};
