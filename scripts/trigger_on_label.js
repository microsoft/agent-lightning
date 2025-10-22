// Copyright (c) Microsoft. All rights reserved.

module.exports = function triggerOnLabel({ core, context, labelName }) {
  if (!labelName) {
    throw new Error("labelName is required");
  }

  if (context.eventName !== "pull_request") {
    core.setOutput("should-run", "true");
    core.notice("Triggering this workflow because event is not a pull request");
    return;
  }

  const action = context.payload.action;
  const labels = (context.payload.pull_request?.labels ?? []).map(
    (label) => label.name
  );

  if (["reopened", "ready_for_review"].includes(action)) {
    if (labels.includes(labelName)) {
      core.setOutput("should-run", "true");
      core.notice(
        `Triggering this workflow because pull request is reopened or ready for review and has the '${labelName}' label.`
      );
    } else {
      core.setOutput("should-run", "false");
      core.notice(
        `Skipping because pull request is missing the '${labelName}' label.`
      );
    }
    return;
  }

  if (action === "labeled") {
    if ((context.payload.label?.name ?? "") === labelName) {
      core.setOutput("should-run", "true");
      core.notice(
        `Triggering this workflow because pull request is labeled with the '${labelName}' label.`
      );
    } else {
      core.setOutput("should-run", "false");
      core.notice(
        `Skipping because label '${
          context.payload.label?.name ?? "unknown"
        }' does not match '${labelName}'.`
      );
    }
    return;
  }

  core.setOutput("should-run", "false");
  core.notice(
    `Skipping this workflow because action '${action}' is not handled. ` +
      `This workflow only triggers on: 'reopened', 'ready_for_review', or 'labeled' actions.`
  );
};
