// Copyright (c) Microsoft. All rights reserved.

function isTriggeredByComment(core, context) {
  if (context.eventName !== "issue_comment") {
    core.notice("[comment check] This workflow is not triggered by a comment.");
    return false;
  }

  const action = context.payload.action;
  const issue = context.payload.issue;
  const commentBody = context.payload.comment?.body ?? "";

  // We only react to new comments on PRs.
  const isPr = !!issue?.pull_request;
  if (action !== "created" || !isPr) {
    core.notice(
      `[comment check] Action '${action}' is not 'created' or the comment is not on a pull request.`
    );
    return false;
  }

  // Require /ci prefix.
  if (!commentBody.trim().startsWith("/ci")) {
    core.notice("[comment check] Comment does not start with '/ci'.");
    return false;
  }

  core.notice(`[comment check] Comment starts with '/ci'`);
  return true;
};

module.exports = function triggerOnLabel({ core, context, labelName }) {
  if (!labelName) {
    throw new Error("labelName is required");
  }
  const triggeredByComment = isTriggeredByComment(core, context);

  if (!triggeredByComment && context.eventName !== "pull_request") {
    core.setOutput("should-run", "true");
    core.notice("Triggering this workflow because event is not a pull request");
    return;
  }

  core.notice(`This workflow is triggered by ${context.eventName}`);

  const labels = (context.payload.pull_request?.labels ?? context.payload.issue?.labels ?? []).map(
    (label) => label.name
  );
  core.notice(`Pull request labels: ${labels.join(", ")}`);

  if (labels.includes(labelName)) {
    core.setOutput("should-run", "true");
    core.notice(
      `Triggering this workflow because pull request has the '${labelName}' label.`
    );
  } else {
    core.setOutput("should-run", "false");
    core.notice(
      `Skipping because pull request is missing the '${labelName}' label.`
    );
  }
};
