// Copyright (c) Microsoft. All rights reserved.

module.exports = async function dispatchComment({ core, github, context }) {
  const payload = context.payload.client_payload || {};
  const pullNumber = payload.pull_number;

  if (!pullNumber) {
    core.notice(
      "No pull_number found in repository_dispatch payload; skipping comment."
    );
    return;
  }

  const { owner, repo } = context.repo;
  const action =
    context.payload.action || payload.ci_label || context.eventName || "dispatch";
  const runId = process.env.GITHUB_RUN_ID;
  const workflowName = context.workflow || "Workflow";
  const runUrl = `https://github.com/${owner}/${repo}/actions/runs/${runId}`;

  const body = `🚀 ${workflowName} dispatched via \`${action}\`. Track progress here: ${runUrl}`;

  await github.rest.issues.createComment({
    owner,
    repo,
    issue_number: pullNumber,
    body,
  });
};
