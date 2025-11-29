// Copyright (c) Microsoft. All rights reserved.

/**
 * Aggregates the “latest completed” results of several dependent workflows and fails
 * this step if any required job/variant is missing or not successful.
 *
 * Usage (from actions/github-script@v8):
 *
 *   const badgeAggregation = require('./scripts/badge_aggregation.js');
 *   const dependencies = [
 *     { workflow: 'examples-calc-x.yml', label: 'calc-x.latest', variants: ['latest'] },
 *     { workflow: 'examples-spider.yml', label: 'spider.latest', variants: ['latest'] },
 *     { workflow: 'examples-apo.yml', label: 'apo.latest', variants: ['latest'] },
 *     { workflow: 'examples-unsloth.yml', label: 'unsloth.latest', variants: ['latest'] },
 *     { workflow: 'tests-full.yml', label: 'tests-full.latest', variants: ['latest'] },
 *   ];
 *   await badgeAggregation({ github, context, core, dependencies });
 *
 * Notes:
 * - Requires the workflow files above to exist in .github/workflows/.
 * - Looks at the default branch "main" unless you override per dependency with dep.branch.
 * - Assumes matrix job names contain the variant in parentheses, e.g. "tests (latest)".
 */
module.exports = async function badgeAggregation({ github, context, core, dependencies }) {
  const failures = [];

  // Defensive: validate inputs early for nicer error messages.
  if (!github?.rest?.actions || !context?.repo || !core) {
    throw new Error('badgeAggregation: expected { github, context, core } from actions/github-script.');
  }
  if (!Array.isArray(dependencies) || dependencies.length === 0) {
    core.info('No dependencies provided; nothing to check.');
    return;
  }

  // Helper: paginate jobs for a run attempt (handles >100 jobs edge case).
  async function listAllJobsForAttempt(run_id, attempt_number) {
    const all = [];
    let page = 1;
    while (true) {
      const { data } = await github.rest.actions.listJobsForWorkflowRunAttempt({
        owner: context.repo.owner,
        repo: context.repo.repo,
        run_id,
        attempt_number,
        per_page: 100,
        page,
      });
      const jobs = data.jobs ?? [];
      all.push(...jobs);
      if (!data.total_count || all.length >= data.total_count || jobs.length === 0) break;
      page += 1;
    }
    return all;
  }

  // For each dependency: find the latest completed run on the target branch; then inspect its jobs.
  for (const dep of dependencies) {
    const branch = dep.branch || 'main';

    // You can pass the workflow file name as workflow_id (e.g. "examples-apo.yml").
    const { data: runsData } = await github.rest.actions.listWorkflowRuns({
      owner: context.repo.owner,
      repo: context.repo.repo,
      workflow_id: dep.workflow,
      branch: 'main', // Always check the main branch status no matter what
      status: 'completed', // only completed runs
      per_page: 50, // retrieve latest 50 so we can filter
      sort: 'created',
      direction: 'desc',
    });

    const filteredRuns = runsData?.workflow_runs?.filter(run => ['schedule', 'workflow_dispatch'].includes(run.event));

    const run = filteredRuns?.[0];
    if (!run) {
      failures.push(`No completed run found for ${dep.label} on branch "${branch}"`);
      continue;
    }

    core.info(`[${dep.label}] Found run ${run.id} with attempt ${run.run_attempt}`);
    // Get the specific attempt we want to inspect (latest attempt for that run).
    const attempt = run.run_attempt ?? 1;

    // Robust: paginate jobs in case the workflow has many.
    const jobs = await listAllJobsForAttempt(run.id, attempt);
    core.info(`[${dep.label}] Found ${jobs.length} jobs: ${jobs.map(j => j.name).join(', ')}`);

    // Match each required variant to a job. We look for the variant in parentheses, e.g. "(latest)".
    for (const variant of dep.variants || []) {
      const matchingJobs = jobs.filter(
        j => typeof j.name === 'string' && j.name.includes(variant)
      );

      if (matchingJobs.length === 0) {
        failures.push(`Missing job for ${dep.label} (variant: ${variant})`);
        continue;
      }

      for (const job of matchingJobs) {
        core.info(`[${dep.label}] ${job.name} => ${job.conclusion}`);

        // Accept only a strict "success".
        if (job.conclusion !== 'success') {
          failures.push(`${dep.label} (${job.name}) concluded ${job.conclusion}`);
        }
      }
    }
  }

  // Surface aggregated result to the workflow.
  if (failures.length) {
    core.setFailed(failures.join(' | '));
  } else {
    core.info('All latest variants succeeded.');
  }
};
