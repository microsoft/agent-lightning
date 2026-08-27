# Handling Failed Rollouts

Rollouts may fail due to transient system issues such as network errors, timeouts or external service failures.

## Retry behavior
- Rollout retries are configured via `RolloutConfig`, including settings such as `max_attempts`, retry conditions and timeouts.
- If a rollout fails and returns `None`, it still counts as an attempt and follows the configured retry limits.

## Batch behavior
- Failed rollouts are handled at the individual rollout level.
- There is currently no built-in mechanism to a automatically skip an entire batch when multiple rollouts fail.

## Best practices
- Retries are useful for transient failures (e.g. temporary network issues).
- If failures occur frequently, this usually indicates an infrastructure problem rather than an issue retries can fix.
- In such cases, it is recommended to address the underlying system issue instead of increasing retry limits.
