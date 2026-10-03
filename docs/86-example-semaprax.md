# Semaprax policy-conformance evaluation

The [Semaprax example]({{ src('examples/semaprax/README.md') }}) demonstrates a small,
offline policy-conformance reward that combines a typed tool proposal, a real
Semaprax Agent Runtime decision, and an externally observed dispatch.

It includes three reproducible cases: an authorized dispatch, a rejected
proposal that is not dispatched, and a rejected proposal followed by an
explicitly labeled external fault injection. Task outcome and policy
conformance are evaluated independently, and the validated metrics can be
published through the standard Agent Lightning rollout event and reward APIs.

The example is intentionally limited to one fixed tool shape. See its README
for the scoring rule, trust boundary, capture instructions, and upstream
attribution.
