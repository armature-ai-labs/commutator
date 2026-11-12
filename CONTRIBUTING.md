# Contributing to Commutator

Thanks for helping. A few rules keep the gateway trustworthy.

## Before you open a pull request

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --workspace
```

All three must pass; CI runs them on x86_64 and on ARM64 (the architecture of NVIDIA Jetson boards).

## Rules

1. **Parity with solo-router stays green.** `tests/parity` replays solo-router's own decisions. If you change routing behaviour on purpose, say so in the PR, and explain why it should differ.
2. **Safety changes need tests.** Anything touching stop handling, guardrails or the fail-safe needs a test showing the behaviour.
3. **No invented numbers.** Latency, power and quality figures must be measured or clearly marked illustrative.
4. **Plain errors.** Error messages say what went wrong and how to fix it.

## Sign-off

Commits need a Developer Certificate of Origin sign-off:

```bash
git commit -s -m "Short description"
```

This adds `Signed-off-by: Your Name <you@example.com>`, confirming you wrote the change or have the right to submit it under Apache 2.0. See https://developercertificate.org/.
