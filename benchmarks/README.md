# DSF Benchmarks

This directory contains benchmark scripts and outputs for validating DSF against legacy or reference implementations.

Benchmarks are not part of the normal unit-test suite. Unit tests live under `tests/`.

## Running the benchmarks

Run the scientific benchmark suite from the repository root:

```bash
tox -e benchmarks
```

This command:

- Regenerates the redshift distributions used by the legacy covariance benchmarks.
- Validates covariance matrices against the legacy implementation.
- Validates custom power-spectrum inputs and regenerates comparison figures.
- Checks lens-averaged power-spectrum numerical regressions.
- Validates the galaxy–matter window against its regression baseline.
- Validates Delta Sigma calculations and regenerates the comparison figure.

Existing legacy covariance references and numerical regression baselines are preserved.

Performance-only Hankel benchmarks and historical baseline generators can be run separately.

## Running individual benchmarks

Individual benchmark scripts can also be executed directly from the repository root using the active Python environment.