# Changelog

All notable changes to DSF are documented here.

## [Unreleased]

### Added
- Dedicated `tox -e benchmarks` environment for running scientific benchmarks and regenerating benchmark figures.
- Documentation for running the scientific benchmark suite.

### Changed
- Migrated tomography integration to Binny v2 (`pybinny>=2.0.0`).
- Updated Binny imports, tomography metadata handling, and predefined bin normalization.
- Updated compatibility tests, documentation examples, and benchmark scripts.
- Regenerated documentation forecast outputs and scientific benchmark figures.
- Condensed and reorganized `pyproject.toml`.
- Updated the dependency lockfile to Binny v2.

### Fixed
- Compatibility with the reorganized Binny v2 public API.
- Population fraction metadata handling for predefined tomographic bins.

## [0.2.0] - 2026-11-12

### Added
- Package import regression tests.
- Updated redshift-distribution inputs for legacy covariance benchmarks.

### Changed
- Migrated tomography integration to the new Binny API (`pybinny>=1.2.0`).
- Updated tomography configuration, metadata handling, documentation, tests, and benchmarks.
- Raised the minimum supported DerivKit version to `1.5.0`.
- Applied Ruff formatting throughout the codebase.

### Fixed
- Compatibility with updated Binny survey presets and binning configuration.
- Legacy covariance benchmark compatibility following the Binny migration.