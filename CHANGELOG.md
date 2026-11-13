# Changelog

All notable changes to DSF are documented here.

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
