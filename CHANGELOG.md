# Changelog

This file records user-visible changes to SynapticDB.

## 0.1.1 - 2026-08-21

SynapticDB 0.1.1 contains intentional breaking API changes from 0.1.0. This
release does not include compatibility aliases for the old names.

### Added

- Added `SynapticDB.get()` for exact memory lookup by UUID or UUID string.
- Added `to_dict()` and `to_json()` to `Memory`, `RecalledMemory`,
  `RecallResult`, and `Stats`.
- Added UUID string support to `connect()`, `feedback()`, and `forget()`.
- Added frozen benchmark rankings for repeatable comparisons with the locked
  baseline without loading the baseline models.
- Added a tested OpenAI Responses API agent example to the README.

### Changed

| 0.1.0 API | 0.1.1 API |
| --- | --- |
| `Synaptic` | `SynapticDB` |
| `remember()` | `store()` |
| `Recalled` | `RecalledMemory` |
| `RecallResult.associative` | `RecallResult.association_results` |

- Flattened recalled memory results. Use `result.memories[0].content` instead
  of `result.memories[0].memory.content`.
- Changed the README quickstart to use a persistent `synaptic.db` database.

### Fixed

- No separate user-visible defect fixes are included in this release.

### Packaging

- Updated the package version to 0.1.1.
- Excluded the test suite from the source distribution.
