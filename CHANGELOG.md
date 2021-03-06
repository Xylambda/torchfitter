# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [2.0.01] - 2021-04-29

### Added

- Added new `reset_parameters` method in the trainer.
- Added requirements file for example.

### Fixed

- Fix error in setup naming.
- Fix moving the tensors to device. Now, it is done in each batch.

### Changed

- Change the `requirements.txt` to remove unnecessary dependencies.


## [1.0.0] - 2021-01-08

### Added

- Added possibility to use L1 and ElasticNet [regularization](https://github.com/Xylambda/torchfitter/pull/3).
- Added new testing module.
- Added tests for the new functionalities.

### Changed

- Updated README to add brief tutorial on how to create regularization algos.
- Updated tests for trainer.

### Fixed

- Fixed minor typos in README.


## [0.2.0] - 2021-01-07

### Added

- Added badges from [shield.io](https://shields.io/)


## [0.1.1] - 2021-01-07

### Added

- Added a CHANGELOG.md

### Fixed

- Fixed error in README example syntax.