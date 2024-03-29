# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [4.2.1] - 2022-09-10

## Changed
- Callbacks have been reorganized in folders.
- Update requirements.
- Update `gh-pages.yml`.
- Update `README.md`
- Update `mixed_precision` parameter and usage.

## Fixed
- Progress bar will only be created under the main process.
- Logging now is handled by `accelerate` library.


## [4.2.0] - 2022-08-08

## Added
- Add `torch.ditribution` example, with code taken from [Romain Strock](https://romainstrock.com/blog/modeling-uncertainty-with-pytorch.html).
- Add `predict` method to `Trainer`. #38
- Add functions to freeze and unfreeze model. #43
- Add function to transform dataset into time series dataset.

## Fixed
- Metrics are now moved to the execution device #41.
- Log level is now used in the Trainer. #40
- `LearningRateScheduler` now does not crash in first epoch when `on_train` is False. #36

## Changed
- Make regularization part of the callbacks system. #37 
- Divide utils into three submodules: `convenience`,`preprocessing` and `data`.
- Update requirements to avoid conflicts.
- Update some tests.

## Removed

- Remove old regularization module and all related code.


## [4.1.2] - 2021-12-24

### Fixed

- Fix PyPi deployment file.


## [4.1.1] - 2021-12-24

### Added

- Add PyPi deployment to the CI/CD.

### Fixed

- Fix `CHANGELOG.md` release dates.

## [4.1.0] - 2021-12-24

### Added

- Add possibility to set the log level of the callbacks.
- Add stochastic weight averaging callback.
- Add `train_test_val_split`.
- Add `log_name` attribute to `tochfitter.callbacks.base.Callback`.

### Changed

- Change `with torch.no_grad()` for `@torch.no_grad()` in trainer.
- Format code with [Black](https://github.com/psf/black).
- Reorganize `utils` module.

### Removed 

- Remove `reset_parameters` method from callbacks.

### Fixed

- Fix `RichProgressBar` not logging appropiate values.
- Fix log level not being correctly set.


## [4.0.0] - 2021-12-16

### Added

- Add more hooks to the callback system.
- Rich progress bar as callback.
- accelerate.Accelerator backend.
- `trainer.Trainer.fit` now returns a dictionary with the train history.

### Changed

- Update README.
- Update metrics handling.

### Removed 

- Remove callback type.


## [3.1.0] - 2021-07-27

### Fixed

- Solve doc typos.
- Fix logger and trainer tests.
- Fix incomplete `quickstart` in docs.
- Fix logging bug in `GPUStats` callback.

### Added

- Add support for mixed precision training.
- Add ElasticNet regularization.
- Add testing methods and their tests: `check_monotonically_decreasing` and `compute_forward_gradient`.
- Add cuda seed setting in Manager.
- Add option to only use deterministic algorithm in the Manager class.

### Changed

- Update logo and README.
- Update tests with new testing methods.
- Make some method on Trainer and Manager private.


## [3.0.0] - 2021-07-27

### Fixed

- Solve bug in callbacks where the handler was not calling in appropiate order.

### Removed
- Remove `ElasticNet` regularization because the implementation was not correct.

### Changed

- Change `params_dict` in the Trainer to a specific class that tracks the internal state.
- Change README.
- Update tests.
- Change logic of TQDM to be updated in each batch instead of in each epoch.
- Change optimization loop to be of type `condition-loop` instead of `iteration-loop`. This is, the loop is now a `while` loop.

### Added

- Add `Manager` class to handle multiple experiments.
- Add support for computing metrics in the optimization loop via `torchmetrics`.
- Add `GPUStats`, `ReduceLROnPlateau` and `ProgressBarLogger` callbacks.
- Add testing utility to check gradients: `compute_forward_gradient`.
- Add more functions to `utils`: `FastTensorDataLoader`, `check_model_on_cuda`.


## [2.0.2] - 2021-05-10

### Fixed

- Solve warning where learning rate scheduler was being called before loss.

### Changed

- Change `_compute_penalty` in favour of `compute_penalty`.
- Change `_train` in favour of `train_step`.
- Change `_validate` in favour of `validation_step`.
- Update tests to be correct.


## [2.0.1] - 2021-04-29

### Added

- Added new `reset_parameters` method in the trainer.
- Added requirements file for example.
- Added `trainer` example in `.py` format.
- Added `manager.ipynb` example.

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