# Schnapsen RL Framework
Analyzer || Applied Project || Credential Project

![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-ml-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Status](https://img.shields.io/badge/status-research-orange)

A custom reinforcement-learning Schnapsen bot trained through asynchronous self-play.

## Table of contents
- [Overview](#overview)
- [Highlights](#highlights)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration notes](#configuration-notes)
- [Sizhong Zhang Contribution](#sizhong-zhang-contribution)
- [Ján Klačan Contribution](#ján-klačan-contribution)
- [License](#license)
- [Copyright and Ownership](#copyright-and-ownership)

## Overview
This project explores a lightweight training pipeline for Schnapsen using reinforcement learning and a simple MLP policy. The system uses asynchronous self-play, a replay buffer, and epsilon-greedy exploration. A supervised-learning pipeline is also included for dataset-driven baselines and comparisons.

## Highlights
- Asynchronous parallel game simulation with multiple worker processes
- Replay buffer for experience storage
- Epsilon decay for exploration control
- Lightweight MLP policy for move scoring
- Tournament scripts that generate win-rate matrices and statistical z-tests

## How it works
1. (Optional) Generate a replay dataset from Rdeep vs Rdeep games.
2. Train a supervised MLP baseline from the dataset.
3. Train an RL policy with asynchronous self-play and replay-buffer updates.
4. Run tournaments to produce win-rate matrices.
5. Post-process matrices with a z-test to label statistically significant wins.

## Repository layout
- [dataset_generator.py](dataset_generator.py): Parallel dataset generation using Rdeep bots.
- [Training/SL_training.py](Training/SL_training.py): Supervised MLP training on replay data.
- [Training/RL_training.py](Training/RL_training.py): Asynchronous RL training with replay buffer.
- [Tournament/tournament_base.py](Tournament/tournament_base.py): Baseline tournament (Random, Bully, Rdeep).
- [Tournament/tournament_SL.py](Tournament/tournament_SL.py): SL model tournaments.
- [Tournament/tournament_RL_depth.py](Tournament/tournament_RL_depth.py): RL models vs Rdeep at different depths.
- [Tournament/tournament_RL_samples.py](Tournament/tournament_RL_samples.py): RL models vs Rdeep at different sample counts.
- [Tournament/tournament_Rdeep_sx.py](Tournament/tournament_Rdeep_sx.py): Rdeep sample-count sweeps.
- [z_test.py](z_test.py): Z-test post-processing for win-rate matrices.

## Requirements
- Python 3.10+ (3.x supported)
- A working Schnapsen engine available to import as `schnapsen` (the scripts import `schnapsen.*` directly)
- Pinned Python dependencies in [requirements.txt](requirements.txt):

```text
torch==2.2.2
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
click==8.1.7
```

## Setup
1. Create and activate a virtual environment:
	```bash
	python -m venv .venv
	source .venv/bin/activate
	```
	Windows (PowerShell):
	```powershell
	.venv\Scripts\Activate.ps1
	```
	Windows (cmd.exe):
	```bat
	.venv\Scripts\activate.bat
	```
2. Install dependencies:
	```bash
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	```
3. Create data and model folders at the repo root:
	- `ML_data_hub/`
	- `ML_model_hub/`
4. Ensure the `schnapsen` package is importable (adjust `sys.path` in the training and tournament scripts if your environment differs).
5. On high-core machines or HPC servers, enable the thread-limiting block at the top of the training and tournament scripts to avoid thread explosion.

## Usage

### Generate a replay dataset (optional, for SL baselines)
```bash
python dataset_generator.py ml create-replay-dataset-parallel
```
The dataset filename and Rdeep depth are configured in [dataset_generator.py](dataset_generator.py).

### Train the supervised MLP baseline
```bash
python Training/SL_training.py
```
Configure the input dataset and output model paths in [Training/SL_training.py](Training/SL_training.py).

### Train the RL policy (asynchronous self-play)
```bash
python Training/RL_training.py
```
Adjust model/log paths, epsilon schedule, and Rdeep depth in [Training/RL_training.py](Training/RL_training.py).

### Run tournaments
```bash
python Tournament/tournament_base.py
python Tournament/tournament_SL.py
python Tournament/tournament_RL_depth.py
python Tournament/tournament_RL_samples.py
python Tournament/tournament_Rdeep_sx.py
```
Each script writes a win-rate matrix CSV (see its `MATRIX_FILE` setting).

### Run z-tests on tournament results
```bash
python z_test.py
```
Choose the input matrix and output name at the top of [z_test.py](z_test.py).

## Configuration notes
- RL training writes a CSV log with win-rate summaries; the filename is configured in [Training/RL_training.py](Training/RL_training.py).
- Tournament scripts expect trained models in `ML_model_hub/` and will raise errors if missing.
- The MLP input feature vector is 173 dimensions (from `get_state_feature_vector` + move features).
- The training and tournament scripts add three parent directories to `sys.path`; update these if your Schnapsen engine is located elsewhere.

## Sizhong Zhang Contribution
- Co-developed the RL-based approach for Schnapsen (initial idea proposed by Ján)
- Designed and implemented the RL architecture and buffer-memory workflow
- Built asynchronous training pipeline and worker synchronization
- Developed exploration and update strategy
- Led system integration and experimental direction

## Ján Klačan Contribution
- Co-developed the RL-based approach for Schnapsen
- Co-designed and implemented the RL architecture
- Ran prototyping and experimentation of more advanced RL algorithms, which were ultimately not covered in the paper (by decision to limit the paper's scope)
- Co-developed exploration and update strategy based on RL theory
- Led theoretical research and paper writing

## License
This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Copyright and Ownership
Copyright © 2026 Sizhong Zhang and Ján Klačan.

The framework provided in this repository is licensed under Apache 2.0. Ownership of any agent trained using this framework, including its learned parameters, weights, and derived artifacts, belongs to the user who trained it. Users retain full rights to agents they train and are free to use, distribute, or commercialize them without restriction.
