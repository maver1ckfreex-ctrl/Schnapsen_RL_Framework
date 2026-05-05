# Schnapsen_RL_Framework
Analyzer || Applied Project || Credential Project

A custom reinforcement-learning Schnapsen bot trained through asynchronous self-play.

## Overview
This project explores a lightweight RL training pipeline for Schnapsen using:
- asynchronous worker-based self-play
- replay-buffer memory
- epsilon-greedy exploration
- target-network synchronization
- MLP-based move evaluation

The final system is based on customized policy-learning RL.  
Earlier exploration was influenced by dynamic-dataset supervised-learning ideas, but the deployed approach is reinforcement-learning based.

## Features
- Asynchronous parallel game simulation with multiple worker processes
- Replay buffer for experience storage
- Epsilon decay for exploration control
- Lightweight MLP policy for move scoring

## Tech Stack
- Python
- PyTorch
- Multiprocessing
- Custom Schnapsen environment integration

## Sizhong Zhang Contribution
- Co-developed the RL-based approach for Schnapsen (initial idea proposed by Ján)
- Designed and implemented the RL architecture and buffer-memory workflow
- Built asynchronous training pipeline and worker synchronization
- Developed exploration and update strategy
- Led system integration and experimental direction

## Ján Klačan Contribution
-[Placeholder]

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Copyright and Ownership

Copyright © 2026 Sizhong Zhang and Ján Klačan.

The framework provided in this repository is licensed under Apache 2.0. Ownership of any agent trained using this framework — including its learned parameters, weights, and derived artifacts — belongs to the user who trained it. Users retain full rights to agents they train and are free to use, distribute, or commercialize them without restriction.
