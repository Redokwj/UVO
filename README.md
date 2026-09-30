# UVO (Universal Visual Odometry)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Early%20Development%20%2F%20WIP-yellow.svg)](#)

**UVO** is an experimental modular visual odometry and SLAM framework developed in Python and PyTorch. The project focuses on exploring modular visual tracking, metric depth conditioning, and factor graph backends for mobile robotics.

> **Note**: This project is in its early stages of development (WIP). APIs, interfaces, and modules are subject to frequent changes as development progresses.

---

## Modular Architecture Overview

The system is structured into modular components:

* **Frontend (uvo.frontend)**: Visual feature detection and tracking (classical optical flow and learning-based keypoints).
* **Metric Scale Prior (uvo.metric)**: Integration interfaces for metric depth models and scale estimators.
* **Backend (uvo.backend)**: Sliding-window optimization, Lie algebra pose representations, and factor graph interfaces.
* **Loop Closure (uvo.loop_closure)**: Place recognition and relative transformation verification.
* **Navigation (uvo.navigation)**: Trajectory representation, local mapping, and path utilities.

---

## Project Structure

`	ext
UVO/
├── uvo/                    # Core Python package
│   ├── core/               # Geometry, frame structures, trajectory representations
│   ├── frontend/           # Visual feature tracking modules
│   ├── metric/             # Depth interfaces and scale estimation
│   ├── backend/            # Optimization backends and factor graph
│   ├── loop_closure/       # Place recognition and loop candidates
│   ├── navigation/         # Costmaps and path serialization
│   └── pipeline.py         # Pipeline orchestrator
├── scripts/                # Utility and evaluation scripts
├── thirdparty/             # Third-party submodules and models
├── pyproject.toml          # Packaging metadata
├── setup.py                # Package installer
└── LICENSE                 # Apache 2.0 License
`

---

## Getting Started

### Installation

1. Clone the repository:
`ash
git clone https://github.com/Redokwj/UVO.git
cd UVO
`

2. Create a virtual environment:
`ash
conda create -n uvo python=3.12 -y
conda activate uvo
`

3. Install requirements and package:
`ash
pip install -r requirements.txt
pip install -e .
`

---

## Roadmap

- [x] Initial repository structure and modular design
- [x] Core SE(3) geometry and frame handling
- [ ] Refactor tracking frontend interfaces
- [ ] Unified metric depth prior abstractions
- [ ] Backend factor graph solver optimizations
- [ ] Benchmarking utilities and test suite

---

## License

Distributed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for details.

Author: **Redokwj**
