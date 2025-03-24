# Harmonize Project: Reproducible Scripts for "Impact of Leakage on Data Harmonization in Machine Learning Pipelines in Class Imbalance Across Sites"

## About

The Forschungszentrum Jülich Machine Learning Library

It is currently being developed and maintained at the [Applied Machine Learning](https://www.fz-juelich.de/en/inm/inm-7/research-groups/applied-machine-learning-aml) group at [Forschungszentrum Juelich](https://www.fz-juelich.de/en), Germany.

## Overview

This repository contains all scripts and resources needed to reproduce the experiments presented in the paper "Impact of Leakage on Data Harmonization in Machine Learning Pipelines in Class Imbalance Across Sites." The paper explores the effectiveness of data harmonization methods, particularly in scenarios where class balance differs across data collection sites, and proposes the **PrettYharmonize** approach to address data leakage issues. Using this repository, researchers can replicate the study results, perform experiments on synthetic and real-world datasets, and validate the PrettYharmonize pipeline.

**Paper Link**: [https://arxiv.org/abs/2410.19643](https://arxiv.org/abs/2410.19643)

## Repository Structure

- `data/`: Users must download the used public dataset and stored inside this folder.
- `data_preprocessing/`: Contains Python scripts to preprocess the data.
- `scripts/`: Contains Python scripts to run harmonization methods and apply PrettYharmonize.
- `outcome/`: Directory where results from experiments are saved.
- `plots/`: Jupyter notebooks with step-by-step examples for reproducing figures.
- `pyproject.toml`: File to set up the environment with dependencies.

## Requirements

The environment can be installed using the `pyproject.toml` file in this repository.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/juaml/harmonize_project.git
   cd harmonize_project

2. **Install PrettyHarmonize:**

   ```bash
    pip install git+https://github.com/juaml/PrettYharmonize

3. **Download data**

  Data must be downloaded by the user and stored in the respectively folders

4. **Pre-processing data**

  The pre processing of the data has to be made using the scripts conteined in `data_preprocessing/`

5. **Run scripts**

  The code for classification or regression in (in)dependence scenarios are stored in `scripts/`

6. **Plot**

For view results

## Citation
```bibtex
If you use PrettYharmonize in your work, please cite the following:
@article{nieto2024impact,
  title={Impact of Leakage on Data Harmonization in Machine Learning Pipelines in Class Imbalance Across Sites},
  author={Nieto, Nicol{\'a}s and Eickhoff, Simon B and Jung, Christian and Reuter, Martin and Diers, Kersten and Kelm, Malte and Lichtenberg, Artur and Raimondo, Federico and Patil, Kaustubh R},
  journal={arXiv preprint arXiv:2410.19643},
  year={2024}
}
```

## Licensing

preattyharmonize is released under the AGPL v3 license:

preattyharmonize, FZJuelich AML machine learning library.
Copyright (C) 2020, authors of preattyharmonize.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.