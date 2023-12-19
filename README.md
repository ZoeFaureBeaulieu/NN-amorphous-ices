# Research data supporting "High-dimensional order parameters and neural network classifiers applied to amorphous ices"

This repository supports the manuscript:

<div align="center">

> **High-dimensional order parameters and neural network classifiers applied to amorphous ices**\
> [Zoé Faure Beaulieu](https://twitter.com/ZFaureBeaulieu), [Volker Deringer](http://deringer.chem.ox.ac.uk), and [Fausto Martelli]()

</div>

---
## Repository Overview

* **[`data/`](data)** contains all extxyz files used to produce the results of the paper.

    **NOTE:** the [mda](data/mda/) and [ice_Ih](data/ice_Ih/) folders contain data taken from the work by[ Rosu-Finsen et al](https://www.science.org/doi/10.1126/science.abq2105). The original repo can be found [here](https://doi.org/10.17863/CAM.78718).
* **[`src/`](scripts)** contains the Python scripts required to run all the experiments. Notably, we provide a python implementation for Steinhard parameter calculations in [src/steinhardt.py](src/steinhardt.py).
* **[`notebooks/`](notebooks)** contains the notebooks used to generate the results in the paper:
    - [NN_classification](notebooks/NN_classification.ipynb): train a NN model with optimised hyperparamaters and apply it to LDA compression structures.
    - [reproduce_NN_results](notebooks/reproduce_NN_results.ipynb): reproduce results from [Connection between liquid and non-crystalline solid phases in water](https://doi.org/10.1063/5.0018923).
    - [sensitivity_analysis](notebooks/sensitivity_analysis.ipynb): perform sensitivity analysis using permutation feature importance.
    - [NN_optimisation](notebooks/NN_optimisation.ipynb): optimise NN parameters using Bayesian optimisation.
    - [benchmarking](notebooks/benchmarking.ipynb): train and test a number of baseline classification models
    - [steinhardt_dist_plots](notebooks/steinhardt_dist_plots.ipynb): KDE plots of Steinhardt parameters in HDA, LDA & MDA
    - [steinhardt_evolution](notebooks/steinhardt_evolution.ipynb): plot thee evolution of Steinhardt parameters as a function of pressure for LDA compression

---

## Reproducing our results

### **1. Clone the repository**
```bash
git clone git@github.com:ZoeFaureBeaulieu/NN-for-Amorphous-Ices.git
cd NN-for-Amorphous-Ices
```

### **2. Install dependencies**
All the dependencies (and their versions) used can be found in [requirements.txt](requirements.txt). To use, first create/activate your virtual environment using conda:
```bash
conda create -n ice python=3.8 -y
conda activate ice
```
Then install dependencies using:
```bash
pip install -r requirements.txt
```
---
