# Research data supporting "[MDA Paper Title]"

This repository supports the manuscript:

<div align="center">

> **[MDA Paper Title]**\
> [Zoé Faure Beaulieu](https://twitter.com/ZFaureBeaulieu), [Volker Deringer](http://deringer.chem.ox.ac.uk), and [Fausto Martelli]()

</div>

---
## Repository Overview

* **[data](data)** contains all extxyz files used to produce the results of the paper
* **[src](scripts)** contains the Python scripts required to run all the experiments.
* **[notebooks](notebooks)** contains the notebooks used to generate the plots in the paper.

---

## Reproducing our results

### **1. Clone the repository**
```bash
git clone git@github.com:ZoeFaureBeaulieu/mda_paper.git
cd mda_paper
```

### **2. Install dependencies**
All the dependencies (and their versions) used can be found in [requirements.txt](requirements.txt). To use, first create/activate your virtual environment using conda:
```bash
conda create -n gpr python=3.8 -y
conda activate gpr
```
Then install dependencies using:
```bash
pip install -r requirements.txt
```

### **3. Run an experiment**
At this stage, you should be ready to run any of the scripts and/or notebooks. 
```
```
---