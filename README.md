# Plankton Image Classification (SYKE)

![IFCB images](collage.png)

## Introduction

> Finnish Environment Institute (abbr. SYKE in Finnish)

> Marine Research Centre (abbr. MRC, sub-organisation of SYKE)

This repository contains code for training and using convolutional neural networks in the automatic classification of plankton images. It is a part of MRC's goal to introduce new automatic monitoring and research methods for the Baltic Sea.

`syke-pic` is tailor-made for MRC's needs. It currently only supports plankton images collected from the Imaging FlowCytobot (IFCB) and relies on very specific cloud services. This means, it's not really meant for outside use.

This project is still a work in progress. For example, it currently has poor test coverage and has been built and used primarily on a Linux machine. Windows compatibility is not guaranteed for everything.

## Setup

First step is to download the code to your machine:

```sh
git clone https://github.com/veot/syke-pic
```

### Requirements

The Conda package and environment management system is recommended. See https://docs.conda.io/en/latest/ for instructions.

`syke-pic/envs/` contains two `yaml`-files that can be used to configure a Conda environment with the required dependencies. `cpu.yaml` and `gpu.yaml` will install PyTorch for CPU or GPU systems, respectively. When installing for GPU support, first change the version of `cudatoolkit` in `gpu.yaml` to match your system.

Install and configure the environment with:

```sh
conda env create -f syke-pic/envs/<yaml>
```

This will create a Conda environment called `sykepic` (name can be changed inside the `yaml`-file).

### Installation

Make sure that the required environment is activated. If you configured Conda from the above instructions, this command looks like:

```sh
conda activate sykepic
```

Then `syke-pic` can be installed with:

```sh
pip install syke-pic/
```

This command installs a Python package and an executable program, both named `sykepic`.

### Development configuration (optional)

Install in editable mode instead:

```sh
pip install -e syke-pic/
```

Additional development requirements can be installed with

```sh
pip install -r sykepic/envs/dev-requirements.txt
```

## Usage

The main interface is from the command line. Invoke the help menu with:

```sh
sykepic --help
```
