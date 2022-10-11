# Plankton Image Classification (SYKE)

![IFCB images](collage.png)

## Introduction

> Finnish Environment Institute (abbr. SYKE in Finnish)

> Marine Research Centre (abbr. MRC, sub-organisation of SYKE)

This repository contains code for training and using convolutional neural networks in the automatic classification of plankton images. It is a part of MRC's goal to introduce new automatic monitoring and research methods for the Baltic Sea.

`syke-pic` is built specifically for MRC's needs and supports only plankton images collected from the Imaging FlowCytobot (IFCB). This means, it's not really meant for outside use.

The code is still a work in progress. For example, it currently has poor test coverage and has been built and used primarily on a Linux machine. Windows compatibility is not guaranteed for everything.

Contact person: kaisa.kraft at syke.fi

## Setup

First step is to download the code to your machine:

```sh
git clone https://github.com/veot/syke-pic
cd syke-pic
```

### Requirements

Python requirements are listed in the files found in the [requirements/](requirements/) directory. There are two versions of the requirements (gpu and cpu), depending on the availability of a PyTorch compatible GPU. If in doubt, use the cpu one.

Files ending in `.txt` contain pinned versions of the dependencies, but installing from the `.in` files should work as well, although this might change in the future as the dependencies evolve.

It is advisable to create a new Python virtual environment to install everything into, e.g., with the built-in `venv` tool:

```sh
python -m venv .venv
source .venv/bin/activate
```

Then install the (cpu) requirements with:

```sh
pip install -r requirements/cpu.txt
# or if that doesn't work:
pip install -r requirements/cpu.in
```

#### Additional requirements for feature extraction

In order to perform feature extraction (calculating biovolumes etc.) for the IFCB images, you will need either the Matlab or Python version of the software. The Matlab version can be installed from its original repository: https://github.com/hsosik/ifcb-analysis, but is harder to setup.

The Python version currently configured to work with `syke-pic` is our fork: https://github.com/veot/ifcb-features. The installation is a simple `git clone` and `pip install`.

### Installation

`syke-pic` can be installed with:

```sh
pip install .
```

This command installs a Python package and an executable program to your virtual environment, both named `sykepic`.

### Development configuration (optional)

Install the package in editable mode instead:

```sh
pip install -e .
```

Additional development requirements can be installed with

```sh
pip install -r requirements/dev.in
```

### Post installation (optional)

To make sure the code is working as intended, you can run the available tests. The only additional requirement is `pytest`, which you can just install separately: `pip install pytest`. It is also included in the development requirements.

Run the tests with this one command:

```sh
pytest
```

## Usage

The main interface is from the command line. Invoke the help menu with:

```sh
sykepic --help
```
