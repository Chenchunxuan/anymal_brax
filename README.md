# ANYmal BRAX
This repository contains code and files to train ANYmal robot in simulation using RL. 
Besides the raw training code, convenience functions are provided for logging training progress in BRAX. 
To benefit from the gradients provided by the fully differentiable simulation, applications of APG are explored.

**DISCLAIMER: This code is not official research code, but was instead developed as part of a course project at ETH (course: Digital Humans). No guarantees for its completeness or correctness can be provided.**

The given code is made publicly available for the benefit of the robotics/simulation communities, to have a trainable model of a real legged robot.

**Authors:** [Turcan Tuna](https://www.turcantuna.com/), [Julian Nubert](https://www.juliannubert.com), Jonas Frey, Sahana Betschen

**Supervisor:** Miguel Zamora

![title_img](images/anymal_running.gif)

## Getting Started
In order to get started, clone this repository together with its submodule(s).

```bash
git clone --recurse-submodules https://github.com/leggedrobotics/anymal_brax.git
```

### Dependencies
In order to run our code, the following dependencies need to be installed. These mainly contain:
* Torch
* JAX
* BRAX

We provide a conda environment file to simplify this installation. Execute:
```bash
conda env create -f conda/anymal-brax-3.11.yml
```
Next activate the given environment.
```bash
conda activate anymal-brax-3.11.yml
```

### Test the Environment
The most brittle part about the setup usually includes the successful JAX installation, suitable for GPU. In order to verify the correctness of the installation, run the following script that we provide:
```bash
python3 bin/test_jax_gpu.py
```
If this is outputting something like
```
Fast took  0.03966069221496582
Slow took 0.11339068412780762
1.0
-1.0
-1.0
```
you are good to go.

### Environment Setup
In a next setup, you can install our environment. This will also install the [BRAX library](https://github.com/google/brax/tree/v0.9.1) at release `v0.9.1` for you.
```bash
pip install -e submodules/brax
pip install -e .
```