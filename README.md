# quantumnetworks

[![License](https://img.shields.io/github/license/Phionx/quantumnetworks.svg?style=popout-square)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/github/release/Phionx/quantumnetworks.svg?style=popout-square)](https://github.com/Phionx/quantumnetworks/releases)
[![](https://img.shields.io/pypi/dm/quantumnetworks.svg?style=popout-square)](https://pypi.org/project/quantumnetworks/)

---

*Please checkout this [presentation](https://docs.google.com/presentation/d/1-2YzJWmRjBr4XqfV-qTjjIAvsZ6RQoBXQy5Cf2bfN64/edit?usp=sharing) and [paper](https://drive.google.com/file/d/1Dz0B8llnb94KIqrA2uE9AOYonnJFUFdU/view?usp=sharing) for a quick overview of `quantumnetworks`!*
## Installation

*Conda users, please make sure to `conda install pip` before running any pip installation if you want to install `quantumnetworks` into your conda environment.*

`quantumnetworks` will soon be published on PyPI. So, to install, simply run:

```sh
pip install quantumnetworks
```
If you also want to download the dependencies needed to run optional tutorials, please use `pip install quantumnetworks[dev]` or `pip install 'quantumnetworks[dev]'` (for `zsh` users).


To check if the installation was successful, run:
```python
python3
>>> import quantumnetworks as qn
```

## Building from source

To build `quantumnetworks` from source, pip install using:
```
git clone git@github.com:Phionx/quantumnetworks.git
cd quantumnetworks
pip install --upgrade .
```

If you also want to download the dependencies needed to run optional tutorials, please use `pip install --upgrade .[dev]` or `pip install --upgrade '.[dev]'` (for `zsh` users).


#### Installation for Devs

If you intend to contribute to this project, please install `quantumnetworks` in develop mode as follows:
```sh
git clone git@github.com:Phionx/quantumnetworks.git
cd quantumnetworks
pip install -e .[dev]
```
Please use `pip install -e '.[dev]'` if you are a `zsh` user.

## Motivation

We present `quantumnetworks` as a numerical simulation tool with which to explore the time-dynamics of a driven, lossness, and nonlinear multi-mode quantum network using the Heisenberg-Langevin Equations. The applications of this tooling span quantum transduction, bosonic quantum error correction systems, quantum communication, and more. 

## Codebase

The codebase is split across `quantumnetworks/systems` and `quantumnetworks/analysis`, which respectively provide solvers and analysis tools for several quantum network systems of interest. 

## Future Directions

Checkout [issues](https://github.com/Phionx/quantumnetworks/issues) to see what we are working on these days!

## Acknowledgements 

Core Devs: [Shantanu Jha](https://github.com/Phionx), [Shoumik Chowdhury](https://github.com/shoumikdc), [Lamia Ateshian ](https://github.com/ateshian)

Thanks to [Professor Luca Daniel](https://www.mit.edu/~dluca/) and our TA, Taqiyyah Safi, for invaluable feedback during the development of this package in the Fall 2021 iteration of *Introduction to Numerical Simulation* (6.336) at MIT.

<!-- ## Reference -->

