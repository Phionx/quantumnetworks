# quantumnetworks

[![License](https://img.shields.io/github/license/Phionx/quantumnetworks.svg?style=popout-square)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/github/release/Phionx/quantumnetworks.svg?style=popout-square)](https://github.com/Phionx/quantumnetworks/releases)
[![](https://img.shields.io/pypi/dm/quantumnetworks.svg?style=popout-square)](https://pypi.org/project/quantumnetworks/)
## Installation

`quantumnetworks` will soon be published on PyPI. So, to install, simply run:

```
pip install quantumnetworks
```


## Building from source

To build `quantumnetworks` from source, pip install using:
```
git clone git@github.com:Phionx/quantumnetworks.git
cd quantumnetworks
pip install --upgrade .
```

If you also want to download the dependencies needed to run optional visualization tools, please use `pip install --upgrade .[visualization]` or `pip install --upgrade '.[visualization]'` (for `zsh` users).


#### Installation for Devs

If you intend to contribute to this project, please install `quantumnetworks` in develop mode as follows:
```
git clone git@github.com:Phionx/quantumnetworks.git
cd quantumnetworks
pip install -e .[visualization]
```
Please use `pip install -e '.[visualization]'` if you are a `zsh` user.

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

