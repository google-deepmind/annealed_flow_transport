# Annealed Flow Transport Monte Carlo

Open source implementation accompanying ICML 2021 paper

by Michael Arbel*, Alexander G. D. G. Matthews* and Arnaud Doucet.

The release contains implementations of
* Annealed Flow Transport Monte Carlo (AFT), this paper.
* Sequential Monte Carlo samplers (SMC), Del Moral et al (2006).
* Variational inference with Normalizing Flows (VI), Rezende and Mohamed (2015).

This implementation of AFT is based on Algorithm 2 in the paper.
See https://arxiv.org/abs/2102.07501 for more details.

## Installation

The code uses Python 3. We recommend using `pip install -e .` which makes an
editable install. A reliable way to do this is within a
[virtual environment](https://docs.python-guide.org/dev/virtualenvs/).

```
virtualenv -p python3.9 ~/venv/annealed_flow_transport
source ~/venv/annealed_flow_transport/bin/activate
pip install -e .
```

A GPU is highly recommended. To use one you will need to install JAX with CUDA
support. For example:

```
pip install --upgrade jax jaxlib==0.1.68+cuda111 -f
https://storage.googleapis.com/jax-releases/jax_releases.html
```

The CUDA version will need to match your GPU drivers.
See the [JAX documentation](https://github.com/google/jax#installation) for more
discussion.

To run the unit tests use the following command:

```
python -m pytest
```

## Usage

The entry point to the code is `main.py` taking a config file as an argument.
As an example from the base directory the following command runs a simple
one dimensional toy example:

```
python main.py --config=configs/single_normal.py
```

This example anneals between two one dimensional normal distributions with the
same scale and two different locations using AFT. The script should print a
sequence of steps and return a log normalizing constant estimate.

The config files use the `ConfigDict` from
[ml_collections](https://github.com/google/ml_collections) to specify all
details of the desired experiment. For example: the algorithm, the MCMC kernel,
and the base distribution and target distribution. More examples can be found in
the `configs` directory.

We have not open sourced code for writing results to disk. The function
`train.run_experiments` called from `main.py` returns a `NamedDict` containing a
summary of results that could be caught and recorded if required.

## Giving Credit

If you use this code in your work, please cite the following paper.

```
@InProceedings{AnnealedFlowTransport2021,
  title={Annealed Flow Transport Monte Carlo},
  author={Michael Arbel and Alexander G. D. G. Matthews and Arnaud Doucet},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  year={2021},
  month = {18--24 Jul}
}
```

## Disclaimer

This is not an official Google product.
