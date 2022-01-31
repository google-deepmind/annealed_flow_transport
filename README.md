# Continual Repeated Annealed Flow Transport Monte Carlo (CRAFT)
# and Annealed Flow Transport Monte Carlo (AFT)

The release contains implementations of

* Continual Repeated Annealed Flow Transport Monte Carlo (CRAFT), (this paper).
* Annealed Flow Transport Monte Carlo (AFT), Arbel et al (2021).
* Stochastic Normalizing Flows (SNF), Wu et al (2020).
* Sequential Monte Carlo samplers (SMC), Del Moral et al (2006).
* Variational inference with Normalizing Flows (VI), Rezende and Mohamed (2015).
* Particle Markov Chain Monte Carlo (PIMH), Andrieu et al (2010).

The implementation of AFT is based on Algorithm 2 of that paper.
See https://arxiv.org/abs/2102.07501 for more details.

The implementation of SNFs differs from the original one in that it exploits
the connection with Annealed Importance Sampling with added normalizing flows.
The training dynamics are still the same. 

The implementation of Particle Markov Chain Monte Carlo is specialized to
the case of a final target of interest rather than a time series and 
assumes an independent proposal, which can be based on SMC, VI or CRAFT.

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
same scale and two different locations using CRAFT. The script should print a
sequence of steps and return a log normalizing constant estimate.


The config files use the `ConfigDict` from
[ml_collections](https://github.com/google/ml_collections) to specify all
details of the desired experiment. For example: the algorithm, the MCMC kernel,
and the base distribution and target distribution. More examples can be found in
the `configs` directory.

If you specify a model snapshot destination in the config file by setting 
`config.save_checkpoint = True` and specifying `config.params_filename` then you can
store a file to evaluate using PIMH. This is then called using:

```
python evaluate.py --config=configs/single_normal.py
```

We have not released code for writing results to disk. The function
`train.run_experiments` called from `main.py` returns a `NamedDict` containing a
summary of results that could be caught and recorded if required.

## License information

The code is licensed under the Apache 2.0 license, which can be found in full in
the `LICENSE` file.

We have released a pickle model parameters file for the VAE example which is
licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0),
Full text is found at https://creativecommons.org/licenses/by/4.0/legalcode.

## Giving Credit

If you use this code in your work, please cite the corresponding paper. If you use our baselines such as SMC, SNF, PIMH, please cite the paper of the two below where we first used the method.

```
@article{CRAFT2022,
  title={Continual Repeated Annealed Flow Transport Monte Carlo},
  author={Alexander G. D. G. Matthews and Michael Arbel and Danilo J. Rezende and Arnaud Doucet},
  Journal = {arXiv},
  year={2022},
  month = {Jan}
}
```

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
