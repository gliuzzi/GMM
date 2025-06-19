# Gradient Method with Momentum (GMM)
Implementation of a globally convergent gradient method with momentum presented in 

[M. Lapucci, G. Liuzzi, S. Lucidi, D.Pucci and M. Sciandrone - A Globally Convergent Gradient Method with Momentum - arXiv:2403.17613 [math.OC]](https://arxiv.org/abs/2403.17613)

<div style="display: flex; gap: 10px;">
  <iframe src="Results/plots_cutest_gmm_lbfgs/n_it.pdf" width="45%" height="500px"></iframe>
  <iframe src="Results/plots_cutest_gmm_lbfgs/n_it.pdf" width="45%" height="500px"></iframe>
</div>

## Installation

A working [Anaconda](https://www.anaconda.com/) installation is required. We suggest the creation of a new conda environment with ```Python 3.11```. All the required packages can be installed with:
```
pip install -r requirements.txt
```
To execute ```CUTEst``` experiments, make sure that the problems are locally [installed](https://jfowkes.github.io/pycutest/_build/html/install.html).

## Usage
The experiments can be executed using:
```
python main_gmm.py [options]
```
The following arguments shall be specified:

<div align='center'>
  
| Short Option  | Long Option           | Type    | Description                                          | Default           |
|---------------|-----------------------|---------|------------------------------------------------------|-------------------|
| `-p`          | `--problem`    | `str`   | Problem Name               | None (required)   |
| `-g_tol`          | `--grad_tol`           | `float`   | Termination condition on the gradient $\ell_\infty$-norm | `1e-6`   |  
| `-m_iter`         | `--max_iter`      | `float`   | Termination condition on the number of iterations | `5000`   |
| `-l_ref`         | `--l_reg`      | `float`   | $\ell_2$-regularization parameter (for training problems only)  | `1e-3`|

</div>


## Acknowledgements
The folder `Preliminary_Experiments` contains the code used to obtain the preliminar results presented in Section 7 of the paper. In the same folder the `.txt` file contains the outcome of those experiments.

The outcome of the other experiments reported in Section 7 are avaialble in the folder `Results`. The performance profiles reported in the paper are avialable in the same folder.

In case you employed our code for research purposes, please cite:

```
@misc{lapucci2024globallyconvergentgradientmethod,
      title={A Globally Convergent Gradient Method with Momentum}, 
      author={Matteo Lapucci and Giampaolo Liuzzi and Stefano Lucidi and Davide Pucci and Marco Sciandrone},
      year={2024},
      eprint={2403.17613},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2403.17613}, 
}
```