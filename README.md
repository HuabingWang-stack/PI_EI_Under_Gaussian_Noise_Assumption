# PI and EI under gaussian noise assumption

This repository contains Python code for Bayesian optimization PI, EI and a modification of PI and EI under gaussian noise assumption.
 It contains three files:

* `bo_acquis.py`: Contains the Bayesian Optimisation, code for PI and EI modified from [bayesian-optimization](https://github.com/thuijskens/bayesian-optimization), and new code for MPI and MEI.
* `plotters.py` : Contains plotter functions for plotting surface for estimated loss and acquisition value in each iteration adapted from [bayesian-optimization](https://github.com/thuijskens/bayesian-optimization).
that contains the optimization code, and utility functions to plot iterations of the algorithm, respectively.
* `PI_EI_MPI_MEI_Benchmark.ipynb`: A tutorial that uses the Bayesian algorithm with the 4 acquisitions to find the global optima on noise corrupted [benchmark functions](http://www.resibots.eu/limbo/bo_benchmarks.html).

The signature of the optimization function is still:

```python
bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
                      gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7)
```

### Background

PI and EI are two acquisition functions that return Probability of Improvement and Expected Improvement with respect to current optima $\tilde{y}$.
In some cases, the evaluations on loss function has a noise $y_i \sim \mathcal{N} (f(\mathbf{x})_i,\sigma^2_y)$. 
PI and EI are modified under the assumption that the current optima has a noise. They calcualtes Probability of Improvement and Expected Improvement with respect to 
posterior variance of loss optimum $\kappa(\tilde{\mathbf{x}},\tilde{\mathbf{x}})$ instead.  (where $\tilde{\mathbf{x}}$ is parameter setting at current optima.)

 Let $\rho$ denotes $ (\kappa (\mathbf{x}, \mathbf{x})+ \kappa (\tilde{\mathbf{x}}, \tilde{\mathbf{x}})-2 \kappa (\mathbf{x}, \tilde{\mathbf{x}}))^{\frac{1}{2}}$. Mathematical expression of Modified PI and EI under gaussian noise assumption:

$$
\text{Modified PI: }  a_{MPI}(x) = \Phi \left(\frac{\mu(\tilde{\mathbf{x}}) - \mu ( \mathbf{x} ) }{\rho})\right)
$$

$$
\text{Modified EI: } a_{MEI} = \Phi(\frac{\mu(\tilde{\mathbf{x}}) - \mu(\mathbf{x})}{\rho})(\mu(\tilde{\mathbf{x}}) - \mu(\mathbf{x}))+
        \phi(\frac{\mu(\tilde{\mathbf{x}}) - \mu(\mathbf{x})}{\rho})\rho
$$

### Current Experiment Result

MPI shows a slightly better result than PI on both original and noised corrupted rastrigin function.
Where rastrigin has a complex surface itself.
The reason is that MPI leverages more importance on exploration than PI.
MEI does not show better performance than EI on benchmark functions without noise.

Perform Bayesian Optimisation on Rastrigin function with PI and MPI; probability of improvement and loss surface in each iteration is plotted.

|                Rastrigin Surface                 |                       PI Searching Trajectory                       |                       MPI Searching Trajectory                       |
|:------------------------------------------------:|:-------------------------------------------------------------------:|:--------------------------------------------------------------------:|
| ![Alt Text](./rastrigin/real_loss_rastrigin.png) | ![Alt Text](./rastrigin/noise_less/PI_rastrigin/bo_2d_new_data.gif) | ![Alt Text](./rastrigin/noise_less/MPI_rastrigin/bo_2d_new_data.gif) |


