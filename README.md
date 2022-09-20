# fenkar_fittings
Finding local minima for model parameters in the Fenton-Karma model.

This code uses randomly sampled (or previously saved) parameters for the model, stimulus, and initial conditions to optimize an L2 loss function for the model solutions over time compared to experimental data (if you are here, you should have this data already).

The code uses [SLSQP](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#slsqp) for the optimization and [Tsitouras fifth order](https://diffeq.sciml.ai/stable/solvers/ode_solve/) method for the ODE solves.

In principle, because this uses a deflation operator `M(P)` to augment the problem and penalize the loss against known model parameter sets `P`, simply re-running the code with
  ```julia --project=. --threads=4 fenkar_fittings.jl``` 
should just generate new parameter sets which are relatively well-suited for the dynamics in the data. There are a number of practical issues involved which may lead to spurious or unhelpful sets being discovered.
