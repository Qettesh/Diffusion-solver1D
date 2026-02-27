# Diffusion-solver1D
This is a concise Java implementation that numerically solves the 1D diffusion (heat) equation using an explicit finite-difference method and incorporates a simple evolutionary algorithm (EA) framework to evolve parameters of the numerical model to match a target solution. The code is self-contained, modular, and focused on clarity.

Key choices/assumptions:

1D diffusion equation: ∂u/∂t = D ∂²u/∂x² on x ∈ [0, L] with Dirichlet boundary conditions.
Explicit forward-time centered-space (FTCS) scheme (stable when D*dt/dx² ≤ 0.5).
EA evolves real-valued genomes [D, dt] to minimize mean-squared error (MSE) between simulated u(x, T) and a given target profile.
Uses simple GA operators: selection by tournament, Gaussian mutation, uniform crossover, elitism.
Reasonable defaults and constraints enforce numerical stability.
Compile and run on Java 11+
