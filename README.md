# dnlp-examples

This repository contains examples accompanying the paper *Disciplined Nonlinear Programming (DNLP)*. Each example demonstrates solving a nonlinear optimization problem using CVXPY with NLP solvers (IPOPT, Knitro).

## Examples

### Circle Packing
Pack circles of varying radii into the smallest possible square. Uses nonconvex distance constraints to prevent overlap, with multi-start optimization to escape local minima.

### Localization
Estimate an unknown source location from noisy range measurements to anchor points. Minimizes the sum of squared errors between measured and predicted distances.

### Non-negative Matrix Factorization (NMF)
Decompose a data matrix into non-negative factors for image denoising. Recovers latent basis images (circle, square, triangle) from noisy mixtures.

### Optimal Control (Car Trajectory)
Plan a trajectory for a car with nonlinear bicycle dynamics. Solves for control inputs (speed, steering) subject to acceleration and steering rate limits for parallel parking.

### Path Planning
Find the shortest path between two points while avoiding circular obstacles. Uses nonconvex collision avoidance constraints.

### Phase Retrieval
Recover a complex signal from magnitude-only measurements. A fundamental problem in imaging and signal processing where phase information is lost.

### Portfolio Construction
Construct a portfolio with target sector risk contributions. Balances expected return against risk while constraining how much each sector contributes to total portfolio variance.

### Power Flow (AC-OPF)
Solve the AC optimal power flow problem on the IEEE 9-bus test system. Minimizes generation cost subject to nonlinear power balance equations and voltage limits.

### Sparse Recovery
Compare convex (L1) vs nonconvex (L1/2) regularization for recovering sparse signals from underdetermined linear measurements. Demonstrates improved recovery with nonconvex penalties.

## Requirements

- CVXPY with [DNLP](https://github.com/cvxgrp/DNLP) support
- NLP solver: IPOPT, Knitro, or COPT
- numpy, scipy, matplotlib
