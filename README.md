# Vectorial Martingale Optimal Transport

This is the numerical implementation of Hiew, Lim, Pass and Souza (2023) "Geometry of vectorial martingale optimal transport and robust option pricing". We compute the dual optimization of VMOT problems using the framework developed by Eckstein and Kupper (2021) with dimensionality improvements allowed by our main result. We employ the Torch project over Python (Pytorch) and its Adam Gradient Descent optimizer implementation.

References
- Hiew, Lim, Pass and Souza (2023) "Geometry of vectorial martingale optimal transport and robust option pricing" https://arxiv.org/abs/2309.04947.
- Eckstein, Guo, Lim, Obloj (2021) "Robust Pricing and Hedging of Options on Multiple Assets and Its Numerics".
- Eckstein, Kupper (2021) "Computation of Optimal Transport and Related Hedging Problems via Penalization and Neural Networks".

Reference GitHub Projects (using Python/TensorFlow):
- https://github.com/stephaneckstein/transport-and-related/tree/master/MartingaleOT
- https://github.com/stephaneckstein/OT_Comparison