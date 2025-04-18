# Vectorial Martingale Optimal Transport

This is the numerical implementation of Hiew, Lim, Pass and Souza (2023) "Geometry of vectorial martingale optimal transport and robust option pricing". We compute the optimal value of VMOT problems on the dual side using the framework of Eckstein and Kupper (2021) with accuracy improvements due to dimensional reductions allowed by our main result. Our main computational tools are neural networks developed upon Python/PyTorch, and its implementation of the Adam gradient descent optimizer.

References
- Hiew, Lim, Pass and Souza (2023) "Geometry of vectorial martingale optimal transport and robust option pricing" https://arxiv.org/abs/2309.04947.
- Eckstein, Guo, Lim, Obloj (2021) "Robust Pricing and Hedging of Options on Multiple Assets and Its Numerics".
- Eckstein, Kupper (2021) "Computation of Optimal Transport and Related Hedging Problems via Penalization and Neural Networks".
