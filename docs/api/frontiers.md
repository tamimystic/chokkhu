# Advanced Frontiers API Reference

Chokkhu implements cutting-edge research frontiers across Deep Equilibrium Models, Spiking Neural Networks, Quantum Machine Learning, Differential Privacy, Optimal Transport, and Diffusion Models in pure NumPy.

---

## 1. Deep Equilibrium Models & Spiking Neural Networks

### Deep Equilibrium Models (`DEQ`)
Finds implicit root fixed points $z^* = f_\theta(z^*, x)$ using Anderson Acceleration:

$$z^{(k+1)} = f_\theta(z^{(k)}, x)$$

### Spiking Neural Networks (`LIFSpikingLayer`)
Simulates biologically inspired Leaky Integrate-and-Fire membrane potential dynamics:

$$\tau_m \frac{d V(t)}{d t} = -(V(t) - V_{\text{rest}}) + R I(t)$$

$$\text{Spike: } S(t) = \Theta(V(t) - V_{\text{thresh}})$$

---

## 2. Privacy & Quantum Machine Learning

### Differential Privacy (`DPSGD_Optimizer`)
Clips per-sample gradients $\|g_i\|_2 \le C$ and injects calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$ to guarantee $(\epsilon, \delta)$-differential privacy.

### Variational Quantum Classifier (`VQC`)
Simulates parameterized unitary quantum gates $U(\theta)$ on state vectors $|\psi\rangle$ with Pauli-Z expectation value measurements $\langle Z \rangle$.

---

## 3. Optimal Transport & Diffusion Models

### Sinkhorn Optimal Transport (`SinkhornDistance`)
Solves entropy-regularized Kantorovich optimal transport distance via matrix scaling iterations:

$$\min_{P \in U(a, b)} \langle P, M \rangle - \epsilon H(P)$$

### Classifier-Free Guided Diffusion (`CFGDiffusion`)
Interpolates between conditional and unconditional score estimations:

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$
