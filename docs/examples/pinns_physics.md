# Physics-Informed Neural Networks (PINNs) Tutorial

In this tutorial, you will solve the non-linear viscous 1D Burgers Equation using Physics-Informed Neural Networks strictly in NumPy.

---

## 1. Mathematical Formulation of Burgers Equation

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0, \quad x \in [-1, 1], \quad t \in [0, 1]$$

Initial condition: $u(x, 0) = -\sin(\pi x)$  
Boundary conditions: $u(-1, t) = u(1, t) = 0$

---

## 2. Training the PINN

```python
import numpy as np
import chokkhu as ck
from chokkhu.sciml.pinns import BurgersPINN

# 1. Initialize Burgers PINN
nu = 0.01 / np.pi
pinn = BurgersPINN(hidden_dim=32, num_layers=3, nu=nu)

# 2. Sample collocation domain points
n_colloc = 300
x_colloc = np.random.uniform(-1, 1, size=(n_colloc, 1))
t_colloc = np.random.uniform(0, 1, size=(n_colloc, 1))

# 3. Sample initial condition points
n_init = 60
x_init = np.random.uniform(-1, 1, size=(n_init, 1))
t_init = np.zeros_like(x_init)
u_init = -np.sin(np.pi * x_init)

# 4. Sample boundary condition points
n_bc = 60
t_bc = np.random.uniform(0, 1, size=(n_bc, 1))
x_bc_left = -np.ones_like(t_bc)
x_bc_right = np.ones_like(t_bc)

# 5. Fit model using physics + boundary losses
pinn.fit(
    x_colloc=x_colloc,
    t_colloc=t_colloc,
    x_init=x_init,
    t_init=t_init,
    u_init=u_init,
    x_bc_left=x_bc_left,
    x_bc_right=x_bc_right,
    t_bc=t_bc,
    epochs=150,
    lr=0.01
)
print("Burgers PINN trained successfully!")
```

---

## 3. Predict & Evaluate Solution

```python
x_test = np.linspace(-1, 1, 50).reshape(-1, 1)
t_test = np.full_like(x_test, 0.25)

u_pred = pinn.predict(x_test, t_test)
print(f"Predicted velocity field at t = 0.25:\n{u_pred[:5]}")
```
