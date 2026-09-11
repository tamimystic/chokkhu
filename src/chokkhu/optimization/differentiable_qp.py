"""Differentiable Quadratic Programming (OptNet) and Implicit Layer Solvers.

Formulated from first principles using Primal-Dual Interior Point Methods (IPM)
and exact KKT implicit matrix differentials in pure NumPy and SciPy.
"""

from typing import Dict, Optional

import numpy as np


class OptNet:
    r"""Differentiable Quadratic Programming (QP) Solver & Implicit Neural Layer.

        Solves the convex quadratic optimization problem:
        \min_z
    rac{1}{2} z^T Q z + p^T z \quad 	ext{s.t.} \quad A z = b, \quad G z \le h

        and computes analytical backward parameter gradients via the KKT optimality conditions:
        egin{bmatrix} Q & A^T & G^T \ A & 0 & 0 \ 	ext{diag}(\lambda^*) G & 0 & 	ext{diag}(G z^* - h) \end{bmatrix}
        egin{bmatrix} 	ext{d}z \ 	ext{d}
    u \ 	ext{d}\lambda \end{bmatrix} = -egin{bmatrix}
    abla_z \mathcal{L} \ 0 \ 0 \end{bmatrix}

        Parameters
        ----------
        max_iter : int, default=50
            Maximum interior-point solver iterations.
        tol : float, default=1e-6
            Primal-dual duality gap convergence tolerance.
        eps_reg : float, default=1e-7
            Diagonal Tikhonov regularization for positive definiteness.
    """

    def __init__(
        self,
        max_iter: int = 50,
        tol: float = 1e-6,
        eps_reg: float = 1e-7,
    ) -> None:
        self.max_iter = max(5, int(max_iter))
        self.tol = float(tol)
        self.eps_reg = float(eps_reg)

        # Cached optimal primal-dual solutions for backward differentiation
        self.z_star: Optional[np.ndarray] = None
        self.nu_star: Optional[np.ndarray] = None
        self.lambda_star: Optional[np.ndarray] = None
        self.Q_cached: Optional[np.ndarray] = None
        self.p_cached: Optional[np.ndarray] = None
        self.A_cached: Optional[np.ndarray] = None
        self.b_cached: Optional[np.ndarray] = None
        self.G_cached: Optional[np.ndarray] = None
        self.h_cached: Optional[np.ndarray] = None

    def forward(
        self,
        Q: np.ndarray,
        p: np.ndarray,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        G: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve primal QP to optimality using Primal-Dual Interior Point Method.

        Parameters
        ----------
        Q : np.ndarray of shape (n, n)
            Symmetric positive semi-definite quadratic objective matrix.
        p : np.ndarray of shape (n,)
            Linear objective vector.
        A : Optional[np.ndarray] of shape (m_eq, n), default=None
            Equality constraint matrix.
        b : Optional[np.ndarray] of shape (m_eq,), default=None
            Equality constraint bounds.
        G : Optional[np.ndarray] of shape (m_ineq, n), default=None
            Inequality constraint matrix.
        h : Optional[np.ndarray] of shape (m_ineq,), default=None
            Inequality constraint upper bounds.

        Returns
        -------
        z_opt : np.ndarray of shape (n,)
            Optimal primal variable solution z*.
        """
        Q_mat = np.asarray(Q, dtype=float)
        p_vec = np.asarray(p, dtype=float).flatten()
        n = p_vec.shape[0]

        # Regularize Q for strict positive definiteness
        Q_reg = Q_mat + self.eps_reg * np.eye(n)

        m_eq = 0 if A is None or b is None else len(b)
        m_in = 0 if G is None or h is None else len(h)

        A_mat = np.zeros((0, n)) if A is None else np.asarray(A, dtype=float)
        b_vec = np.zeros(0) if b is None else np.asarray(b, dtype=float).flatten()

        G_mat = np.zeros((0, n)) if G is None else np.asarray(G, dtype=float)
        h_vec = np.zeros(0) if h is None else np.asarray(h, dtype=float).flatten()

        # Initialize primal-dual variables
        z: np.ndarray = np.zeros(n, dtype=float)
        nu: np.ndarray = np.zeros(m_eq, dtype=float)
        lam: np.ndarray = np.ones(m_in, dtype=float) if m_in > 0 else np.zeros(0)
        s: np.ndarray = np.ones(m_in, dtype=float) if m_in > 0 else np.zeros(0)

        # Initial feasible slack s if G is present
        if m_in > 0:
            viol = G_mat.dot(z) - h_vec
            s = np.maximum(1.0, -viol + 1.0)

        for _ in range(self.max_iter):
            # Evaluate KKT residuals
            r_dual = Q_reg.dot(z) + p_vec
            if m_eq > 0:
                r_dual += A_mat.T.dot(nu)
            if m_in > 0:
                r_dual += G_mat.T.dot(lam)

            r_eq = A_mat.dot(z) - b_vec if m_eq > 0 else np.zeros(0)
            r_in = G_mat.dot(z) + s - h_vec if m_in > 0 else np.zeros(0)

            gap = float(np.dot(s, lam)) if m_in > 0 else 0.0
            res_norm = float(
                np.linalg.norm(r_dual)
                + (np.linalg.norm(r_eq) if m_eq > 0 else 0.0)
                + (np.linalg.norm(r_in) if m_in > 0 else 0.0)
            )

            if res_norm < self.tol and (m_in == 0 or gap / m_in < self.tol):
                break

            mu = gap / m_in if m_in > 0 else 0.0
            sigma = 0.1  # Centering parameter

            # Solve Newton system
            # [ Q + G^T diag(lam/s) G     A^T ] [ dz  ] = - [ r_dual + G^T (lam - sigma*mu/s - diag(lam/s)*r_in) ]
            # [          A                 0   ] [ dnu ]   - [ r_eq                                            ]
            if m_in > 0:
                D_inv = lam / (s + 1e-12)
                H_kkt = Q_reg + (G_mat.T * D_inv).dot(G_mat)
                rhs_z = -(
                    r_dual
                    + G_mat.T.dot(lam - (sigma * mu) / (s + 1e-12) - D_inv * r_in)
                )
            else:
                H_kkt = Q_reg
                rhs_z = -r_dual

            if m_eq > 0:
                KKT_mat = np.block(
                    [
                        [H_kkt, A_mat.T],
                        [A_mat, np.zeros((m_eq, m_eq))],
                    ]
                )
                rhs_kkt = np.concatenate([rhs_z, -r_eq])
                try:
                    sol = np.linalg.solve(
                        KKT_mat + 1e-8 * np.eye(KKT_mat.shape[0]), rhs_kkt
                    )
                except np.linalg.LinAlgError:
                    sol = np.linalg.pinv(KKT_mat).dot(rhs_kkt)

                dz = sol[:n]
                dnu = sol[n:]
            else:
                try:
                    dz = np.linalg.solve(H_kkt, rhs_z)
                except np.linalg.LinAlgError:
                    dz = np.linalg.pinv(H_kkt).dot(rhs_z)
                dnu = np.zeros(0)

            if m_in > 0:
                ds = -r_in - G_mat.dot(dz)
                dlam = -(lam * ds + s * lam - sigma * mu) / (s + 1e-12)

                # Line search step size (fraction-to-boundary rule)
                alpha_s = 1.0
                alpha_lam = 1.0
                neg_ds = ds < 0
                if np.any(neg_ds):
                    alpha_s = min(1.0, float(0.99 * np.min(-s[neg_ds] / ds[neg_ds])))
                neg_dlam = dlam < 0
                if np.any(neg_dlam):
                    alpha_lam = min(
                        1.0, float(0.99 * np.min(-lam[neg_dlam] / dlam[neg_dlam]))
                    )

                alpha = min(alpha_s, alpha_lam)
            else:
                alpha = 1.0
                ds = np.zeros(0)
                dlam = np.zeros(0)

            z = z + alpha * dz
            if m_eq > 0:
                nu = nu + alpha * dnu
            if m_in > 0:
                s = s + alpha * ds
                lam = lam + alpha * dlam

        # Cache variables for backward pass
        self.z_star = z.copy()
        self.nu_star = nu.copy() if m_eq > 0 else None
        self.lambda_star = lam.copy() if m_in > 0 else None
        self.Q_cached = Q_reg
        self.p_cached = p_vec
        self.A_cached = A_mat if m_eq > 0 else None
        self.b_cached = b_vec if m_eq > 0 else None
        self.G_cached = G_mat if m_in > 0 else None
        self.h_cached = h_vec if m_in > 0 else None

        return z

    def backward(
        self,
        grad_output: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute exact analytical parameter gradients via KKT implicit differentiation.

        Parameters
        ----------
        grad_output : np.ndarray of shape (n,)
            Upstream loss gradient with respect to optimal solution z*: nabla_z L.

        Returns
        -------
        gradients : Dict[str, np.ndarray]
            Dictionary with gradients: 'dQ', 'dp', 'dA', 'db', 'dG', 'dh'.
        """
        if self.z_star is None or self.Q_cached is None:
            raise ValueError("Forward pass must be executed before backward pass.")

        z_star = self.z_star
        n = z_star.shape[0]
        gz = np.asarray(grad_output, dtype=float).flatten()

        m_eq = 0 if self.A_cached is None else self.A_cached.shape[0]
        m_in = 0 if self.G_cached is None else self.G_cached.shape[0]

        nu_star = self.nu_star if self.nu_star is not None else np.zeros(0)
        lam_star = self.lambda_star if self.lambda_star is not None else np.zeros(0)

        # Construct full linearized KKT Jacobian matrix:
        # [ Q          A^T      G^T      ] [ d_z   ]   - [ gz ]
        # [ A           0        0       ] [ d_nu  ] = - [ 0  ]
        # [ diag(lam)G  0   diag(Gz - h) ] [ d_lam ]   - [ 0  ]
        blocks_row1 = [self.Q_cached]
        if m_eq > 0:
            assert self.A_cached is not None
            blocks_row1.append(self.A_cached.T)
        if m_in > 0:
            assert self.G_cached is not None
            blocks_row1.append(self.G_cached.T)

        total_dim = n + m_eq + m_in
        KKT = np.zeros((total_dim, total_dim), dtype=float)
        KKT[:n, :n] = self.Q_cached

        col_offset = n
        if m_eq > 0:
            assert self.A_cached is not None
            KKT[:n, col_offset : col_offset + m_eq] = self.A_cached.T
            KKT[col_offset : col_offset + m_eq, :n] = self.A_cached
            col_offset += m_eq

        if m_in > 0:
            assert self.G_cached is not None and self.h_cached is not None
            KKT[:n, col_offset : col_offset + m_in] = self.G_cached.T
            # Complementary slackness row: diag(lam) G z + diag(Gz - h) lam = 0
            Gz_h = self.G_cached.dot(z_star) - self.h_cached
            KKT[col_offset : col_offset + m_in, :n] = np.diag(lam_star).dot(
                self.G_cached
            )
            KKT[col_offset : col_offset + m_in, col_offset : col_offset + m_in] = (
                np.diag(Gz_h)
            )

        rhs = np.zeros(total_dim, dtype=float)
        rhs[:n] = -gz

        # Solve for adjoint variables
        try:
            d_vars = np.linalg.solve(KKT + 1e-8 * np.eye(total_dim), rhs)
        except np.linalg.LinAlgError:
            d_vars = np.linalg.pinv(KKT).dot(rhs)

        d_z = d_vars[:n]
        d_nu = d_vars[n : n + m_eq] if m_eq > 0 else np.zeros(0)
        d_lam = d_vars[n + m_eq :] if m_in > 0 else np.zeros(0)

        # Compute parameter differentials
        dQ = 0.5 * (np.outer(d_z, z_star) + np.outer(z_star, d_z))
        dp = d_z

        grads: Dict[str, np.ndarray] = {"dQ": dQ, "dp": dp}

        if m_eq > 0:
            grads["dA"] = np.outer(d_nu, z_star) + np.outer(nu_star, d_z)
            grads["db"] = -d_nu

        if m_in > 0:
            grads["dG"] = np.outer(d_lam, z_star) + np.outer(lam_star, d_z)
            grads["dh"] = -d_lam

        return grads
