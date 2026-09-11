"""Machine Unlearning and Concept Scrubbing Suite for Chokkhu.

References:
    - Bourtoule et al., "Machine Learning that Forgets", IEEE S&P 2021 (SISA).
    - Golatkar et al., "Eternal Sunshine of the Spotless Net: Pruning & Anonymization", CVPR 2020 (Fisher Scrubbing).
    - Kurmanji et al., "SCRUB: Sub-network Unlearning with Bounded Degradation", NeurIPS 2023.
    - Ravfogel et al., "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection", ACL 2020.
"""

from typing import Dict, List, Union, Any, Optional, Callable, Tuple
import numpy as np


class SISARetraining:
    """Sharded, Isolated, Sliced, Aggregated (SISA) Exact Machine Unlearning.

    Partitions dataset into S shards and R slices per shard.
    Retrains only affected slices within the designated shard upon receiving forget requests.
    """

    def __init__(
        self,
        model_builder: Optional[Callable[[], Any]] = None,
        num_shards: int = 3,
        num_slices: int = 2,
        task: str = "classification",
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize SISA unlearning pipeline.

        Args:
            model_builder: Callable returning a new, uninitialized model instance.
                           Model must have .fit(X, y) and .predict(X) methods.
            num_shards: Number of disjoint data shards S.
            num_slices: Number of incremental data slices R per shard.
            task: Task type ('classification' or 'regression').
            seed: Random seed for deterministic data partitioning.
        """
        self.model_builder = model_builder
        self.num_shards = int(num_shards)
        self.num_slices = int(num_slices)
        self.task = task
        self.seed = seed

        # Internal state: shards[s]['slices'][r] = (X_slice, y_slice, indices_slice)
        # shards[s]['models'][r] = model checkpoint trained up to slice r
        self.shards: List[Dict[str, Any]] = []
        self.is_fitted = False

    def _default_model_builder(self) -> Any:
        """Default ridge/logistic regression model if none provided."""

        class LinearModel:
            def __init__(self, task: str = "classification"):
                self.task = task
                self.w = None
                self.b = 0.0

            def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearModel":
                X = np.asarray(X, dtype=np.float64)
                y = np.asarray(y, dtype=np.float64)
                N, D = X.shape
                # Ridge regression closed form: (X^T X + lambda I)^(-1) X^T y
                X_bias = np.hstack([X, np.ones((N, 1))])
                lam = 1e-3
                w_full = np.linalg.pinv(X_bias.T @ X_bias + lam * np.eye(D + 1)) @ (
                    X_bias.T @ y
                )
                self.w = w_full[:D]
                self.b = w_full[D]
                return self

            def predict(self, X: np.ndarray) -> np.ndarray:
                X = np.asarray(X, dtype=np.float64)
                scores = X @ self.w + self.b
                if self.task == "classification":
                    return np.where(scores >= 0.5, 1.0, 0.0)
                return scores

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                X = np.asarray(X, dtype=np.float64)
                scores = X @ self.w + self.b
                # Sigmoid
                p1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
                return np.stack([1.0 - p1, p1], axis=-1)

        return LinearModel(task=self.task)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SISARetraining":
        """Partition data and train SISA shard-slice model hierarchy."""
        X = np.asarray(X)
        y = np.asarray(y)
        N = len(X)

        builder = (
            self.model_builder
            if self.model_builder is not None
            else self._default_model_builder
        )
        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(N)

        # Split into shards
        shard_indices_list = np.array_split(shuffled_indices, self.num_shards)
        self.shards = []

        for s_idx, shard_indices in enumerate(shard_indices_list):
            shard_dict: Dict[str, Any] = {"slices": [], "models": []}
            # Split shard into slices
            slice_indices_list = np.array_split(shard_indices, self.num_slices)

            accumulated_X: Optional[np.ndarray] = None
            accumulated_y: Optional[np.ndarray] = None
            accumulated_idx: Optional[np.ndarray] = None

            for r_idx, slice_idx in enumerate(slice_indices_list):
                X_s = X[slice_idx]
                y_s = y[slice_idx]
                shard_dict["slices"].append(
                    {
                        "X": X_s,
                        "y": y_s,
                        "indices": slice_idx,
                    }
                )

                if accumulated_X is None:
                    accumulated_X = X_s
                    accumulated_y = y_s
                    accumulated_idx = slice_idx
                else:
                    accumulated_X = np.concatenate([accumulated_X, X_s], axis=0)
                    accumulated_y = np.concatenate([accumulated_y, y_s], axis=0)
                    accumulated_idx = np.concatenate(
                        [accumulated_idx, slice_idx], axis=0
                    )

                # Train model up to current slice
                model = builder()
                model.fit(accumulated_X, accumulated_y)
                shard_dict["models"].append(model)

            self.shards.append(shard_dict)

        self.is_fitted = True
        return self

    def forget(self, forget_indices: Union[List[int], np.ndarray]) -> "SISARetraining":
        """Execute unlearning request by retraining only the affected slices in the affected shards."""
        if not self.is_fitted:
            raise ValueError("SISARetraining must be fitted before forget()")

        forget_set = set(np.asarray(forget_indices).tolist())
        builder = (
            self.model_builder
            if self.model_builder is not None
            else self._default_model_builder
        )

        for shard in self.shards:
            # Find earliest slice in this shard that contains a forget index
            earliest_slice_to_retrain = None

            for r_idx, slice_data in enumerate(shard["slices"]):
                curr_indices = slice_data["indices"]
                overlap = [idx for idx in curr_indices if idx in forget_set]

                if overlap:
                    # Remove forget indices from slice
                    keep_mask = np.array(
                        [idx not in forget_set for idx in curr_indices], dtype=bool
                    )
                    slice_data["X"] = slice_data["X"][keep_mask]
                    slice_data["y"] = slice_data["y"][keep_mask]
                    slice_data["indices"] = curr_indices[keep_mask]

                    if earliest_slice_to_retrain is None:
                        earliest_slice_to_retrain = r_idx

            # If this shard was affected, retrain from earliest slice onward
            if earliest_slice_to_retrain is not None:
                # Accumulate data up to earliest slice
                accumulated_X: Optional[np.ndarray] = None
                accumulated_y: Optional[np.ndarray] = None
                for r in range(earliest_slice_to_retrain):
                    s_data = shard["slices"][r]
                    if accumulated_X is None:
                        accumulated_X = s_data["X"]
                        accumulated_y = s_data["y"]
                    else:
                        accumulated_X = np.concatenate(
                            [accumulated_X, s_data["X"]], axis=0
                        )
                        accumulated_y = np.concatenate(
                            [accumulated_y, s_data["y"]], axis=0
                        )

                # Retrain from earliest slice onward
                for r in range(earliest_slice_to_retrain, len(shard["slices"])):
                    s_data = shard["slices"][r]
                    if accumulated_X is None:
                        accumulated_X = s_data["X"]
                        accumulated_y = s_data["y"]
                    else:
                        accumulated_X = np.concatenate(
                            [accumulated_X, s_data["X"]], axis=0
                        )
                        accumulated_y = np.concatenate(
                            [accumulated_y, s_data["y"]], axis=0
                        )

                    if accumulated_X is not None and len(accumulated_X) > 0:
                        model = builder()
                        model.fit(accumulated_X, accumulated_y)
                        shard["models"][r] = model
                    else:
                        shard["models"][r] = None

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Aggregate predictions from the final models of all shards."""
        if not self.is_fitted:
            raise ValueError("SISARetraining must be fitted before predict()")

        X = np.asarray(X)
        shard_preds = []

        for shard in self.shards:
            final_model = shard["models"][-1]
            if final_model is not None:
                shard_preds.append(final_model.predict(X))

        if not shard_preds:
            raise ValueError("No active models available for prediction")

        stacked = np.stack(shard_preds, axis=0)  # [S, N]
        if self.task == "classification":
            # Majority voting
            if stacked.dtype == np.float64 or stacked.dtype == np.float32:
                rounded = np.round(stacked).astype(int)
            else:
                rounded = stacked.astype(int)
            # Mode along axis 0
            mean_votes = np.mean(rounded, axis=0)
            return np.where(mean_votes >= 0.5, 1, 0)
        else:
            # Mean for regression
            return np.mean(stacked, axis=0)


class FisherScrubbing:
    """Fisher Information Parameter Shift for Fast Approximate Machine Unlearning.

    Formula:
        delta_theta = - (F_retain + lambda * I)^(-1) * grad_forget
    where F_retain is the empirical Fisher Information Matrix over the retain dataset.
    """

    def __init__(
        self,
        damping: float = 1e-4,
        noise_sigma: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize FisherScrubbing.

        Args:
            damping: Regularization parameter lambda added to Fisher matrix diagonal.
            noise_sigma: Standard deviation of differential privacy perturbation noise.
            seed: Random seed for noise reproducibility.
        """
        self.damping = float(damping)
        self.noise_sigma = float(noise_sigma)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def compute_empirical_fisher(
        w: np.ndarray,
        b: float,
        X_retain: np.ndarray,
        y_retain: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute empirical Fisher matrix F and parameter gradient for a linear model.

        Model: p = sigmoid(X w + b), Loss: Binary Cross-Entropy
        """
        X = np.asarray(X_retain, dtype=np.float64)
        y = np.asarray(y_retain, dtype=np.float64)
        N, D = X.shape

        X_ext = np.hstack([X, np.ones((N, 1))])  # include bias
        theta: np.ndarray = np.append(w, b)

        scores = X_ext @ theta
        p = 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))

        # Per-sample gradients: g_i = (p_i - y_i) * x_i
        per_sample_grads = (p - y)[:, np.newaxis] * X_ext  # [N, D+1]

        # Empirical Fisher = (1/N) * sum_i g_i g_i^T
        Fisher = (per_sample_grads.T @ per_sample_grads) / N
        return Fisher, theta

    def scrub(
        self,
        w: np.ndarray,
        b: float,
        X_forget: np.ndarray,
        y_forget: np.ndarray,
        X_retain: np.ndarray,
        y_retain: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Scrub sample influence from linear model parameters.

        Returns:
            (w_scrubbed, b_scrubbed)
        """
        w = np.asarray(w, dtype=np.float64)
        X_f = np.asarray(X_forget, dtype=np.float64)
        y_f = np.asarray(y_forget, dtype=np.float64)
        X_r = np.asarray(X_retain, dtype=np.float64)
        y_r = np.asarray(y_retain, dtype=np.float64)

        N_f, D = X_f.shape
        X_f_ext = np.hstack([X_f, np.ones((N_f, 1))])

        # 1. Compute Fisher on retain set
        Fisher, theta = self.compute_empirical_fisher(w, b, X_r, y_r)

        # 2. Compute gradient on forget set
        scores_f = X_f_ext @ theta
        p_f = 1.0 / (1.0 + np.exp(-np.clip(scores_f, -500, 500)))
        grad_forget = np.mean((p_f - y_f)[:, np.newaxis] * X_f_ext, axis=0)

        # 3. Invert regularized Fisher matrix
        D_total = D + 1
        inv_Fisher = np.linalg.pinv(Fisher + self.damping * np.eye(D_total))

        # 4. Parameter shift
        delta_theta = -inv_Fisher @ grad_forget

        # 5. Optional DP noise
        if self.noise_sigma > 0.0:
            noise = self.rng.normal(0.0, self.noise_sigma, size=D_total)
            delta_theta += inv_Fisher @ noise

        theta_scrubbed = theta + delta_theta
        w_scrubbed = theta_scrubbed[:D]
        b_scrubbed = float(theta_scrubbed[D])

        return w_scrubbed, b_scrubbed


class SCRUB:
    """Selective Continual Residual Unlearning with Bounded Degradation (SCRUB).

    Alternates between:
        1. Maximizing loss on forget set (forgetting step).
        2. Minimizing distillation KL divergence on retain set (retention step).
    """

    def __init__(
        self,
        lr: float = 0.05,
        epochs: int = 10,
        forget_steps: int = 1,
        retain_steps: int = 2,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        """Initialize SCRUB unlearning.

        Args:
            lr: Learning rate for gradient updates.
            epochs: Total unlearning optimization epochs.
            forget_steps: Inner steps on forget set per epoch.
            retain_steps: Inner distillation steps on retain set per epoch.
            alpha: Weight for retain set distillation loss.
            beta: Weight for forget set ascent loss.
        """
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.forget_steps = int(forget_steps)
        self.retain_steps = int(retain_steps)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def unlearn(
        self,
        w_teacher: np.ndarray,
        b_teacher: float,
        X_forget: np.ndarray,
        y_forget: np.ndarray,
        X_retain: np.ndarray,
        y_retain: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Perform SCRUB mini-max distillation on model parameters."""
        w_student = np.copy(w_teacher).astype(np.float64)
        b_student = float(b_teacher)

        X_f = np.asarray(X_forget, dtype=np.float64)
        y_f = np.asarray(y_forget, dtype=np.float64)
        X_r = np.asarray(X_retain, dtype=np.float64)
        y_r = np.asarray(y_retain, dtype=np.float64)

        # Precompute teacher outputs on retain set
        teacher_scores_r = X_r @ w_teacher + b_teacher
        p_teacher_r = 1.0 / (1.0 + np.exp(-np.clip(teacher_scores_r, -500, 500)))

        for epoch in range(self.epochs):
            # 1. Forget Step: Gradient Ascent on Forget Set
            for _ in range(self.forget_steps):
                scores_f = X_f @ w_student + b_student
                p_student_f = 1.0 / (1.0 + np.exp(-np.clip(scores_f, -500, 500)))
                # Gradient of BCE loss
                grad_w_f = np.mean((p_student_f - y_f)[:, np.newaxis] * X_f, axis=0)
                grad_b_f = np.mean(p_student_f - y_f)

                # Ascent (maximize error / entropy on forget set)
                w_student += self.lr * self.beta * grad_w_f
                b_student = float(b_student + self.lr * self.beta * grad_b_f)

            # 2. Retain Step: Distillation Gradient Descent on Retain Set
            for _ in range(self.retain_steps):
                scores_r = X_r @ w_student + b_student
                p_student_r = 1.0 / (1.0 + np.exp(-np.clip(scores_r, -500, 500)))

                # Distillation gradient: match teacher probabilities
                grad_w_distill = np.mean(
                    (p_student_r - p_teacher_r)[:, np.newaxis] * X_r, axis=0
                )
                grad_b_distill = np.mean(p_student_r - p_teacher_r)

                # Task gradient: match ground truth labels
                grad_w_task = np.mean((p_student_r - y_r)[:, np.newaxis] * X_r, axis=0)
                grad_b_task = np.mean(p_student_r - y_r)

                total_grad_w = self.alpha * grad_w_distill + grad_w_task
                total_grad_b = self.alpha * grad_b_distill + grad_b_task

                # Descent (minimize distance to teacher on retain set)
                w_student -= self.lr * total_grad_w
                b_student = float(b_student - self.lr * total_grad_b)

        return w_student, b_student


class NullspaceConceptScrubbing:
    """Iterative Nullspace Projection for Concept Erasure & Representation Scrubbing.

    Projects representation vectors onto the orthogonal complement nullspace of protected concepts:
        P_perp = I - V (V^T V + eps * I)^(-1) V^T
        H_scrubbed = H @ P_perp
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize NullspaceConceptScrubbing."""
        self.eps = float(eps)
        self.P_perp: Optional[np.ndarray] = None
        self.concept_basis: Optional[np.ndarray] = None

    def fit(
        self,
        representations: np.ndarray,
        concept_labels: np.ndarray,
        num_directions: int = 1,
    ) -> "NullspaceConceptScrubbing":
        """Compute concept subspace and orthogonal projection matrix.

        Args:
            representations: Feature matrix H of shape (N, d).
            concept_labels: Binary/categorical concept labels (N,).
            num_directions: Number of top concept directions to project out.
        """
        H = np.asarray(representations, dtype=np.float64)
        c = np.asarray(concept_labels)
        N, d = H.shape

        unique_labels = np.unique(c)
        if len(unique_labels) < 2:
            raise ValueError(
                "Need at least 2 distinct concept classes to identify concept directions"
            )

        # Compute mean difference directions across classes
        concept_vectors = []
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                mean_i = np.mean(H[c == unique_labels[i]], axis=0)
                mean_j = np.mean(H[c == unique_labels[j]], axis=0)
                diff = mean_i - mean_j
                norm = np.linalg.norm(diff)
                if norm > self.eps:
                    concept_vectors.append(diff / norm)

        if not concept_vectors:
            self.P_perp = np.eye(d, dtype=np.float64)
            return self

        # Stack concept directions: [k, d] -> transpose to [d, k]
        V = np.stack(concept_vectors, axis=1)

        # Orthonormalize concept basis via SVD / QR
        U, S, Vt = np.linalg.svd(V, full_matrices=False)
        k_keep = min(num_directions, U.shape[1])
        V_ortho = U[:, :k_keep]  # [d, k_keep]

        self.concept_basis = V_ortho

        # P_perp = I - V_ortho V_ortho^T
        self.P_perp = np.eye(d, dtype=np.float64) - V_ortho @ V_ortho.T
        return self

    def transform(self, representations: np.ndarray) -> np.ndarray:
        """Project representations onto the concept nullspace."""
        if self.P_perp is None:
            raise ValueError(
                "NullspaceConceptScrubbing must be fitted before transform()"
            )
        H = np.asarray(representations, dtype=np.float64)
        return H @ self.P_perp

    def scrub_weights(
        self, weight_matrix: np.ndarray, mode: str = "input"
    ) -> np.ndarray:
        """Scrub linear layer weight matrix to neutralize concept interactions.

        Args:
            weight_matrix: Weight tensor W.
            mode: 'input' (neutralizes reading from concept) or 'output' (neutralizes writing to concept).
        """
        if self.P_perp is None:
            raise ValueError(
                "NullspaceConceptScrubbing must be fitted before scrub_weights()"
            )
        W = np.asarray(weight_matrix, dtype=np.float64)

        if mode == "input":
            # If W is (d_in, d_out)
            if W.shape[0] == self.P_perp.shape[0]:
                return self.P_perp @ W
            # If W is (d_out, d_in)
            elif W.shape[1] == self.P_perp.shape[0]:
                return W @ self.P_perp
            else:
                raise ValueError(
                    f"Shape mismatch between W {W.shape} and P_perp {self.P_perp.shape}"
                )
        elif mode == "output":
            if W.shape[0] == self.P_perp.shape[0]:
                return self.P_perp @ W
            elif W.shape[1] == self.P_perp.shape[0]:
                return W @ self.P_perp
            else:
                raise ValueError(
                    f"Shape mismatch between W {W.shape} and P_perp {self.P_perp.shape}"
                )
        else:
            raise ValueError(f"Unsupported mode '{mode}'")
