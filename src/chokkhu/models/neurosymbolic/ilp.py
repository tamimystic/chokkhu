r"""Differentiable Inductive Logic Programming (\partial ILP) Engine.

Formulated from first principles using continuous T_P deduction operators,
fuzzy logic t-norms, and gradient-based rule induction in pure NumPy.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DifferentiableILP:
    r"""Differentiable Inductive Logic Programming (\partial ILP) Engine.

    Induces symbolic first-order logic rules from relational examples via continuous
    immediate consequence operator T_P deductions:
    v_{t+1} = v_t \oplus \bigoplus_{C \in \text{Rules}} w_C \cdot \bigotimes_{b \in \text{body}(C)} v_t(b)

    Parameters
    ----------
    predicates : List[str]
        List of predicate symbol identifiers (e.g. ["parent", "ancestor"]).
    constants : List[str]
        List of domain entities/constants (e.g. ["alice", "bob", "charlie"]).
    max_steps : int, default=3
        Number of forward T_P deduction steps.
    learning_rate : float, default=0.1
        Gradient descent learning rate for rule clause weights.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        predicates: List[str],
        constants: List[str],
        max_steps: int = 3,
        learning_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.predicates = list(predicates)
        self.constants = list(constants)
        self.max_steps = max(1, int(max_steps))
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        # 1. Enumerate all binary ground atoms: p(c_i, c_j)
        self.ground_atoms: List[Tuple[str, str, str]] = []
        self.atom_to_idx: Dict[Tuple[str, str, str], int] = {}

        idx = 0
        for p in self.predicates:
            for c1 in self.constants:
                for c2 in self.constants:
                    atom = (p, c1, c2)
                    self.ground_atoms.append(atom)
                    self.atom_to_idx[atom] = idx
                    idx += 1

        self.num_atoms = len(self.ground_atoms)

        # 2. Candidate rule templates:
        # Rule 1: p(X, Y) <- q(X, Y)
        # Rule 2: p(X, Y) <- q(Y, X) (symmetry)
        # Rule 3: p(X, Y) <- q(X, Z), r(Z, Y) (transitivity / chaining)
        self.rules: List[Dict[str, Any]] = []
        self._generate_rule_templates()

        self.num_rules = len(self.rules)
        # Rule log-weights
        self.rule_weights = self.rng.randn(self.num_rules) * 0.1

    def _generate_rule_templates(self) -> None:
        """Generate candidate inductive Horn clause templates."""
        for p_head in self.predicates:
            for p_body1 in self.predicates:
                # Direct & Inverse
                self.rules.append(
                    {
                        "head": p_head,
                        "type": "direct",
                        "body": [p_body1],
                    }
                )
                self.rules.append(
                    {
                        "head": p_head,
                        "type": "inverse",
                        "body": [p_body1],
                    }
                )
                # Transitive chaining
                for p_body2 in self.predicates:
                    self.rules.append(
                        {
                            "head": p_head,
                            "type": "chain",
                            "body": [p_body1, p_body2],
                        }
                    )

    def forward_deduction(
        self,
        initial_valuations: np.ndarray,
        rule_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Execute continuous T_P forward deduction steps.

        Parameters
        ----------
        initial_valuations : np.ndarray of shape (num_atoms,)
            Initial truth values in [0, 1] from background knowledge.
        rule_weights : Optional[np.ndarray] of shape (num_rules,), default=None
            Rule clause weights. If None, uses internal parameters.

        Returns
        -------
        final_valuations : np.ndarray of shape (num_atoms,)
            Deduced truth degrees for all ground atoms.
        """
        w = self.rule_weights if rule_weights is None else rule_weights
        # Softmax rule probabilities per head predicate
        probs = np.exp(w - np.max(w))
        probs = probs / (np.sum(probs) + 1e-12)

        v = np.asarray(initial_valuations, dtype=float).copy()

        for _ in range(self.max_steps):
            v_next = v.copy()

            for r_idx, rule in enumerate(self.rules):
                p_head = rule["head"]
                r_type = rule["type"]
                r_prob = probs[r_idx]

                if r_type == "direct":
                    p_body = rule["body"][0]
                    for c1 in self.constants:
                        for c2 in self.constants:
                            h_idx = self.atom_to_idx[(p_head, c1, c2)]
                            b_idx = self.atom_to_idx[(p_body, c1, c2)]
                            derived_val = r_prob * v[b_idx]
                            # Fuzzy probabilistic sum: a + b - a*b
                            v_next[h_idx] = (
                                v_next[h_idx]
                                + derived_val
                                - (v_next[h_idx] * derived_val)
                            )

                elif r_type == "inverse":
                    p_body = rule["body"][0]
                    for c1 in self.constants:
                        for c2 in self.constants:
                            h_idx = self.atom_to_idx[(p_head, c1, c2)]
                            b_idx = self.atom_to_idx[(p_body, c2, c1)]
                            derived_val = r_prob * v[b_idx]
                            v_next[h_idx] = (
                                v_next[h_idx]
                                + derived_val
                                - (v_next[h_idx] * derived_val)
                            )

                elif r_type == "chain":
                    p_b1, p_b2 = rule["body"]
                    for c1 in self.constants:
                        for c2 in self.constants:
                            h_idx = self.atom_to_idx[(p_head, c1, c2)]
                            # Max over intermediate constants Z: v(p_b1(c1, z)) * v(p_b2(z, c2))
                            chain_vals = []
                            for cz in self.constants:
                                b1_idx = self.atom_to_idx[(p_b1, c1, cz)]
                                b2_idx = self.atom_to_idx[(p_b2, cz, c2)]
                                chain_vals.append(v[b1_idx] * v[b2_idx])
                            max_chain = float(np.max(chain_vals)) if chain_vals else 0.0
                            derived_val = r_prob * max_chain
                            v_next[h_idx] = (
                                v_next[h_idx]
                                + derived_val
                                - (v_next[h_idx] * derived_val)
                            )

            v = np.clip(v_next, 0.0, 1.0)

        return v

    def fit(
        self,
        background_facts: List[Tuple[str, str, str]],
        positive_examples: List[Tuple[str, str, str]],
        negative_examples: List[Tuple[str, str, str]],
        epochs: int = 40,
    ) -> "DifferentiableILP":
        """Learn rule weights via gradient descent on classification loss."""
        # Initial truth vector from background facts
        v0: np.ndarray = np.zeros(self.num_atoms, dtype=float)
        for fact in background_facts:
            if fact in self.atom_to_idx:
                v0[self.atom_to_idx[fact]] = 1.0

        pos_indices = [
            self.atom_to_idx[e] for e in positive_examples if e in self.atom_to_idx
        ]
        neg_indices = [
            self.atom_to_idx[e] for e in negative_examples if e in self.atom_to_idx
        ]

        eps = 1e-4

        for _ in range(epochs):
            v_final = self.forward_deduction(v0)

            # Cross entropy loss
            pos_loss: float = (
                -float(np.sum(np.log(np.maximum(v_final[pos_indices], 1e-12))))
                if pos_indices
                else 0.0
            )
            neg_loss: float = (
                -float(np.sum(np.log(np.maximum(1.0 - v_final[neg_indices], 1e-12))))
                if neg_indices
                else 0.0
            )
            base_loss = float(pos_loss + neg_loss)

            # Numerical gradient over rule weights
            grad_w = np.zeros_like(self.rule_weights)
            for r_i in range(self.num_rules):
                w_perturbed = self.rule_weights.copy()
                w_perturbed[r_i] += eps
                v_pert = self.forward_deduction(v0, rule_weights=w_perturbed)

                p_loss: float = (
                    -float(np.sum(np.log(np.maximum(v_pert[pos_indices], 1e-12))))
                    if pos_indices
                    else 0.0
                )
                n_loss: float = (
                    -float(np.sum(np.log(np.maximum(1.0 - v_pert[neg_indices], 1e-12))))
                    if neg_indices
                    else 0.0
                )
                grad_w[r_i] = (float(p_loss + n_loss) - base_loss) / eps

            self.rule_weights -= self.learning_rate * grad_w

        return self

    def predict_atom(
        self,
        background_facts: List[Tuple[str, str, str]],
        query_atom: Tuple[str, str, str],
    ) -> float:
        """Predict confidence truth degree for a query atom."""
        v0: np.ndarray = np.zeros(self.num_atoms, dtype=float)
        for fact in background_facts:
            if fact in self.atom_to_idx:
                v0[self.atom_to_idx[fact]] = 1.0

        v_final = self.forward_deduction(v0)
        if query_atom in self.atom_to_idx:
            return float(v_final[self.atom_to_idx[query_atom]])
        return 0.0
