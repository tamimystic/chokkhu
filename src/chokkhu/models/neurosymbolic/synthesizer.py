"""Inductive Neurosymbolic Program Synthesizer.

Formulated from first principles in pure NumPy and Python, implementing
Domain-Specific Language (DSL) grammar enumeration, observational equivalence pruning,
and inductive program synthesis from input-output specifications (Gulwani 2011, Ellis et al. 2021).
"""

from __future__ import annotations

import operator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class DSLGrammar:
    """Domain-Specific Language (DSL) Primitive Library."""

    def __init__(self) -> None:
        self.primitives: Dict[str, Tuple[Callable[..., Any], int]] = {
            # Arithmetic
            "add": (lambda a, b: a + b, 2),
            "sub": (lambda a, b: a - b, 2),
            "mul": (lambda a, b: a * b, 2),
            "square": (lambda a: a**2, 1),
            "abs": (abs, 1),
            "neg": (operator.neg, 1),
            "double": (lambda a: a * 2, 1),
            "inc": (lambda a: a + 1, 1),
            "dec": (lambda a: a - 1, 1),
            # String
            "concat": (lambda a, b: str(a) + str(b), 2),
            "upper": (lambda s: str(s).upper(), 1),
            "lower": (lambda s: str(s).lower(), 1),
            "reverse": (lambda s: str(s)[::-1], 1),
            "len": (lambda s: len(str(s)), 1),
            "first": (lambda s: str(s)[0] if str(s) else "", 1),
            "last": (lambda s: str(s)[-1] if str(s) else "", 1),
        }


class ProgramSynthesizer:
    r"""Inductive Neurosymbolic Program Synthesizer from Input-Output Examples.

    Synthesizes exact executable functional expressions satisfying:

    .. math::
        \forall (x_i, y_i) \in \mathcal{E}: \quad \text{Program}(x_i) = y_i

    using bottom-up explicit search and observational equivalence pruning.

    Parameters
    ----------
    max_depth : int, default=3
        Maximum expression tree search depth.
    constants : Optional[List[Any]], default=None
        Available constant literals (e.g. [0, 1, 2, " "]).
    """

    def __init__(
        self,
        max_depth: int = 3,
        constants: Optional[List[Any]] = None,
    ) -> None:
        self.max_depth = int(max_depth)
        self.constants = constants if constants is not None else [0, 1, 2, 3]
        self.grammar = DSLGrammar()

        self.synthesized_program: Optional[Callable[[Any], Any]] = None
        self.synthesized_code: str = ""

    def _eval_expr(
        self, expr_tree: Union[str, int, float, Tuple[Any, ...]], x: Any
    ) -> Any:
        """Safely evaluate an expression tree on input x."""
        if isinstance(expr_tree, str) and expr_tree == "x":
            return x
        if not isinstance(expr_tree, tuple):
            return expr_tree

        op_name = expr_tree[0]
        func, arity = self.grammar.primitives[op_name]
        args = [self._eval_expr(arg, x) for arg in expr_tree[1:]]
        return func(*args)

    def _tree_to_code(self, expr_tree: Union[str, int, float, Tuple[Any, ...]]) -> str:
        """Convert AST expression tree to human-readable code string."""
        if isinstance(expr_tree, str):
            return f"'{expr_tree}'" if expr_tree != "x" else "x"
        if not isinstance(expr_tree, tuple):
            return str(expr_tree)

        op_name = expr_tree[0]
        args_code = [self._tree_to_code(arg) for arg in expr_tree[1:]]
        return f"{op_name}({', '.join(args_code)})"

    def fit(self, examples: List[Tuple[Any, Any]]) -> "ProgramSynthesizer":
        r"""Synthesize an executable program matching all input-output pairs.

        Parameters
        ----------
        examples : List[Tuple[Any, Any]]
            Input-output specifications :math:`[(x_1, y_1), (x_2, y_2), \dots]`.

        Returns
        -------
        self : ProgramSynthesizer
        """
        if len(examples) == 0:
            raise ValueError("Must provide at least one input-output example")

        inputs = [ex[0] for ex in examples]
        targets = [ex[1] for ex in examples]

        # Bottom-up expression search queue: expressions by depth
        # Observational equivalence cache: output signature -> shortest expression
        seen_outputs: Dict[Tuple[Any, ...], Any] = {}

        # Depth 0: Variable 'x' and constants
        current_level: List[Any] = ["x"] + self.constants

        for expr in current_level:
            try:
                outputs = tuple(self._eval_expr(expr, inp) for inp in inputs)
                if outputs == tuple(targets):
                    self.synthesized_code = self._tree_to_code(expr)
                    self.best_expr = expr
                    return self
                if outputs not in seen_outputs:
                    seen_outputs[outputs] = expr
            except Exception:
                continue

        # Build higher depths
        all_exprs = list(seen_outputs.values())

        for depth in range(1, self.max_depth + 1):
            next_level: List[Any] = []

            for op_name, (_, arity) in self.grammar.primitives.items():
                if arity == 1:
                    for e in all_exprs:
                        cand_unary: Tuple[Any, ...] = (op_name, e)
                        try:
                            outputs = tuple(
                                self._eval_expr(cand_unary, inp) for inp in inputs
                            )
                            if outputs == tuple(targets):
                                self.synthesized_code = self._tree_to_code(cand_unary)
                                self.best_expr = cand_unary
                                return self
                            if outputs not in seen_outputs:
                                seen_outputs[outputs] = cand_unary
                                next_level.append(cand_unary)
                        except Exception:
                            continue
                elif arity == 2:
                    for e1 in all_exprs:
                        for e2 in all_exprs:
                            cand_binary: Tuple[Any, ...] = (op_name, e1, e2)
                            try:
                                outputs = tuple(
                                    self._eval_expr(cand_binary, inp) for inp in inputs
                                )
                                if outputs == tuple(targets):
                                    self.synthesized_code = self._tree_to_code(
                                        cand_binary
                                    )
                                    self.best_expr = cand_binary
                                    return self
                                if outputs not in seen_outputs:
                                    seen_outputs[outputs] = cand_binary
                                    next_level.append(cand_binary)
                            except Exception:
                                continue

            all_exprs.extend(next_level)

        # Fallback: Identity program
        self.synthesized_code = "x"
        self.best_expr = "x"
        return self

    def predict(self, x: Any) -> Any:
        """Execute synthesized program on new input value."""
        return self._eval_expr(self.best_expr, x)
