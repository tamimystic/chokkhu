"""Base Estimator Framework and Scikit-Learn Standard Protocol Compliance for Chokkhu."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np

from chokkhu.core.exceptions import NotFittedError


class ChokkhuModel:
    """Base class for all estimators in Chokkhu, providing scikit-learn standard compliance."""

    _estimator_type: Optional[str] = None

    def fit(self, X: Any, y: Any = None) -> Any:
        """Fit the model to training data."""
        raise NotImplementedError

    def predict(self, X: Any) -> Any:
        """Predict target labels or values for input data."""
        raise NotImplementedError

    def predict_proba(self, X: Any) -> Any:
        """Predict class probabilities for input data."""
        raise NotImplementedError

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Args:
            deep: If True, will return the parameters for this estimator and
                  contained subobjects that are estimators.
        Returns:
            Parameter names mapped to their values.
        """
        out: Dict[str, Any] = {}
        init_signature = inspect.signature(self.__class__.__init__)
        for param in init_signature.parameters.values():
            if param.name == "self" or param.kind in (
                param.VAR_POSITIONAL,
                param.VAR_KEYWORD,
            ):
                continue
            value = getattr(self, param.name, None)
            out[param.name] = value
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params(deep=True).items()
                out.update((f"{param.name}__{k}", val) for k, val in deep_items)
        return out

    def set_params(self, **params: Any) -> ChokkhuModel:
        """Set the parameters of this estimator.

        Returns:
            Estimator instance (self).
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        nested_params: Dict[str, Dict[str, Any]] = {}

        for key, value in params.items():
            key_split = key.split("__", 1)
            if len(key_split) == 2:
                sub_obj, sub_param = key_split
                if sub_obj not in nested_params:
                    nested_params[sub_obj] = {}
                nested_params[sub_obj][sub_param] = value
            else:
                if key not in valid_params:
                    raise ValueError(
                        f"Invalid parameter '{key}' for estimator {self.__class__.__name__}. "
                        f"Valid parameters are: {list(valid_params.keys())}."
                    )
                setattr(self, key, value)

        for sub_obj, sub_params in nested_params.items():
            if hasattr(self, sub_obj):
                getattr(self, sub_obj).set_params(**sub_params)

        return self

    def score(
        self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Return the coefficient of determination R^2 or accuracy score.

        Args:
            X: Test samples.
            y: True values for X.
            sample_weight: Sample weights.
        Returns:
            Score value (accuracy for classifier, R2 for regressor).
        """
        from chokkhu.evaluation.metrics import accuracy_score, r2_score

        y_pred = self.predict(X)
        y_true_arr = np.asarray(y)

        # Check if classifier or regressor
        if getattr(self, "_estimator_type", None) == "classifier" or (
            hasattr(self, "classes_") or issubclass(self.__class__, ClassifierMixin)
        ):
            return float(accuracy_score(y_true_arr, y_pred))
        else:
            return float(r2_score(y_true_arr, y_pred))

    def _validate_data(
        self,
        X: Any,
        y: Any = None,
        reset: bool = True,
        cast_to_float: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate input data array shapes and types."""
        from chokkhu.core.tensor import Tensor

        if isinstance(X, Tensor):
            X_arr = X.data
        else:
            X_arr = np.asarray(X)

        if cast_to_float and not np.issubdtype(X_arr.dtype, np.floating):
            X_arr = X_arr.astype(np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if reset:
            self.n_features_in_ = X_arr.shape[1] if X_arr.ndim > 1 else 1

        y_arr = None
        if y is not None:
            if isinstance(y, Tensor):
                y_arr = y.data
            else:
                y_arr = np.asarray(y)

            if len(X_arr) != len(y_arr):
                raise ValueError(
                    f"Found input variables with inconsistent numbers of samples: "
                    f"[{len(X_arr)}, {len(y_arr)}]"
                )

        return X_arr, y_arr

    def __repr__(self) -> str:
        params = self.get_params(deep=False)
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({params_str})"


class ClassifierMixin:
    """Mixin class for all classifiers in Chokkhu."""

    _estimator_type = "classifier"

    def score(
        self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        from chokkhu.evaluation.metrics import accuracy_score

        return float(accuracy_score(np.asarray(y), self.predict(X)))  # type: ignore[attr-defined]


class RegressorMixin:
    """Mixin class for all regressors in Chokkhu."""

    _estimator_type = "regressor"

    def score(
        self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        from chokkhu.evaluation.metrics import r2_score

        return float(r2_score(np.asarray(y), self.predict(X)))  # type: ignore[attr-defined]


BaseEstimator = ChokkhuModel


def check_is_fitted(
    estimator: Any,
    attributes: Optional[Union[str, Sequence[str]]] = None,
    msg: Optional[str] = None,
    all_or_any: Any = all,
) -> None:
    """Perform is_fitted validation for estimator.

    Args:
        estimator: Estimator instance to check.
        attributes: Attribute name(s) given as string or list/tuple of strings.
        msg: The custom error message to use.
    """
    if msg is None:
        msg = (
            f"This {estimator.__class__.__name__} instance is not fitted yet. "
            f"Call 'fit' with appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError(f"{estimator} is not an estimator, missing 'fit' method.")

    # Explicit is_fitted attribute check
    if hasattr(estimator, "is_fitted") and not estimator.is_fitted:
        raise NotFittedError(msg)
    if hasattr(estimator, "_is_fitted") and not estimator._is_fitted:
        raise NotFittedError(msg)

    if attributes is not None:
        if isinstance(attributes, str):
            attrs = [attributes]
        else:
            attrs = list(attributes)
        for attr in attrs:
            if not hasattr(estimator, attr):
                raise NotFittedError(msg)
            val = getattr(estimator, attr)
            if val is None or (isinstance(val, np.ndarray) and val.size == 0):
                raise NotFittedError(msg)
    else:
        # Check trailing underscore attributes
        fitted_attrs = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]
        if not fitted_attrs:
            raise NotFittedError(msg)
        for attr in fitted_attrs:
            val = getattr(estimator, attr)
            if isinstance(val, np.ndarray) and val.size == 0:
                raise NotFittedError(msg)


def clone(estimator: Any, safe: bool = True) -> Any:
    """Construct a new unfitted estimator with the same parameters."""
    if estimator is None:
        return None
    if not hasattr(estimator, "get_params"):
        return copy.deepcopy(estimator)
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    return new_object
