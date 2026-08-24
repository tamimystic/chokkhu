from .tensor import Tensor
from .backend import xp


def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    # Ensure y_true is a tensor
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true)

    diff = y_pred - y_true
    loss = (diff**2).sum() / Tensor(float(y_pred.shape[0]))
    return loss


def cross_entropy_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true)

    # Numerical stability: subtract max
    # We will do this carefully using backend math inside a custom autograd block
    # to avoid creating a massive graph for standard ops.

    # Actually, let's implement Softmax + NLL natively inside the tensor graph for simplicity right now.
    # Softmax
    max_logits = xp.max(y_pred.data, axis=1, keepdims=True)
    exp_logits = xp.exp(y_pred.data - max_logits)
    probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)

    # NLL
    batch_size = y_pred.shape[0]

    # We need to return a Tensor with backprop defined manually for stability
    # Or just use pure tensor operations if possible.
    # For efficiency, we will write a custom backward for cross entropy.

    true_labels = y_true.data.astype(int)

    if len(true_labels.shape) == 1:
        # one-hot encode
        one_hot = xp.zeros_like(probs)
        one_hot[xp.arange(batch_size), true_labels] = 1.0
    else:
        one_hot = true_labels

    log_probs = xp.log(probs + 1e-8)
    loss_data = -xp.sum(one_hot * log_probs) / batch_size

    out = Tensor(
        loss_data,
        requires_grad=y_pred.requires_grad,
        _children=(y_pred,),
        _op="crossentropy",
    )

    def _backward():
        if y_pred.requires_grad:
            grad = (probs - one_hot) / batch_size
            y_pred.grad += grad

    out._backward = _backward
    return out
