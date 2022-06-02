import torch
from torch import nn, Tensor
import copy

from typing import List, Tuple, Union, Callable

LossFnType = Union[Callable[[nn.Module, Tensor], Tensor], Callable[[nn.Module, Tuple[Tensor, ...]], Tensor]]
BatchType = Union[Tensor, Tuple[Tensor, ...]]
DerRetType = Union[Tuple[Tensor, ...], Tensor]


def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    for i in range(len(names[:-1])):
        obj = getattr(obj, names[i], None)
        if obj is None:
            return
    if hasattr(obj, names[-1]):
        delattr(obj, names[-1])


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor, is_nn_param: bool = False) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _set_nested_attr(obj, ['conv', 'weight'], value)
    """
    for i in range(len(names[:-1])):
        obj = getattr(obj, names[i], None)
        if obj is None:
            return
    if is_nn_param:
        setattr(obj, names[-1], nn.Parameter(value))
    else:
        setattr(obj, names[-1], value)


def count_parameters(model: nn.Module, count_non_differentiable: bool = True) -> int:
    """
    This function counts the overall number of scalar parameters in the model.
    
    Args:
        model (nn.Module): The model with parameters to count.
        count_non_differentiable (bool, optional): If set "True" the whole set of parameters is affected.
            If set "False" only differentiable parameters are counted. Defaults to "True".
    
    Returns:
        int: The return value. The number of parametrs the model stores.
    """
    if count_non_differentiable:
        return sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...], is_nn_param: bool = False) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p, is_nn_param)


class Oracle(object):
    """
    This class works as wrapper around model with only differentiable parameters. Takes model instance
    and loss function which evaluates the model quality. The class supports interaction with parameters,
    view them as 1d Torch tensors, gets and sets parameter values. The Oracle computes loss function value,
    gradient in two ways, hessian.
    
    Attributes:
        _model (nn.Module): The model with differentiable parameters.
        _loss_fn (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
            instances. Returns differentiable Tensor scalar.
    """
    
    def __init__(self, model: nn.Module, loss_fn: LossFnType, inplace_copy_model: bool = False) -> None:
        """
        Class constructor.
        
        Args:
            model (nn.Module): The model with differentiable parameters.
            loss_fn (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
                instances. Returns differentiable Tensor scalar.
            inplace_copy_model (bool, optional): This flag defines whether to perform deep copy of the model
                or to use the same referenced instance. Defaults to "False".
        """
        if inplace_copy_model:
            self._model = copy.deepcopy(model)
        else:
            self._model = model
        self._loss_fn = loss_fn
    
    def get_flat_params(self, detach: bool = True) -> Tensor:
        """
        Use this method to get the model parameters copy in 1d vector form.
        
        Args:
            detach (bool, optional): If equals "False", the result 1d Tensor is treated as differentiable
                vector-function of nn.Parameter instances. Defaults to "True".
        
        Returns:
            Tensor: 1d Tensor of the model parameters.
        """
        views = []
        for p in self._model.parameters():
            if detach:
                views.append(p.detach().view(-1))
            else:
                views.append(p.view(-1))
        return torch.cat(views, 0)
    
    @torch.no_grad()
    def set_flat_params(self, flat_params: Tensor) -> None:
        """
        Use this method to set the model parameters from 1d vector form.
        
        Args:
            flat_params (Tensor): 1d float Tensor used to store model parameters.
        """
        offset = 0
        for p in self._model.parameters():
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.copy_(flat_params[offset:offset + numel].view_as(p))
            offset += numel
    
    @torch.no_grad()
    def loss_function_val(self, signal_batch: BatchType) -> Tensor:
        """
        This method computes loss function value and returns nondifferentiable scalar Tensor.
        
        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality.
        
        Returns:
            float scalar Tensor: The loss function value. This value is nondifferentiable.
        """
        return self._loss_fn(self._model, signal_batch)
    
    def hessian(self, signal_batch: BatchType, compute_fn_val: bool = False) -> DerRetType:
        """
        This method computes hessian value of the loss function and optionally returns loss function value
        as nondifferentiable scalar. The hessian is computed using torch.autograd.functional.hessian function.
        
        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality.
            compute_fn_val (bool, optional): If set "True" the loss function value is returned. Defaults to "False".
            outer_jacobian_strategy (str, optional): The Hessian is computed by computing the Jacobian of a Jacobian. The inner Jacobian is always computed in reverse-mode AD.
                Setting strategy to "forward-mode" or "reverse-mode" determines whether the outer Jacobian will be computed with forward or reverse mode AD.
                Currently, computing the outer Jacobian in "forward-mode" requires vectorized=True. Defaults to "reverse-mode".
        
        Returns:
            float scalar Tensor, optional: The loss function value. This value is nondifferentiable.
            Tensor: The hessian value.
        """
        #self._model.zero_grad()
        if compute_fn_val:
            loss_val = self.loss_function_val(signal_batch)
        
        params, names = extract_weights(self._model)
        
        def f(*weights):
            load_weights(self._model, names, weights)
            return self._loss_fn(self._model, signal_batch)
        
        H_res = torch.autograd.functional.hessian(f, tuple(params), create_graph=False, vectorize=True)
        load_weights(self._model, names, params, is_nn_param=True)
        
        param_size = count_parameters(self._model)
        H = torch.zeros(param_size, param_size, dtype=H_res[0][0].dtype, device=H_res[0][0].device)
        param_list_len = len(params)
        row_offset, col_offset = 0, 0
        for row in range(param_list_len):
            row_size = params[row].numel()
            for col in range(param_list_len):
                col_size = params[col].numel()
                H[row_offset:row_offset + row_size, col_offset:col_offset + col_size] = H_res[row][col].view(
                    row_size, col_size)
                col_offset += col_size
            row_offset += row_size
            col_offset = 0
        self._model.zero_grad()
        
        if compute_fn_val:
            return loss_val, H
        return H
    
    def gradient_through_jacobian(self, signal_batch: BatchType, compute_fn_val: bool = False) -> DerRetType:
        """
        This method computes gradient value of the loss function and optionally returns loss function value
        as nondifferentiable scalar. The gradient is computed using torch.autograd.functional.jacobian function.
        
        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality.
            compute_fn_val (bool, optional): If set "True" the loss function value is returned. Defaults to "False".
        
        Returns:
            float scalar Tensor, optional: The loss function value. This value is nondifferentiable.
            Tensor: The gradient value.
        """
        #self._model.zero_grad()
        if compute_fn_val:
            loss_val = self.loss_function_val(signal_batch)
        
        params, names = extract_weights(self._model)
        
        def f(*weights):
            load_weights(self._model, names, weights)
            return self._loss_fn(self._model, signal_batch)
        
        J = torch.autograd.functional.jacobian(f, tuple(params), create_graph=False, vectorize=True)
        load_weights(self._model, names, params, is_nn_param=True)
        
        J_res = []
        for J in J:
            J_res.append(J.detach().view(1, J.numel()))
        J = torch.cat(J_res, dim=1)
        self._model.zero_grad()
        
        if compute_fn_val:
            return loss_val, J.view(J.numel())
        return J.view(J.numel())
    
    def gradient(self, signal_batch: BatchType, compute_fn_val: bool = False) -> DerRetType:
        """
        This method computes gradient value of the loss function and optionally returns loss function value.
        The gradient is computed using .backward() method called on loss function value.
        
        Args:
            signal_batch (tuple of Tensor instances): The batch of signals used to compute model quality.
            compute_fn_val (bool, optional): If set "True" the loss function value is returned. Defaults to "False".
        
        Returns:
            float scalar Tensor, optional: The loss function value. This value is differentiable.
            Tensor: The gradient value.
        """
        self._model.zero_grad()
        loss_val = self._loss_fn(self._model, signal_batch)
        loss_val.backward()
        
        views = []
        params = tuple(self._model.parameters())
        for p in params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        flat_grad = torch.cat(views, 0)
        
        self._model.zero_grad()
        
        if compute_fn_val:
            return loss_val, flat_grad
        return flat_grad
