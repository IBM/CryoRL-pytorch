from typing import Any, Dict, Optional, Sequence, Tuple, Union, Type
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ModuleType = Type[nn.Module]
from tianshou.data import Batch, to_torch, to_torch_as
from tianshou.utils.net.common import MLP, Net

def apply_logit_mask( logits: torch.Tensor, mask: Optional[np.ndarray]
) -> torch.Tensor:
    """Compute the q value based on the network's raw output and action mask."""
    if mask is not None:
        # the masked q value should be smaller than logits.min()
        min_value = logits.min() - logits.max() - 1.0
        logits = logits + to_torch_as(1 - mask, logits) * min_value
    return logits

class NetV3(Net):   # Net parameters to be configured in CONF FILE
    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,) -> None:

        super().__init__(state_shape,
                         action_shape,           #action space = 1
                         hidden_sizes,
                         norm_layer,
                         activation,
                         device,
                         softmax,
                         concat,
                         num_atoms,
                         dueling_param)

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},) -> Tuple[torch.Tensor, Any]:
        s = torch.as_tensor(
            s, device=self.device, dtype=torch.float32)  # type: ignore
        s = s.view((-1,) + s.shape[-1:])
        logits, state = super().forward(s, state, info)

        return logits, state



class ActorV2(nn.Module):
    """Simple actor network.
    Will create an actor operated in discrete action space with structure of
    preprocess_net ---> action_shape.
    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param bool softmax_output: whether to apply a softmax layer over the last
        layer's output.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    .. seealso::
        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        print (input_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
            device=self.device
        )
        self.softmax_output = softmax_output

    def forward(
        self,
        #s: Union[np.ndarray, torch.Tensor],
        batch: Batch,
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = batch['obs']
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        logits, h = self.preprocess(obs_, state)
        logits = self.last(logits)

        num_batch = obs_.shape[0]
        logits = logits.view(num_batch, -1)

        mask = getattr(obs, "mask", None)
        if mask is not None:
            logits = apply_mask(logits, mask)
        #print (logits.shape)

        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        return logits, h


class CriticV2(nn.Module):
    """Simple critic network. Will create an actor operated in discrete \
    action space with structure of preprocess_net ---> 1(q value).
    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int last_size: the output dimension of Critic network. Default to 1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    .. seealso::
        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        last_size: int = 1,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = last_size
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            last_size,
            hidden_sizes,
            device=self.device
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(last_size)

    def forward(
        #self, s: Union[np.ndarray, torch.Tensor], **kwargs: Any
        self, batch: Batch, **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
#        logits, _ = self.preprocess(s, state=kwargs.get("state", None))
        obs = batch['obs']
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        logits, h = self.preprocess(obs_, state=kwargs.get("state", None))
        logits = self.last(logits)

        num_batch = obs_.shape[0]
        logits = logits.view(num_batch, -1)
        mask = getattr(obs, "mask", None)
        if mask is not None:
            logits = apply_mask(logits, mask)
        logits = self.avg_pool(logits.unsqueeze(0)).squeeze(0)
        #print (logits.shape)
        return logits
