import torch
from torch import nn
from typing import Any, Dict, List, Type, Tuple, Union, Optional, Sequence
ModuleType = Type[nn.Module]
import numpy as np
from tianshou.utils.net.common import Net


class NetV1(nn.Module):   # Net parameters to be configured in CONF FILE
    def __init__(self, in_feature_dim, out_feature_dim, hidden_dim):
        super().__init__() #23*4

        self.model = nn.Sequential(*[
            nn.Linear(in_feature_dim, hidden_dim), nn.ReLU(inplace=True),  # change the size of hidden layer dimension
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_dim, out_feature_dim)
        ])

    def forward(self, obs,  state=None, info={}):
        input_feature = obs
        if not isinstance(input_feature, torch.Tensor):
            input_feature = torch.tensor(input_feature, dtype=torch.float).cuda()
        batch, num_holes, _ = input_feature.shape
        input_feature = input_feature.view(batch*num_holes, -1)
        logits_output = self.model(input_feature)
        logits_output = torch.reshape(logits_output, (batch, -1))

        if hasattr(info, 'visual_sim'):
            logits_output += torch.as_tensor(info['visual_sim'], device=self.device, dtype=torch.float32)

        return logits_output, state

class NetV2(Net):   # Net parameters to be configured in CONF FILE
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
                         int(1),           #action space = 1
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
        num_batch = s.shape[0]
        s = torch.as_tensor(
            s, device=self.device, dtype=torch.float32)  # type: ignore
        s = s.view((-1,) + s.shape[-1:])
        logits, state = super().forward(s, state, info)
        logits = logits.view(num_batch, -1) if self.num_atoms <= 1 else logits.view(num_batch, -1, self.num_atoms)

        if hasattr(info, 'visual_sim'):
            #print (info['visual_sim'].shape)
            #print (logits.shape)
            logits += torch.as_tensor(info['visual_sim'], device=self.device, dtype=torch.float32)
            #I = torch.argsort(logits, dim=1, descending=True)
            #print (I.shape)
            #x = logits[:, I]
            #y = info['visual_sim'][:,I.cpu().numpy()]
            #print ('---', x[0,0:10])
            #print ('===', y[0,0:10])

        return logits, state

class DummyNet(Net):   # Net parameters to be configured in CONF FILE
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
                         int(1),           #action space = 1
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
        #print (info)
        logits = torch.zeros(s.shape[0:2]).cuda(device=self.device)
        if hasattr(info, 'planning'):
            logits += torch.as_tensor(
                info['planning'], device=self.device, dtype=torch.float32)  # type: ignore

        return logits, state
