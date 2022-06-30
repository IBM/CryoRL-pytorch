import numpy as np
import torch
from tianshou.policy import DQNPolicy
from typing import Any, Dict, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as

'''
class VisualSimilarity:
    def __init__(self):
        super(VisualSimilarity, self).__init__()
        # the last dim
        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    def compute_visual_similarity(self, info):
        features = getattr(info, 'visual_features', None)
        history = getattr(info, 'history', None)
 
        if features is None:
            return 0.0

        num_features = features.shape[1] # second dim
        history = history[0]

        positive_sim = 0.0
        print (features.shape)
        positive_idx = [k for k, v in enumerate(history)  if v > 0]
        print ('++++', positive_idx)
        if len(positive_idx) > 0:
            positive_sim = [torch.mean(self.cos(features[:,k:k+1], features[:, positive_idx]),dim=1) for k in range(num_features)]
            #print (positive_sim[0])
            positive_sim = torch.stack(positive_sim).transpose(0,1)
            print (positive_sim.shape)

        negative_sim = 0.0
        negative_idx = [k for k, v in enumerate(history)  if v < 0]
        print ('---', negative_idx)
        if len(negative_idx) > 0:
            negative_sim = [torch.mean(self.cos(features[:,k:k+1], features[:, negative_idx]),dim=1) for k in range(num_features)]
            negative_sim = torch.stack(negative_sim).transpose(0,1)
            print (negative_sim.shape)

        #print (positive_sim.shape, negative_sim.shape)
        return positive_sim / (positive_sim+negative_sim+1e-5)
'''

class DQNWithSimilarityPolicy(DQNPolicy):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, optim, discount_factor, estimation_step,target_update_freq, reward_normalization, is_double, **kwargs)
        self.visual_sim = VisualSimilarity()

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.
        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::
            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )
        :param float eps: in [0, 1], for epsilon-greedy exploration method.
        :return: A :class:`~tianshou.data.Batch` which has 3 keys:
            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.
        .. seealso::
            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        logits, h = model(obs_, state=state, info=batch.info)
        #visual_sim = self.visual_sim.compute_visual_similarity(info=batch.info)
#        logits +=
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=h)
