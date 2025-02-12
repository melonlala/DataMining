import torch
import torch.nn as nn
from torch import Tensor


class NegativeSamplingLoss(nn.Module):
    """The negative sampling loss function.

    Args:
        eps (float, optional): For numerical stability. Defaults to 1e-7.
    """

    def __init__(self, eps: float = 1e-7):

        super().__init__()
        self.eps = eps

    def forward(
        self,
        cur_embs: torch.Tensor,
        pos_embs: torch.Tensor,
        neg_embs: torch.Tensor,
    ) -> Tensor:
        # current: b,h
        # pos_embeddings: b,n_pos,h
        # neg_embeddings: b,n_neg,h

        B, H = cur_embs.shape
        cur_embs = cur_embs.reshape(B, 1, H)  # unsqueeze dim 1 for broadcasting
        pos_embs = pos_embs.reshape(B, -1, H)
        neg_embs = neg_embs.reshape(B, -1, H)

        # TODO (Task 3)
        # Calculate the negative sampling loss.
        #
        # The implementation of this loss might vary depending on how you deal with
        # the multiple positive and negative samples for each node.
        # We do not require your implementation to be exactly the same as the one
        # presented in the lecture slides, but it should be conceptually similar.
        #
        # Be careful that the loss sometimes goes to NaN.
        # We leave it up to you to figure out how to prevent / mitigate this issue.

        ##################
        # Your Code Here #
        ##################

        pos_dot = torch.bmm(pos_embs, cur_embs.transpose(1, 2)).squeeze(2)
        neg_dot = torch.bmm(neg_embs, cur_embs.transpose(1, 2)).squeeze(2)
        pos_log_sigmoid = -torch.nn.functional.logsigmoid(pos_dot).mean()
        neg_log_sigmoid = -torch.nn.functional.logsigmoid(-neg_dot).mean()
        loss = pos_log_sigmoid + neg_log_sigmoid

        # End of TODO

        # return loss
        return loss
