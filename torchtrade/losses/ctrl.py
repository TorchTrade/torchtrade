from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey
from torch import nn

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _reduce


class CTRLLoss(LossModule):
    """Cross-Trajectory Representation Learning (CTRL) Loss.

    CTRL is a self-supervised representation learning method for RL that improves
    zero-shot generalization by training encoders to recognize behavioral similarity
    across trajectories without using rewards.

    The loss combines:
    - Prototype-based contrastive learning using Sinkhorn algorithm for soft cluster assignments
    - MYOW (Make Your Own Winners) loss for cross-trajectory representation consistency

    Reference: "Cross-Trajectory Representation Learning for Zero-Shot Generalization in RL"
    https://arxiv.org/abs/2106.02193

    Args:
        encoder_network (TensorDictModule): The encoder network that produces representations.
            This is typically the shared encoder used by both actor and critic.
        embedding_dim (int): Dimension of the encoder output embeddings.

    Keyword Args:
        projection_dim (int, optional): Dimension of the projection space for prototypes.
            Defaults to 128.
        num_prototypes (int, optional): Number of learnable prototype vectors.
            Defaults to 512.
        sinkhorn_iters (int, optional): Number of Sinkhorn-Knopp iterations for
            computing soft cluster assignments. Defaults to 3.
        temperature (float, optional): Temperature parameter for softmax and Sinkhorn.
            Lower values make assignments more peaked. Defaults to 0.1.
        window_len (int, optional): Length of sliding window for trajectory segments.
            Defaults to 4.
        myow_k (int, optional): Number of nearest prototypes to consider in MYOW loss.
            Defaults to 5.
        myow_coeff (float, optional): Coefficient for the MYOW loss term.
            Defaults to 1.0.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. Defaults to ``"mean"``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.objectives.ctrl import CTRLLoss
        >>> # Create a simple encoder
        >>> encoder = TensorDictModule(
        ...     nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 128)),
        ...     in_keys=["observation"],
        ...     out_keys=["embedding"]
        ... )
        >>> # Create CTRL loss
        >>> ctrl_loss = CTRLLoss(encoder, embedding_dim=128)
        >>> # Create sample data (batch of trajectory windows)
        >>> batch_size = 32
        >>> window_len = 4
        >>> data = TensorDict({
        ...     "observation": torch.randn(batch_size, window_len, 64),
        ... }, batch_size=[batch_size, window_len])
        >>> # Compute loss
        >>> loss_td = ctrl_loss(data)
        >>> loss_td["loss_ctrl"]
        tensor(..., grad_fn=<...>)
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using
        '.set_keys(key_name=key_value)' and their default values.

        Attributes:
            observation (NestedKey): The input tensordict key for observations.
                Defaults to ``"observation"``.
            embedding (NestedKey): The output key for encoder embeddings.
                Defaults to ``"embedding"``.
        """

        observation: NestedKey = "observation"
        embedding: NestedKey = "embedding"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys

    encoder_network: TensorDictModule

    def __init__(
        self,
        encoder_network: TensorDictModule,
        embedding_dim: int,
        *,
        projection_dim: int = 128,
        num_prototypes: int = 512,
        sinkhorn_iters: int = 3,
        temperature: float = 0.1,
        window_len: int = 4,
        myow_k: int = 5,
        myow_coeff: float = 1.0,
        reduction: str | None = None,
    ):
        if reduction is None:
            reduction = "mean"

        self._in_keys = None
        self._out_keys = None
        super().__init__()

        # Store the encoder network (not functionalized - we just use it for forward passes)
        self.encoder_network = encoder_network
        self.embedding_dim = embedding_dim

        # Hyperparameters
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("sinkhorn_iters", torch.tensor(sinkhorn_iters))
        self.register_buffer("myow_k", torch.tensor(myow_k))
        self.register_buffer("myow_coeff", torch.tensor(myow_coeff))
        self.window_len = window_len
        self.reduction = reduction

        # Projection head: maps encoder output to projection space
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

        # Prediction head: for contrastive prediction
        self.prediction_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # Learnable prototype vectors
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, projection_dim))
        nn.init.xavier_uniform_(self.prototypes)

        self.num_prototypes = num_prototypes
        self.projection_dim = projection_dim

    @property
    def functional(self):
        """CTRL loss is not functional - it uses the encoder directly."""
        return False

    def _set_in_keys(self):
        keys = list(self.encoder_network.in_keys)
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    def out_keys(self):
        if self._out_keys is None:
            self._out_keys = ["loss_ctrl", "loss_proto", "loss_myow"]
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        """CTRL does not use a value estimator."""
        pass

    @staticmethod
    def _sinkhorn(
        scores: torch.Tensor, iters: int = 3, epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Apply Sinkhorn-Knopp algorithm for balanced soft assignments.

        The Sinkhorn algorithm iteratively normalizes rows and columns to obtain
        a doubly-stochastic matrix, which provides balanced cluster assignments.

        Args:
            scores: Similarity scores of shape (batch_size, num_prototypes)
            iters: Number of Sinkhorn iterations
            epsilon: Small constant for numerical stability

        Returns:
            Soft assignment matrix of shape (batch_size, num_prototypes)
        """
        Q = torch.exp(scores)
        Q = Q / (Q.sum(dim=0, keepdim=True) + epsilon)

        for _ in range(iters):
            # Row normalization
            Q = Q / (Q.sum(dim=1, keepdim=True) + epsilon)
            # Column normalization
            Q = Q / (Q.sum(dim=0, keepdim=True) + epsilon)

        # Final row normalization to get proper probabilities
        Q = Q / (Q.sum(dim=1, keepdim=True) + epsilon)
        return Q

    def _compute_prototype_loss(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prototype-based contrastive loss.

        Uses the prediction head output and Sinkhorn assignments to compute
        a cross-entropy loss that encourages consistent prototype assignments.

        Args:
            z: Projected embeddings of shape (batch_size, projection_dim)
            p: Predicted embeddings of shape (batch_size, projection_dim)

        Returns:
            Prototype loss scalar
        """
        # Normalize prototypes and embeddings
        prototypes = F.normalize(self.prototypes, dim=1)
        z_norm = F.normalize(z, dim=1)
        p_norm = F.normalize(p, dim=1)

        # Compute similarity scores
        scores_z = torch.mm(z_norm, prototypes.t()) / self.temperature
        scores_p = torch.mm(p_norm, prototypes.t()) / self.temperature

        # Get soft assignments using Sinkhorn (stop gradient for targets)
        with torch.no_grad():
            q_target = self._sinkhorn(
                scores_z.detach(), int(self.sinkhorn_iters.item())
            )

        # Cross-entropy loss between predictions and targets
        log_p = F.log_softmax(scores_p, dim=1)
        loss = -torch.sum(q_target * log_p, dim=1)

        return loss

    def _compute_myow_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MYOW (Make Your Own Winners) loss.

        MYOW encourages representations from different trajectories that map
        to nearby prototypes to be similar, promoting cross-trajectory consistency.

        Args:
            z1: Projected embeddings from first trajectory segment (batch_size, projection_dim)
            z2: Projected embeddings from second trajectory segment (batch_size, projection_dim)

        Returns:
            MYOW loss scalar
        """
        # Normalize prototypes and embeddings
        prototypes = F.normalize(self.prototypes, dim=1)
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        # Compute distances to prototypes
        dist1 = torch.cdist(z1_norm, prototypes)  # (batch_size, num_prototypes)
        dist2 = torch.cdist(z2_norm, prototypes)  # (batch_size, num_prototypes)

        # Find k nearest prototypes for each embedding
        k = min(int(self.myow_k.item()), self.num_prototypes)
        _, indices1 = torch.topk(dist1, k, dim=1, largest=False)  # (batch_size, k)
        _, indices2 = torch.topk(dist2, k, dim=1, largest=False)  # (batch_size, k)

        # Find samples that share at least one prototype in their top-k
        # Create binary masks for prototype membership
        mask1 = torch.zeros(z1.shape[0], self.num_prototypes, device=z1.device)
        mask2 = torch.zeros(z2.shape[0], self.num_prototypes, device=z2.device)
        mask1.scatter_(1, indices1, 1.0)
        mask2.scatter_(1, indices2, 1.0)

        # Compute overlap: (batch_size, batch_size) matrix
        overlap = torch.mm(mask1, mask2.t())  # Number of shared prototypes
        neighbors = overlap > 0  # Boolean mask for pairs with shared prototypes

        if not neighbors.any():
            # No neighbors found, return zero loss
            return torch.zeros(z1.shape[0], device=z1.device)

        # Compute cosine similarity between all pairs
        similarity = torch.mm(z1_norm, z2_norm.t())  # (batch_size, batch_size)

        # MYOW loss: maximize similarity for neighbor pairs
        # We use 1 - similarity as the loss (want to minimize distance)
        loss_matrix = 1 - similarity

        # Mask out non-neighbor pairs and compute mean loss per sample
        loss_matrix = loss_matrix * neighbors.float()
        num_neighbors = neighbors.float().sum(dim=1).clamp(min=1)
        loss = loss_matrix.sum(dim=1) / num_neighbors

        return loss

    def _extract_windows(
        self, batch: TensorDictBase
    ) -> tuple[TensorDictBase, TensorDictBase]:
        """Extract two sets of windows from trajectory for cross-trajectory comparison.

        For CTRL, we need to compare representations across different trajectory
        segments. This function splits the batch into two halves for comparison.

        Args:
            batch: TensorDict batch of shape (batch_size, ...)

        Returns:
            Tuple of two TensorDict batches for comparison
        """
        batch_size = batch.batch_size[0]
        half = batch_size // 2

        if half == 0:
            # Batch too small, duplicate
            return batch, batch

        # Split batch in half for comparison
        batch1 = batch[:half]
        batch2 = batch[half : 2 * half]

        return batch1, batch2

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute CTRL loss.

        Args:
            tensordict: Input tensordict containing observations.
                Expected shape: (batch_size, [window_len], *obs_shape)

        Returns:
            TensorDict with keys:
                - loss_ctrl: Total CTRL loss
                - loss_proto: Prototype contrastive loss component
                - loss_myow: MYOW cross-trajectory loss component
        """
        # Extract two batches for comparison (splits batch in half)
        batch1, batch2 = self._extract_windows(tensordict)

        # Forward through encoder to get embeddings
        # The encoder will handle extracting the necessary keys from each batch
        with torch.set_grad_enabled(True):
            td1 = self.encoder_network(batch1)
            td2 = self.encoder_network(batch2)

        emb_key = self.tensor_keys.embedding
        z1_raw = td1.get(emb_key)
        z2_raw = td2.get(emb_key)

        if z1_raw is None:
            raise KeyError(
                f"Could not find embedding key '{emb_key}' in encoder output. "
                f"Available keys: {list(td1.keys())}. "
                f"Make sure the encoder network outputs to key '{emb_key}'."
            )

        # Project to prototype space
        z1 = self.projection_head(z1_raw)
        z2 = self.projection_head(z2_raw)

        # Predict for contrastive loss
        p1 = self.prediction_head(z1)
        p2 = self.prediction_head(z2)

        # Compute prototype contrastive loss (symmetric)
        loss_proto_1 = self._compute_prototype_loss(z1.detach(), p2)
        loss_proto_2 = self._compute_prototype_loss(z2.detach(), p1)
        loss_proto = (loss_proto_1 + loss_proto_2) / 2

        # Compute MYOW loss
        loss_myow = self._compute_myow_loss(z1, z2)

        # Total loss
        loss_ctrl = loss_proto + self.myow_coeff * loss_myow

        # Build output tensordict
        td_out = TensorDict(
            {
                "loss_ctrl": loss_ctrl,
                "loss_proto": loss_proto,
                "loss_myow": loss_myow,
            },
        )

        # Apply reduction
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction)
            if name.startswith("loss_")
            else value,
        )

        return td_out


class CTRLPPOLoss(LossModule):
    """Combined CTRL and PPO loss for end-to-end training.

    This module combines the ClipPPOLoss with CTRLLoss for joint training
    of the policy and encoder representation. It provides a convenient wrapper
    that handles both losses in a single forward pass.

    Args:
        ppo_loss (LossModule): The PPO loss module (e.g., ClipPPOLoss).
        ctrl_loss (CTRLLoss): The CTRL loss module.

    Keyword Args:
        ctrl_coeff (float, optional): Coefficient for the CTRL loss term.
            Defaults to 1.0.

    Examples:
        >>> from torchrl.objectives import ClipPPOLoss
        >>> from torchrl.objectives.ctrl import CTRLLoss, CTRLPPOLoss
        >>> # Create PPO and CTRL losses
        >>> ppo_loss = ClipPPOLoss(actor, critic)
        >>> ctrl_loss = CTRLLoss(encoder, embedding_dim=128)
        >>> # Combine them
        >>> combined_loss = CTRLPPOLoss(ppo_loss, ctrl_loss, ctrl_coeff=0.5)
        >>> # Forward pass computes both losses
        >>> loss_td = combined_loss(data)
    """

    @dataclass
    class _AcceptedKeys:
        """Accepted keys for CTRLPPOLoss."""

        pass

    default_keys = _AcceptedKeys

    def __init__(
        self,
        ppo_loss: LossModule,
        ctrl_loss: CTRLLoss,
        *,
        ctrl_coeff: float = 1.0,
    ):
        super().__init__()
        self.ppo_loss = ppo_loss
        self.ctrl_loss = ctrl_loss
        self.register_buffer("ctrl_coeff", torch.tensor(ctrl_coeff))
        self._in_keys = None
        self._out_keys = None

    @property
    def functional(self):
        return False

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._in_keys = list(set(self.ppo_loss.in_keys + self.ctrl_loss.in_keys))
        return self._in_keys

    @property
    def out_keys(self):
        if self._out_keys is None:
            self._out_keys = list(self.ppo_loss.out_keys) + list(
                self.ctrl_loss.out_keys
            )
        return self._out_keys

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        self.ppo_loss._forward_value_estimator_keys(**kwargs)

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute combined PPO and CTRL losses.

        Args:
            tensordict: Input tensordict containing observations, actions, etc.

        Returns:
            TensorDict with all PPO loss keys plus CTRL loss keys.
        """
        # Compute PPO loss
        td_ppo = self.ppo_loss(tensordict)

        # Compute CTRL loss
        td_ctrl = self.ctrl_loss(tensordict)

        # Combine outputs
        td_out = td_ppo.clone()
        for key, value in td_ctrl.items():
            td_out.set(key, value)

        # Scale CTRL loss
        if "loss_ctrl" in td_out.keys():
            td_out.set("loss_ctrl", td_out.get("loss_ctrl") * self.ctrl_coeff)

        return td_out

    def make_value_estimator(self, *args, **kwargs):
        """Forward to PPO loss value estimator."""
        return self.ppo_loss.make_value_estimator(*args, **kwargs)
