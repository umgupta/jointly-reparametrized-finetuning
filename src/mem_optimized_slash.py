"""
Here we demonstrate how to implement the SLaSh model with memory 
optimization. We generate random projection matrices on the fly by using the 
random number generator's state
"""


import logging

import torch
from torch import nn
from torch.autograd import Function
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaEncoder,
    RobertaLayer,
    RobertaModel,
    RobertaOutput,
    RobertaPooler,
)

from src.models import FullRobertaForSequenceClassification, do_winit, do_zinit

logger = logging.getLogger()


class Projection(Function):
    """
        Custom autograd function to implement the projection layer
        It uses random numbers state to generate the projection matrix on the
        fly
    """
    @staticmethod
    def forward(ctx, z, rng, shape, mode):
        old_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        w = do_winit(shape[0], shape[1], mode, device="cuda")
        torch.set_rng_state(old_rng)

        result = (z @ w).detach()
        ctx.rng = rng
        ctx.shape = shape
        ctx.mode = mode
        return result

    @staticmethod
    def backward(ctx, grad_output):
        rng, shape, mode = ctx.rng, ctx.shape, ctx.mode

        old_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        w = do_winit(shape[0], shape[1], mode, device="cuda")
        torch.set_rng_state(old_rng)

        return grad_output @ w.data.transpose(0, 1), None, None, None

###### Override the Roberta classes to use the custom autograd function ######
class MemOptimizedSLaShRobertaOutput(RobertaOutput):
    def __init__(self, config, z_vector=None, w_param=None):
        super().__init__(config)
        self.z_vector = z_vector
        self.rng = w_param[0]
        self.shape = w_param[1]
        self.mode = w_param[2]

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # insert just before the layer norm
        bias = Projection.apply(self.z_vector, self.rng, self.shape, self.mode).unsqueeze(dim=1)
        hidden_states = self.LayerNorm(hidden_states + input_tensor + bias)
        return hidden_states


class MemOptimizedSLaShRobertaLayer(RobertaLayer):
    def __init__(self, config, z_vector, w_param):
        super().__init__(config)
        self.output = MemOptimizedSLaShRobertaOutput(config, z_vector, w_param)


class MemOptimizedSLaShRobertaEncoder(RobertaEncoder):
    def __init__(self, config, z_vector, weight_params):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList(
            [MemOptimizedSLaShRobertaLayer(config, z_vector, w_param) for w_param in weight_params]
        )
        self.gradient_checkpointing = False


class MemOptimizedSLaShRobertaModel(RobertaModel):
    def __init__(self, config, add_pooling_layer=True, z_vector=None, weight_params=None):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = MemOptimizedSLaShRobertaEncoder(config, z_vector, weight_params)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.init_weights()


class MemOptimizedSLaShRobertaForSequenceClassification(FullRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        z_size = config.z_size
        if config.bias_position == "intermediate":
            raise Exception("Some error")

        d_size = config.hidden_size
        self.z_vector = nn.Parameter(do_zinit(z_size, mode=config.z_init))
        self.weight_params = []
        for _ in range(config.num_hidden_layers):
            rng = torch.get_rng_state()
            # do this to simulate random generation; but we don't maintain it.
            do_winit(z_size, d_size, mode=config.w_init)
            self.weight_params.append((rng, (z_size, d_size), config.w_init))

        self.roberta = MemOptimizedSLaShRobertaModel(
            config, add_pooling_layer=True, z_vector=self.z_vector, weight_params=self.weight_params
        )
        self.init_weights()

    def set_up_for_training(self, use_mask_embeddings, use_dropout):
        super().set_up_for_training(use_mask_embeddings, use_dropout)
        self.requires_grad_(False)
        self.classifier.requires_grad_(True)
        self.z_vector.requires_grad_(True)
