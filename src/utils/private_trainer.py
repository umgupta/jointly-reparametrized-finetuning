"""
Overrides the hugging face trainer.

This trainer makes opacus and differential-privacy specific changes to the
 trainer.
    - Taking the virtual step
    - Setting the DP-optimizer

Note: We use torch's extenpanded weight (EW) feature to compute per-sample
 gradients. However, these feature doesnot compute per sample gradient 
 correctly for nn.Parameter declared in SLaSh or JR_WARP modules. For this 
 reason, we have to compute per sample gradients manually. We do this by 
 computing gradient of some particular layer/parameter (denote as P) for which 
 EW works. Then, we use the gradient formula derived by hand.

We ask optimizer to skip updating P. As a further safeguard, we clear the 
gradient of P right after manual gradient computation in `opacus_utils.py` to 
avoid updates (see `virtual_step` function in `opacus_utils.py`).
"""


import logging
from typing import Any, Dict, Union

import torch
from torch import nn
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import find_labels, is_sagemaker_mp_enabled

from src.models import (
    PrivateJR_WARPRobertaForSequenceClassification,
    PrivateSLaShRobertaForSequenceClassification,
)

logger = logging.getLogger(__name__)


class PrivateTrainer(Trainer):
    def __init__(self, **kwargs):
        self.privacy_engine = kwargs.pop("privacy_engine")
        self.private_training_args = kwargs.pop("private_training_args")
        self.expected_batch_size = kwargs.pop("expected_batch_size")
        super().__init__(**kwargs)
        default_label_names = find_labels(self.model._module.__class__)
        self.label_names = (
            default_label_names if self.args.label_names is None else self.args.label_names
        )

    def _set_signature_columns_if_needed(self):
        super(PrivateTrainer, self)._set_signature_columns_if_needed()
        self._signature_columns += [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
            "head_mask",
            "inputs_embeds",
            "labels",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
        ]
        self._signature_columns = list(set(self._signature_columns))

    def create_optimizer(self):
        """
        Set up the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            # breakpoint()
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if p.requires_grad is True
                    ],
                    "weight_decay": self.args.weight_decay,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            #     self.optimizer = OSS(
            #         params=optimizer_grouped_parameters,
            #         optim=optimizer_cls,
            #         **optimizer_kwargs,
            #     )
            # else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")

            # if is_sagemaker_mp_enabled():
            #     self.optimizer = smp.DistributedOptimizer(self.optimizer)

        self.optimizer = self.privacy_engine._prepare_optimizer(
            self.optimizer,
            noise_multiplier=self.private_training_args.noise,
            max_grad_norm=self.private_training_args.per_sample_grad_norm,
            expected_batch_size=self.expected_batch_size,
        )

        # Skip the update for specific model due to EW specific issues
        if isinstance(opt_model._module, PrivateSLaShRobertaForSequenceClassification):
            self.optimizer.set_metadata(
                model=opt_model,
                skip_update=[
                    layer.output.dense.bias for layer in opt_model._module.roberta.encoder.layer
                ],
            )
        if isinstance(opt_model._module, PrivateJR_WARPRobertaForSequenceClassification):
            self.optimizer.set_metadata(
                model=opt_model,
                skip_update=[layer.id.bias for layer in opt_model._module.roberta.encoder.layer],
            )
        return self.optimizer

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     raise NotImplementedError("DP currently doesn't support this")
        #
        # if self.use_amp:
        #     raise NotImplementedError("DP currently doesn't support this.")
        # else:
        # breakpoint()
        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # Compared to the original HF implementation, we have to remove the loss scaling by the number of gradient
        # accumulation steps since opacus scales the gradients accordingly. However, we still need to scale the loss
        # that is returned in order for the logging to work correctly. Hence we scale the loss after the call to
        # loss.backward()

        # if self.use_amp:
        #     raise NotImplementedError("DP currently doesn't support this")
        # elif self.use_apex:
        #     raise NotImplementedError("DP currently doesn't support this")
        # elif self.deepspeed:
        #     raise NotImplementedError("DP currently doesn't support this")
        # else:

        loss.backward()

        # # do a virtual step to clean up the .grad_sample
        self.optimizer.virtual_step()
        # breakpoint()

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # breakpoint()
        outputs = model(inputs.get("input_ids"), **inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
