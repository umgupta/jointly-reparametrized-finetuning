"""Some overrides for Opacus' DPOptimizer to make it work with transformers' Trainer and our use case."""


import logging
from typing import Callable, List, Optional, Union

import torch
from opacus import PrivacyEngine
from opacus.optimizers import AdaClipDPOptimizer, DPOptimizer, DPPerLayerOptimizer, \
    DistributedDPOptimizer, \
    DistributedPerLayerOptimizer, SimpleDistributedPerLayerOptimizer
from torch import optim
from torch.optim import Optimizer
from transformers import TrainerCallback

from src.models import PrivateEfficientRobertaForSequenceClassification, \
    PrivateJR_WARPRobertaForSequenceClassification, PrivateSLaShRobertaForSequenceClassification, \
    PrivateSharedWARPRobertaForSequenceClassification

logger = logging.getLogger(__name__)


class PrivacyEngineWrapper(PrivacyEngine):
    def _prepare_optimizer(
            self,
            optimizer: optim.Optimizer,
            *,
            noise_multiplier: float,
            max_grad_norm: Union[float, List[float]],
            expected_batch_size: int,
            loss_reduction: str = "mean",
            distributed: bool = False,
            clipping: str = "flat",
            noise_generator=None,
            grad_sample_mode="hooks",
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optim_class = get_optimizer_class(
            clipping=clipping,
            distributed=distributed,
            grad_sample_mode=grad_sample_mode,
        )

        return optim_class(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
        )


class EWDPOptimizer(DPOptimizer):
    """
        ``torch.optim.Optimizer`` wrapper that adds additional functionality to clip per
        sample gradients and add Gaussian noise.

        Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
        ``DPOptimzer`` assumes that parameters over which it performs optimization belong
        to GradSampleModule and therefore have the ``grad_sample`` attribute.

        On a high level ``DPOptimizer``'s step looks like this:
        1) Aggregate ``p.grad_sample`` over all parameters to calculate per sample norms
        2) Clip ``p.grad_sample`` so that per sample norm is not above threshold
        3) Aggregate clipped per sample gradients into ``p.grad``
        4) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
        max grad norm limit (``std = noise_multiplier * max_grad_norm``).
        5) Call underlying optimizer to perform optimization step

        Examples:
            >>> module = MyCustomModel()
            >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
            >>> dp_optimizer = DPOptimizer(
            ...     optimizer=optimizer,
            ...     noise_multiplier=1.0,
            ...     max_grad_norm=1.0,
            ...     expected_batch_size=4,
            ... )
        """

    def __init__(
            self,
            optimizer: Optimizer,
            *,
            noise_multiplier: float,
            max_grad_norm: float,
            expected_batch_size: Optional[int],
            loss_reduction: str = "mean",
            generator=None,
            secure_mode: bool = False,
    ):
        super(EWDPOptimizer, self).__init__(
            optimizer,
            noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode
        )

        # tensors are hashable, yay
        self.skip_updating_params = []
        self.model = None
        self.grad_sample_storage = {p: None for p in self.params}

    def _get_flat_grad_sample(self, p: torch.Tensor):
        """
        Return parameter's per sample gradients as a single tensor.

        By default, per sample gradients (``p.grad_sample``) are stored as one tensor per
        batch basis. Therefore, ``p.grad_sample`` is a single tensor if holds results from
        only one batch, and a list of tensors if gradients are accumulated over multiple
        steps. This is done to provide visibility into which sample belongs to which batch,
        and how many batches have been processed.

        This method returns per sample gradients as a single concatenated tensor, regardless
        of how many batches have been accumulated

        Args:
            p: Parameter tensor. Must have ``grad_sample`` attribute

        Returns:
            ``p.grad_sample`` if it's a tensor already, or a single tensor computed by
            concatenating every tensor in ``p.grad_sample`` if it's a list

        Raises:
            ValueError
                If ``p`` is missing ``grad_sample`` attribute
        """

        # if not hasattr(p, "grad_sample"):
        #     raise ValueError(
        #         "Per sample gradient not found. Are you using GradSampleModule?"
        #     )
        if self.grad_sample_storage[p] is None:
            raise ValueError(
                "Per sample gradient is not initialized. Not updated in backward pass?"
            )
        if isinstance(self.grad_sample_storage[p], torch.Tensor):
            ret = self.grad_sample_storage[p]
        elif isinstance(self.grad_sample_storage[p], list):
            ret = torch.cat(self.grad_sample_storage[p], dim=0)
        else:
            raise ValueError(f"Unexpected grad_sample type: {type(self.grad_sample_storage[p])}")

        return ret

    @property
    def grad_samples(self) -> List[torch.Tensor]:
        """
        Returns a flat list of per sample gradient tensors (one per parameter)
        """
        ret = []
        for p in self.params:
            ret.append(self._get_flat_grad_sample(p))
        return ret

    @property
    def accumulated_iterations(self) -> int:
        """
        Returns number of batches currently accumulated and not yet processed.

        In other words ``accumulated_iterations`` tracks the number of forward/backward
        passed done in between two optimizer steps. The value would typically be 1,
        but there are possible exceptions.

        Used by privacy accountants to calculate real sampling rate.
        """
        vals = []
        for p in self.params:
            if not self.grad_sample_storage[p]:
                raise ValueError(
                    "Per sample gradient not found. Are you using GradSampleModule?"
                )
            if isinstance(self.grad_sample_storage[p], torch.Tensor):
                vals.append(1)
            elif isinstance(self.grad_sample_storage[p], list):
                vals.append(len(self.grad_sample_storage[p]))
            else:
                raise ValueError(
                    f"Unexpected grad_sample type: {type(self.grad_sample_storage[p])}"
                )

        if len(set(vals)) > 1:
            raise ValueError(
                "Number of accumulated steps is inconsistent across parameters"
            )
        return vals[0]

    def scale_grad(self):
        """
        Applies given ``loss_reduction`` to ``p.grad``.

        Does nothing if ``loss_reduction="sum"``. Divides gradients by
        ``self.expected_batch_size`` if ``loss_reduction="mean"``
        """
        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad /= self.expected_batch_size * self.accumulated_iterations

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.

        Clears ``p.grad``, ``p.grad_sample`` and ``p.summed_grad`` for all of it's parameters

        Notes:
            ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` and
            ``p.summed_grad`` is never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list

        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """

        # print("optimizer's zero grad called")
        if set_to_none is False:
            logger.debug(
                "Despite set_to_none is set to False, "
                "opacus will set p.grad_sample and p.summed_grad to None due to "
                "non-trivial gradient accumulation behaviour"
            )

        for p in self.params:
            p.grad_sample = None

            if not self._is_last_step_skipped:
                p.summed_grad = None
            self.grad_sample_storage[p] = None
        self.original_optimizer.zero_grad(set_to_none)

    def pre_step(
            self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """

        # reattach to .grad_sample is not necessary but we reattach to make our life easier

        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:

        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            for p in self.params:
                for skip_p in self.skip_updating_params:
                    if p is skip_p:
                        p.grad = torch.zeros_like(p.grad)

            return self.original_optimizer.step()
        else:
            return None

    def _compute_z_vector_grad_sample(self):
        model = self.model._module
        if isinstance(model, PrivateEfficientRobertaForSequenceClassification):
            grad_sample_array = [
                layer.output.dense.bias.grad_sample @ layer.output.weight.data.transpose(0, 1) for
                layer
                in model.roberta.encoder.layer
            ]
            model.z_vector.grad_sample = sum(grad_sample_array)
            return True

        if isinstance(model, PrivateSharedWARPRobertaForSequenceClassification):
            grad_sample_array = [
                layer.id.bias.grad_sample @ layer.weight.data.transpose(0, 1) for layer
                in model.roberta.encoder.layer
            ]
            model.z_vector.grad_sample = sum(grad_sample_array)
            return True

        return False

    def set_metadata(self, model, skip_update):
        self.model = model

        self.skip_updating_params = skip_update

    """Below we compute per-sample gradient for z_vector manually, and clear the intermediate module's gradient"""

    def virtual_step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        # finetuning specific, but should let things pass if not using that model
        computed_z_vector_grad_sample = self._compute_z_vector_grad_sample()
        if computed_z_vector_grad_sample:
            # set bias's grad samples to zero, so they are equivalent to no grad
            # This should already be taken care by metadata set operation in trainer; but ensuring
            # here too
            model = self.model._module
            if isinstance(model, PrivateSLaShRobertaForSequenceClassification):
                for layer in model.roberta.encoder.layer:
                    layer.output.dense.bias.grad_sample = torch.zeros_like(
                        layer.output.dense.bias.grad_sample
                    )
            if isinstance(model, PrivateJR_WARPRobertaForSequenceClassification):
                for layer in model.roberta.encoder.layer:
                    layer.id.bias.grad_sample = torch.zeros_like(layer.id.bias.grad_sample)

        # move .grad_sample to self.grad_sample_storage and empty grad_sample for next iteration
        for p in self.params:
            if self.grad_sample_storage[p] is None:
                self.grad_sample_storage[p] = torch.clone(p.grad_sample)
                p.grad_sample = None

            elif isinstance(self.grad_sample_storage[p], torch.Tensor):
                self.grad_sample_storage[p] = [self.grad_sample_storage[p],
                                               torch.clone(p.grad_sample)]
                p.grad_sample = None

            elif isinstance(self.grad_sample_storage[p], list):
                self.grad_sample_storage[p].append(torch.clone(p.grad_sample))
                p.grad_sample = None


def get_optimizer_class(clipping: str, distributed: bool, grad_sample_mode: str = None):
    if clipping == "flat" and distributed is False:
        return EWDPOptimizer
    elif clipping == "flat" and distributed is True:
        return DistributedDPOptimizer
    elif clipping == "per_layer" and distributed is False:
        return DPPerLayerOptimizer
    elif clipping == "per_layer" and distributed is True:
        if grad_sample_mode == "hooks":
            return DistributedPerLayerOptimizer
        elif grad_sample_mode == "ew":
            return SimpleDistributedPerLayerOptimizer
        else:
            raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    elif clipping == "adaptive" and distributed is False:
        return AdaClipDPOptimizer
    raise ValueError(
        f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    )


class OptimizerZeroGradCallback(TrainerCallback):
    """
    This class registers all the necessary callbacks to make transformers.Trainer compatible with opacus.
    """

    def on_step_end(self, args, state, control, **kwargs):
        kwargs.get("optimizer").zero_grad()
