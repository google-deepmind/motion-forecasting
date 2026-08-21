"""
Exponential Moving Average (EMA) for model parameters.

Maintains a shadow copy of model parameters that is updated each training step:
    ema_param = decay * ema_param + (1 - decay) * model_param

At evaluation/inference time, the EMA parameters are swapped into the model
for better generalization (smoother loss landscape).

Usage:
    ema = EMAModel(model, decay=0.9999)
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        ema.update()          # update shadow params after each optimizer step

    ema.swap_into_model()     # swap EMA weights into model for eval
    evaluate(model)
    ema.swap_into_model()     # swap back to training weights
"""

import copy
import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average of model parameters.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate. Higher = smoother, slower-updating average.
               Typical values: 0.999 to 0.9999.
        start_step: Don't update EMA until this many steps have passed.
                    Useful to let the model warm up before tracking.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, start_step: int = 0):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")

        self.decay = decay
        self.start_step = start_step
        self.step = 0

        # Shadow parameters: deep copy of trainable parameters only
        self.shadow_params = [
            p.clone().detach() for p in model.parameters() if p.requires_grad
        ]
        # Keep a reference list of which model params we track (by identity)
        self._model_params = [p for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def update(self):
        """Update shadow parameters with current model parameters.

        Call this once after each optimizer.step().
        Before start_step, this just copies the model params (decay=0 effectively).
        """
        self.step += 1

        if self.step <= self.start_step:
            # During warmup, just copy model params directly (no smoothing)
            for shadow, param in zip(self.shadow_params, self._model_params):
                shadow.copy_(param.data)
            return

        decay = self.decay
        for shadow, param in zip(self.shadow_params, self._model_params):
            # shadow = decay * shadow + (1 - decay) * param
            shadow.lerp_(param.data, 1.0 - decay)

    def swap_into_model(self):
        """Swap EMA shadow params into the model (and model params into shadow).

        Call this before evaluation, then call again after to restore training weights.
        This is its own inverse: calling it twice is a no-op.
        """
        for shadow, param in zip(self.shadow_params, self._model_params):
            tmp = param.data.clone()
            param.data.copy_(shadow)
            shadow.copy_(tmp)

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'shadow_params': [p.clone() for p in self.shadow_params],
            'decay': self.decay,
            'step': self.step,
            'start_step': self.start_step,
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.step = state_dict['step']
        self.start_step = state_dict['start_step']
        for shadow, saved in zip(self.shadow_params, state_dict['shadow_params']):
            shadow.copy_(saved)

    def __repr__(self):
        return (
            f"EMAModel(decay={self.decay}, step={self.step}, "
            f"start_step={self.start_step}, "
            f"num_params={len(self.shadow_params)})"
        )
