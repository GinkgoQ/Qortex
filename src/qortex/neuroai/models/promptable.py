# src/qortex/neuroai/models/promptable.py
"""Abstract base for promptable (interactive) segmentation model adapters.

Extends the existing ModelAdapter contract (models/_base.py) without
modifying it. A promptable adapter always implements predict_with_prompt();
predict() (the base ABC's required method) either falls back to automatic
mode when the model declares supports_automatic_mode=True (see
InteractionContract, zoo/schema.py), or raises ModelAdapterError directing
the caller to predict_with_prompt() -- reusing the existing exception type
already used by huggingface.py/torch.py/plugin.py for adapter-level
errors, rather than inventing a new one.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.zoo.schema import InteractionContract


class PromptableModelAdapter(ModelAdapter):
    @abstractmethod
    def interaction_contract(self) -> InteractionContract:
        """Return the model's real, confirmed prompt capabilities."""

    @abstractmethod
    def predict_with_prompt(self, batch: Any, prompt: Prompt) -> ModelOutput:
        """Run inference using the given prompt (points/boxes/text)."""

    def predict_automatic(self, batch: Any) -> ModelOutput:
        """Run inference without a prompt, for models that declare
        supports_automatic_mode=True. Not implemented by default -- only
        adapters whose model genuinely has an automatic/whole-image mode
        (e.g. VISTA3D) override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support automatic (promptless) inference."
        )

    def predict(self, batch: Any) -> ModelOutput:
        if self.interaction_contract().supports_automatic_mode:
            return self.predict_automatic(batch)
        raise ModelAdapterError(
            f"{type(self).__name__} requires a prompt for inference. "
            "Use predict_with_prompt(batch, prompt) instead of predict()."
        )


__all__ = ["PromptableModelAdapter"]
