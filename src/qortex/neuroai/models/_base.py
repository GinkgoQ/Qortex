"""ModelAdapter abstract base class.

Every model adapter must:
  - inspect()          → ModelProfile         (no weights loaded)
  - required_input()   → InputContract
  - output_schema()    → OutputContract
  - load(runtime)      → None                 (load weights to device)
  - predict(batch)     → ModelOutput          (run inference)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qortex.neuroai.contracts import InputContract, ModelProfile, OutputContract
from qortex.neuroai.spec import RuntimeSpec


@dataclass
class ModelOutput:
    """Standardised model output."""

    output_type: str                            # "classification" | "segmentation" | ...
    raw: Any                                    # the raw model output tensor or array
    class_name: str | None = None               # predicted class name (classification)
    class_index: int | None = None              # predicted class index
    probabilities: dict[str, float] = field(default_factory=dict)
    bbox: list[float] | None = None             # detection: [x1,y1,x2,y2]
    mask: Any | None = None                     # segmentation mask
    embedding: Any | None = None                # embedding vector
    regression_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Abstract base for all NeuroAI model adapters."""

    _loaded: bool = False

    @abstractmethod
    def inspect(self) -> ModelProfile:
        """Inspect the model without loading weights.

        Returns a ``ModelProfile`` including input/output contracts.
        Must be fast — no weight download if possible.
        """

    @abstractmethod
    def required_input(self) -> InputContract:
        """Return the formal input contract this model expects."""

    @abstractmethod
    def output_schema(self) -> OutputContract:
        """Return the formal output schema this model produces."""

    @abstractmethod
    def load(self, runtime: RuntimeSpec) -> None:
        """Load the model weights into the target device.

        Must be called before ``predict()``.
        """

    @abstractmethod
    def predict(self, batch: Any) -> ModelOutput:
        """Run inference on the given batch.

        Parameters
        ----------
        batch:
            A QortexData object (QortexTimeSeries, QortexVolume, etc.) or a
            pre-processed tensor.

        Returns
        -------
        ModelOutput
            Structured output including class probabilities, masks, etc.
        """

    def predict_batch(self, items: list[Any]) -> list[ModelOutput]:
        """Predict on multiple items sequentially.

        Override for batched inference.
        """
        return [self.predict(item) for item in items]

    def unload(self) -> None:
        """Release GPU/CPU memory."""
        self._loaded = False
