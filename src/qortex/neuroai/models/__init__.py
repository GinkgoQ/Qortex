from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.models._registry import make_model_adapter
from qortex.neuroai.models._contracts import (
    ModelContractEntry,
    lookup as lookup_model_contract,
    list_entries as list_model_contracts,
)

__all__ = [
    "ModelAdapter",
    "ModelOutput",
    "make_model_adapter",
    # Contract registry
    "ModelContractEntry",
    "lookup_model_contract",
    "list_model_contracts",
    # Concrete adapters available on demand:
    # HuggingFaceAdapter, ONNXModelAdapter, TorchModelAdapter,
    # MONAIBundleAdapter, BrainDecodeAdapter, UltralyticsAdapter, CustomPluginAdapter
]
