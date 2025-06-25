from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseWorkflow(ABC):
    """Base class for all SynthDoc workflows."""

    @abstractmethod
    def process(self, config) -> Dict[str, Any]:
        """Process the workflow with given configuration and return HuggingFace dataset format."""
        pass

    def _create_hf_dataset_format(
        self, samples: list, metadata: dict = None
    ) -> Dict[str, Any]:
        """Helper method to create HuggingFace dataset format."""
        return {"train": samples, "metadata": metadata or {}}
