import torch
from abc import ABC, abstractmethod
import logging


# Define the abstract base class for embedding handlers
class EmbeddingHandler(ABC):
    def __init__(self, no_gpu: bool = False):
        """
        Base class for embedding handlers.

        Args:
            no_gpu (bool): Flag to disable GPU usage.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_gpu else "cpu"
        )
        # self.logger = self.setup_logger()

    @abstractmethod
    def get_embedding(
        self, protein_sequences: dict, model_id, output_dir: str, layer: int
    ):
        pass

    def check_device(self):
        """
        Logs the device being used for computations.
        """
        if self.device.type == "cuda":
            self.logger.info(f"Using GPU for {self.model_id} embeddings.")
        else:
            self.logger.info(f"Using CPU for {self.model_id} embeddings.")
