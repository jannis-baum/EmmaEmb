from ema.embedding.embedding_handler import EmbeddingHandler


# Script kindly provided by Evolutionary Scale
# https://github.com/evolutionaryscale/esm
class Esm3(EmbeddingHandler):
    def __init__(
        self,
        no_gpu: bool = False,
    ):
        super().__init__(no_gpu)
        self.model = None

    def get_embedding(self, protein_sequences, output_dir: str):
        pass
