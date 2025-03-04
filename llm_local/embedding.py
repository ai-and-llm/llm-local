from sentence_transformers import SentenceTransformer
from torch import Tensor


class EmbeddingModel(object):
    def __init__(self):
        """ "
        Embedding model.
        https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        """
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def generate_embedding(self, data: str) -> Tensor:
        """
        Generates vector embedding from 'data'
        """
        embedding = self.model.encode(data)

        return embedding
