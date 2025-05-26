import numpy as np
import pandas as pd
from typing import Tuple, List
import torch
from torch.nn.functional import normalize

class QueryMatcher:
    def __init__(self, feature_file: str, filename_file: str, metric: str = "cosine"):

        """
        Initialize the matcher with precomputed map features and filenames.
        """

        self.features = np.load(feature_file)  # Shape: [N, 1280]
        self.filenames = pd.read_csv(filename_file)["filename"].tolist()
        self.metric = metric

        # Convert to torch tensor for efficient computation
        self.feature_tensor = torch.from_numpy(self.features).float()  # [N, 1280]

        if self.metric == "cosine":
            self.feature_tensor = normalize(self.feature_tensor, dim=1)

    def match(self, query_vector: torch.Tensor, top_k: int = 1) -> List[Tuple[str, float]]:

        """
        Match a query vector to the map features and return top-k results.

        Returns:
            List of (filename, score) tuples.
        """
        
        # Normalize query vector if using cosine similarity
        if self.metric == "cosine":
            query_vector = normalize(query_vector.unsqueeze(0), dim=1)  # [1, 1280]
            similarity = torch.matmul(self.feature_tensor, query_vector.T).squeeze()  # [N]
            top_scores, top_idxs = torch.topk(similarity, top_k, largest=True)
        elif self.metric == "l2":
            diff = self.feature_tensor - query_vector.unsqueeze(0)
            dists = torch.norm(diff, dim=1)
            top_scores, top_idxs = torch.topk(dists, top_k, largest=False)
        else:
            raise ValueError("Unsupported metric: choose 'cosine' or 'l2'")

        results = []
        for idx, score in zip(top_idxs.tolist(), top_scores.tolist()):
            filename = self.filenames[idx]
            results.append((filename, float(score)))

        return results
