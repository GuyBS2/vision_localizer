import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize
from typing import Tuple, List


class QueryMatcher:
    """
    Matches a query feature vector to a database of map features.

    Supports cosine similarity or L2 distance as the matching metric.
    """

    def __init__(self, feature_file: str, filename_file: str, metric: str = "cosine"):
        """
        Initialize the matcher with precomputed map features and filenames.

        Args:
            feature_file: Path to the .npy file containing map feature vectors (shape: [N, D]).
            filename_file: Path to the .csv file with 'filename' column listing image filenames.
            metric: Similarity metric to use: 'cosine' or 'l2'.
        """
        self.features = np.load(feature_file)  # [N, D]
        self.filenames = pd.read_csv(filename_file)["filename"].tolist()
        self.metric = metric

        self.feature_tensor = torch.from_numpy(self.features).float()  # [N, D]

        if self.metric == "cosine":
            self.feature_tensor = normalize(self.feature_tensor, dim=1)

    def match(self, query_vector: torch.Tensor, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Match a query vector to the map features and return top-k results.

        Args:
            query_vector: A torch.Tensor of shape [D], typically D=1280.
            top_k: Number of top results to return.

        Returns:
            A list of tuples: (filename, score), sorted by best match first.
        """
        if self.metric == "cosine":
            query_vector = normalize(query_vector.unsqueeze(0), dim=1)  # [1, D]
            similarity = torch.matmul(self.feature_tensor, query_vector.T).squeeze()  # [N]
            top_scores, top_idxs = torch.topk(similarity, top_k, largest=True)

        elif self.metric == "l2":
            diff = self.feature_tensor - query_vector.unsqueeze(0)  # [N, D]
            dists = torch.norm(diff, dim=1)  # [N]
            top_scores, top_idxs = torch.topk(dists, top_k, largest=False)

        else:
            raise ValueError("Unsupported metric: choose 'cosine' or 'l2'")

        results = [
            (self.filenames[idx], float(score))
            for idx, score in zip(top_idxs.tolist(), top_scores.tolist())
        ]

        return results
