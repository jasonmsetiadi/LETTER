import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def array_to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """Converts numpy array to torch tensor safely across numpy 1.x and 2.x."""
    try:
        return torch.from_numpy(arr).to(dtype=dtype)
    except Exception:
        # Fallback for torch versions incompatible with NumPy 2.x C-API
        try:
            return torch.frombuffer(memoryview(arr), dtype=dtype).reshape(arr.shape)
        except Exception:
            return torch.tensor(arr.tolist(), dtype=dtype)


class EmbDataset(Dataset):
    """Loads item embeddings (.npy or .pt) for tokenizer training."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Embedding file not found: {data_path}")

        if data_path.endswith(".npy"):
            arr = np.load(data_path)
            self.tensor_embeddings = array_to_tensor(arr, dtype=torch.float32)
        elif data_path.endswith(".pt") or data_path.endswith(".pth"):
            loaded = torch.load(data_path, map_location="cpu")
            if isinstance(loaded, dict) and "embeddings" in loaded:
                self.tensor_embeddings = loaded["embeddings"].float()
            elif isinstance(loaded, torch.Tensor):
                self.tensor_embeddings = loaded.float()
            else:
                raise ValueError(f"Unrecognized tensor format in {data_path}")
        else:
            raise ValueError(f"Unsupported embedding format: {data_path}")

        if self.tensor_embeddings.ndim == 1:
            self.tensor_embeddings = self.tensor_embeddings.unsqueeze(0)
        self.dim = self.tensor_embeddings.shape[-1]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.tensor_embeddings[index], index

    def __len__(self) -> int:
        return len(self.tensor_embeddings)

