import torch
import numpy as np
from tensordict import TensorDict
from datasets import Dataset


def dataset_to_td(ds: Dataset, device="cpu") -> TensorDict:
    """
    Convert a HuggingFace Dataset back into a TensorDict.
    Fully robust across datasets versions and nested array types.
    """

    np_dict = ds[:]   # dict[str, list | np.ndarray]

    td_data = {}

    for key, value in np_dict.items():
        # Ensure numpy array (handles lists-of-lists)
        np_array = np.asarray(value)
        tensor = torch.from_numpy(np_array).to(device)

        if "." in key:
            td_data[tuple(key.split("."))] = tensor
        else:
            td_data[key] = tensor

    return TensorDict(
        td_data,
        batch_size=[len(ds)],
        device=device,
    )