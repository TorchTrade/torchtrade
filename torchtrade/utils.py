import torch
import numpy as np
from tensordict import TensorDict
from datasets import Dataset


def td_to_dataset(td: TensorDict) -> Dataset:
    """
    Convert a TensorDict (possibly nested) into a HuggingFace Dataset.
    Nested keys are flattened using dot notation.
    """

    def _flatten(td, prefix=""):
        out = {}
        for key, value in td.items():
            name = f"{prefix}{key}"
            if isinstance(value, TensorDict):
                out.update(_flatten(value, prefix=name + "."))
            else:
                if not torch.is_tensor(value):
                    raise TypeError(f"Expected torch.Tensor, got {type(value)}")
                out[name] = value.detach().cpu().numpy()
        return out

    flat_dict = _flatten(td)

    # HuggingFace expects each value to have len == batch_size
    return Dataset.from_dict(flat_dict)


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