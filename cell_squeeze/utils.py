# %%
import numpy as np
from rich import print as rprint


def flatten_array_dict(array_dict: dict[str, dict | np.ndarray], delimiter: str = "$"):
    flat_dict = {}
    for key, value in array_dict.items():
        if isinstance(value, dict):
            flatt_value_dict = flatten_array_dict(value, delimiter)
            for subkey, subvalue in flatt_value_dict.items():
                flat_key = f"{key}{delimiter}{subkey}"
                flat_dict[flat_key] = subvalue
        else:
            flat_dict[key] = value
    return flat_dict


def unflatten_array_dict(flat_dict: dict[str, np.ndarray], delimiter="$"):
    unflat_dict = {}

    flat_keys = set()

    groups: dict[str, list[str]] = {}
    for key in flat_dict.keys():
        if delimiter not in key:
            flat_keys.add(key)
            continue
        parent_key, subkey = key.split(delimiter, 1)
        groups.setdefault(parent_key, []).append(subkey)

    for key in flat_keys:
        unflat_dict[key] = flat_dict[key]

    for parent_key, subkeys in groups.items():
        unflat_dict[parent_key] = {}
        sub_dict = {
            subkey: flat_dict[f"{parent_key}{delimiter}{subkey}"] for subkey in subkeys
        }
        unflat_sub_dict = unflatten_array_dict(sub_dict, delimiter)
        unflat_dict[parent_key].update(unflat_sub_dict)

    return unflat_dict


d_easy = {
    "goop": {
        "data": np.array([1, 2, 3]),
        "shape": (3, 1),
        "cluster": {
            "data": np.array([4, 5, 6]),
            "shape": (3, 1),
            "hola": {
                "d_": np.array([[4, 5, 6], [6, 5, 1]]),
                "s_": (2, 3),
            },
        },
    },
    "arr": np.array([1, 2, 0]),
}

d_hard = {
    "goop": {
        "data": np.array([1, 2, 3]),
        "shape": (3, 1),
        "clusters": [
            {
                "data": np.array([4, 5, 6]),
                "shape": (3, 1),
            },
            {
                "data": np.array([6, 5, 1]),
                "shape": (3, 1),
            },
        ],
    },
    "arr": np.array([1, 2, 0]),
}

rprint(d_easy)
fd = flatten_array_dict(d_easy)
rprint(fd)
d_easy_2 = unflatten_array_dict(fd)
rprint(d_easy_2)
# %%
