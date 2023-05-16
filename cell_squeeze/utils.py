# %%
import re
import numpy as np
from rich import print as rprint

seq_sentinel = "#"


def flatten_array_dict(array_dict: dict[str, dict | np.ndarray], delimiter: str = "$"):
    """Assumptions:
    1. root level is always a dict, i.e. lists are always nested in dicts
    2. keys are always strings
    3. lists are always lists of dicts
    """
    flat_dict = {}
    for key, value in array_dict.items():
        if delimiter in key:
            raise ValueError(f"Delimiter {delimiter} is not allowed in keys")

        if isinstance(value, dict):
            flat_value_dict = flatten_array_dict(value, delimiter)
            for subkey, subvalue in flat_value_dict.items():
                flat_key = f"{key}{delimiter}{subkey}"
                flat_dict[flat_key] = subvalue

        elif isinstance(value, list):
            for i, subvalue in enumerate(value):
                flat_subvalue_dict = flatten_array_dict(subvalue, delimiter)
                flat_key_prefix = f"{key}{seq_sentinel}{i}"
                for subkey, flat_subvalue in flat_subvalue_dict.items():
                    flat_key = f"{flat_key_prefix}{delimiter}{subkey}"
                    flat_dict[flat_key] = flat_subvalue
        else:
            flat_dict[key] = value
    return flat_dict


def unflatten_array_dict(flat_dict: dict[str, np.ndarray], delimiter="$"):
    regex = re.compile(f"(\\{delimiter}|{seq_sentinel})")

    unflat_dict = {}

    flat_keys = set()

    list_groups: dict[str, list[str]] = {}
    dict_groups: dict[str, list[str]] = {}

    for key in flat_dict.keys():
        # Find the first index of # or $, whichever comes first
        match = regex.search(key)
        if match is None:
            flat_keys.add(key)
            continue
        if match.group() == seq_sentinel:
            parent_key, subkey = key.split(seq_sentinel, 1)
            list_groups.setdefault(parent_key, []).append(subkey)
        else:
            parent_key, subkey = key.split(delimiter, 1)
            dict_groups.setdefault(parent_key, []).append(subkey)

    assert set(list_groups.keys()).isdisjoint(
        set(dict_groups.keys())
    ), "List and dict groups must be disjoint"

    for key in flat_keys:
        unflat_dict[key] = flat_dict[key]

    for parent_key, subkeys in dict_groups.items():
        unflat_dict[parent_key] = {}
        sub_dict = {
            subkey: flat_dict[f"{parent_key}{delimiter}{subkey}"] for subkey in subkeys
        }
        unflat_sub_dict = unflatten_array_dict(sub_dict, delimiter)
        unflat_dict[parent_key].update(unflat_sub_dict)

    for parent_key, subkeys in list_groups.items():
        unflat_dict[parent_key] = []
        subgroups = {}
        for key in subkeys:
            seq_key, subsubkey = key.split(delimiter, 1)
            subgroups.setdefault(int(seq_key), []).append(subsubkey)
        for i, subdictkeys in subgroups.items():
            sub_dict = {
                subdictkey: flat_dict[
                    f"{parent_key}{seq_sentinel}{i}{delimiter}{subdictkey}"
                ]
                for subdictkey in subdictkeys
            }
            unflat_sub_dict = unflatten_array_dict(sub_dict, delimiter)
            unflat_dict[parent_key].append(unflat_sub_dict)

    return unflat_dict


def test_case(d):
    rprint(d)
    fd = flatten_array_dict(d)
    rprint(fd)
    d2 = unflatten_array_dict(fd)
    rprint(d2)


# %%
if __name__ == "__main__":
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

    d_hard1 = {"clusters": [{"v1": np.array([1, 2, 3]), "v2": np.array([4, 5, 6, 7])}]}
    d_hard2 = {
        "clusters": [{"v1": np.array([1, 2, 3])}, {"v2": np.array([4, 5, 6, 7])}]
    }

    d_harder = {
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

    test_case(d_easy)
    test_case(d_hard1)
    test_case(d_hard2)
    test_case(d_harder)
