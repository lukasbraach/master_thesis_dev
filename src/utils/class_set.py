from typing import List, Set

import hydra


def new(classpaths: List[str]) -> Set[type]:
    result = []

    for path in classpaths:
        cls = hydra.utils.get_class(path)
        result.append(cls)

    return set(result)
