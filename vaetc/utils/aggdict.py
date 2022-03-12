from typing import Optional, Any, Dict, List, TypeVar
import numpy as np

T = TypeVar("T")
def zip_dict(dicts: List[Dict[str, T]]) -> Dict[str, List[T]]:

    result: Dict[str, List[T]] = {}

    for d in dicts:
        for key in d:
            result.setdefault(key, [])
            result[key].append(d[key])
    
    return result

T = TypeVar("T")
def mean_dict(dicts: List[Dict[str, T]]) -> Dict[str, float]:

    list_dict = zip_dict(dicts)

    result: Dict[str, float] = {}
    for key in list_dict:
        result[key] = float(np.mean(list_dict[key]))
    
    return result