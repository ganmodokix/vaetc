from typing import Any, Callable, Type
from .abstract import RLModel

_MODEL_DICTIONARY = {}

def register_model(model_name: str, class_func: Callable[[dict],RLModel]):
    """ Register a model class for :func:`model_by_params`

    Args:
        model_name (str): A model name
        class_func (class): constructor function.
    """

    if model_name in _MODEL_DICTIONARY:
        raise ValueError(f'Model "{model_name}" has already been registered')

    _MODEL_DICTIONARY[model_name] = class_func

def model_by_params(model_name: str, hyperparameters: dict) -> RLModel:
    """ Create an RLModel instance by a registered model name
    
    Args:
        model_name (str): A model name
        hyperparameters (dict): Hyperparameters to give to the constructor

    Returns:
        RLModel: A constructed model instance
    """

    if model_name not in _MODEL_DICTIONARY:
        raise ValueError(f"Model '{model_name} has not been registered'")

    return _MODEL_DICTIONARY[model_name](hyperparameters)
