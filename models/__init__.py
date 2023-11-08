# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def make(name, config, **kwargs):
    model = models[name](config, **kwargs)
    return model


from . import (
    background
)
