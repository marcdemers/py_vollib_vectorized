from numba import jit

from .config import use_cache, use_jit, force_nopython

def maybe_jit(*jit_args, **jit_kwargs):
    # global use_jit, force_nopython, use_cache
    if force_nopython:
        jit_kwargs.update({"nopython": True})
    if use_cache:
        jit_kwargs.update({"cache": True})

    def wrapper(fun):
        if use_jit:
            return jit(*jit_args, **jit_kwargs)(fun)
        else:
            return fun

    return wrapper
