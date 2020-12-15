from numba import jit

use_jit = True
force_nopython = True
use_cache = True


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
