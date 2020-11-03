from numba import jit


use_jit = False
force_nopython = True

def maybe_jit(*jit_args, **jit_kwargs):
    if force_nopython:
        jit_kwargs.update({"nopython": True})
    def wrapper(fun):
        if use_jit:
            return jit(*jit_args, **jit_kwargs)(fun)
        else:
            return fun

    return wrapper