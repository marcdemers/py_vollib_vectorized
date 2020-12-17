import warnings

import numpy as np
import pandas as pd
from py_vollib.helpers.constants import FLOAT_MAX, MINUS_FLOAT_MAX
from py_vollib.helpers.exceptions import PriceIsAboveMaximum, PriceIsBelowIntrinsic

binary_flag = {'c': 1, 'p': -1, 1: 1, -1: -1, '1': 1, '-1': -1}

def _preprocess_flags(flags, dtype):
    return np.array([binary_flag[f] for f in flags], dtype=dtype)


def _maybe_format_data(data, dtype):
    if isinstance(data, (int, float)):
        return np.array([data], dtype=dtype)
    elif isinstance(data, pd.Series):
        return data.values.astype(dtype)
    elif isinstance(data, pd.DataFrame):
        assert data.shape[1] == 1, "You passed a `pandas.DataFrame` object that contains more than (1) column!"
        return data.values.astype(dtype)
    elif isinstance(data, (list, tuple)):
        return np.array(data, dtype=dtype)
    elif isinstance(data, np.ndarray):
        return data.astype(dtype=dtype)
    raise ValueError(f"Data type {type(data)} unsupported, must be in: list, tuple, np.array, pd.Series.")


def maybe_format_data_and_broadcast(*all_data, dtype):
    data_numpy = tuple(np.ravel(_maybe_format_data(d, dtype=dtype)) for d in all_data)
    return [np.array(a, dtype=dtype) for a in np.broadcast_arrays(*data_numpy)]


def _validate_data(*all_data):
    number_elems = [len(x) for x in all_data]
    check = np.all([x == number_elems[0] for x in number_elems])
    if not check:
        raise ValueError(
            f"All input values must contain the same number of elements. Found number of elements = {number_elems}")


def _check_below_and_above_intrinsic(Ks, Fs, qs, prices, on_error):
    _intrinsic = np.zeros(shape=Ks.shape)
    _intrinsic[qs < 0] = (Ks - Fs)[qs < 0]
    _intrinsic[qs > 0] = (Fs - Ks)[qs > 0]
    _intrinsic = np.abs(np.maximum(_intrinsic, 0.))

    _max_price = np.zeros(shape=Ks.shape)
    _max_price[qs < 0] = Ks[qs < 0]
    _max_price[qs > 0] = Fs[qs > 0]

    _below_intrinsic_array, _above_max_price = [], []

    if on_error != "ignore":
        if np.any(prices < _intrinsic):
            _below_intrinsic_array = np.argwhere(prices < _intrinsic).ravel().tolist()
            if on_error == "warn":
                warnings.warn(
                    "Found Below Intrinsic contracts at index {}".format(_below_intrinsic_array),
                    stacklevel=2)
            elif on_error == "raise":
                raise PriceIsBelowIntrinsic
        if np.any(prices >= _max_price):
            _above_max_price = np.argwhere(prices >= _max_price).ravel().tolist()
            if on_error == "warn":
                warnings.warn(
                    "Found Above Maximum Price contracts at index {}".format(_above_max_price),
                    stacklevel=2)
            elif on_error == "raise":
                raise PriceIsAboveMaximum

    return _below_intrinsic_array, _above_max_price


def _check_minus_above_float(values, on_error):
    _below_float_min_array, _above_float_max_array = [], []
    if on_error != "ignore":
        if np.any(values == FLOAT_MAX):
            _above_float_max_array = np.argwhere(values == FLOAT_MAX).ravel().tolist()
            if on_error == "warn":
                warnings.warn(
                    "Found PriceAboveMaximum contracts at index {}".format(_above_float_max_array)
                )
            elif on_error == "raise":
                raise PriceIsAboveMaximum()
        elif np.any(values == MINUS_FLOAT_MAX):
            _below_float_min_array = np.argwhere(values == MINUS_FLOAT_MAX).ravel().tolist()
            if on_error == "warn":
                warnings.warn(
                    "Found PriceBelowMaximum contracts at index {}".format(_below_float_min_array)
                )
            elif on_error == "raise":
                PriceIsBelowIntrinsic()
    return _below_float_min_array, _above_float_max_array

def _validate_df_col(col, df):
    if not isinstance(col, str):
        raise ValueError(f"Column {col} must be a string!")
    elif col not in df:
        raise ValueError(f"Column {col} not found in dataframe!")
