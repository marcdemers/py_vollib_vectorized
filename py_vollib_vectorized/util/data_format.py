import warnings

import numpy as np
import pandas as pd
from py_vollib.helpers.constants import FLOAT_MAX, MINUS_FLOAT_MAX
from py_vollib.helpers.exceptions import PriceIsAboveMaximum, PriceIsBelowIntrinsic

binary_flag = {'c': 1, 'p': -1, 1: 1, -1: -1, '1': 1, '-1': -1}

def _preprocess_flags(flags, dtype):
    # 向量化 flag 映射, 替代逐元素 dict 查询的 Python 循环(10 万合约约省 15-20ms)
    arr = np.asarray(flags)
    if arr.dtype.kind in 'USO':  # 字符串/object 输入(pd.Series 的 str 列为 object): 'c'/'p'/'1'/'-1'
        out = np.where(arr == 'c', 1.0,
                       np.where(arr == 'p', -1.0,
                                np.where(arr == '1', 1.0,
                                         np.where(arr == '-1', -1.0, np.nan))))
        # object 数组可能混入数值 1/-1(与原 binary_flag dict 行为一致), 对未匹配项补数值映射
        nan_mask = np.isnan(out)
        if nan_mask.any() and arr.dtype.kind == 'O':
            out = np.where(nan_mask & (arr == 1), 1.0, np.where(nan_mask & (arr == -1), -1.0, out))
    else:  # 数值输入: 1/-1
        out = np.where(arr == 1, 1.0, np.where(arr == -1, -1.0, np.nan))
    if np.isnan(out).any():
        # 显性报错: 列出全部非法 flag 值
        bad = sorted({str(x) for x in np.asarray(flags)[np.isnan(out)].tolist()})
        raise ValueError(f"Unsupported flag value(s) {bad}, must be 'c'/'p' (or 1/-1).")
    return out.astype(dtype, copy=False)


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
        # dtype 已匹配时 astype(copy=False) 返回原数组, 免一次全量复制
        return data.astype(dtype=dtype, copy=False)
    raise ValueError(f"Data type {type(data)} unsupported, must be in: list, tuple, np.array, pd.Series.")


def maybe_format_data_and_broadcast(*all_data, dtype):
    data_numpy = tuple(np.ravel(_maybe_format_data(d, dtype=dtype)) for d in all_data)
    # 等长快速路径: 无需广播时跳过 broadcast_arrays + np.array 的二次复制
    n0 = len(data_numpy[0])
    if all(len(a) == n0 for a in data_numpy):
        return list(data_numpy)
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

    # 始终计算违规索引, 供调用方将无效合约置 NaN; on_error 仅控制告警/抛错,
    # 保证 warn/ignore/raise 三种模式对无效合约的输出一致(均为 NaN)
    _below_intrinsic_array = np.argwhere(prices < _intrinsic).ravel().tolist()
    _above_max_price = np.argwhere(prices >= _max_price).ravel().tolist()

    if on_error == "warn":
        if _below_intrinsic_array:
            warnings.warn(
                "Found Below Intrinsic contracts at index {}".format(_below_intrinsic_array),
                stacklevel=2)
        if _above_max_price:
            warnings.warn(
                "Found Above Maximum Price contracts at index {}".format(_above_max_price),
                stacklevel=2)
    elif on_error == "raise":
        if _below_intrinsic_array:
            raise PriceIsBelowIntrinsic
        if _above_max_price:
            raise PriceIsAboveMaximum

    return _below_intrinsic_array, _above_max_price


def _check_minus_above_float(values, on_error):
    # 始终计算越界索引, 供调用方将无效合约置 NaN; on_error 仅控制告警/抛错
    _above_float_max_array = np.argwhere(values == FLOAT_MAX).ravel().tolist()
    _below_float_min_array = np.argwhere(values == MINUS_FLOAT_MAX).ravel().tolist()
    if on_error == "warn":
        if _above_float_max_array:
            warnings.warn(
                "Found PriceAboveMaximum contracts at index {}".format(_above_float_max_array)
            )
        if _below_float_min_array:
            warnings.warn(
                "Found PriceBelowMaximum contracts at index {}".format(_below_float_min_array)
            )
    elif on_error == "raise":
        if _above_float_max_array:
            raise PriceIsAboveMaximum()
        if _below_float_min_array:
            raise PriceIsBelowIntrinsic()
    return _below_float_min_array, _above_float_max_array

def _validate_df_col(col, df):
    if not isinstance(col, str):
        raise ValueError(f"Column {col} must be a string!")
    elif col not in df:
        raise ValueError(f"Column {col} not found in dataframe!")
