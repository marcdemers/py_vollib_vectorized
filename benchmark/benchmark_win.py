# -*- coding: utf-8 -*-
"""benchmark.py 的 Windows 适配版 — 去掉 signal.SIGALRM，缩小规模"""
import json, time, warnings, os
import pandas as pd

# 切换到包根目录，确保 tests/ 相对路径可用
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter('ignore')

with open('tests/test_data_py_vollib.json', 'rb') as f:
    d = json.load(f)
    test_df = pd.DataFrame(d['data'], index=d['index'], columns=d['columns'])
    test_df_calls = test_df.copy()
    test_df_calls['flag'] = 'c'
    test_df_calls['q'] = 0
    test_df_puts = test_df.copy()
    test_df_puts['flag'] = 'p'
    test_df_puts['q'] = 0

test_df = pd.concat((test_df_calls.iloc[:100], test_df_puts.iloc[:100]), axis=0)
big_test_df = pd.concat([test_df for _ in range(500)]).sample(frac=1., random_state=0)

def timeit(fn, N):
    t0 = time.time()
    fn(N)
    return time.time() - t0

def for_loop(N):
    from py_vollib.black_scholes.implied_volatility import implied_volatility
    from py_vollib.black_scholes import black_scholes
    tt = big_test_df.iloc[:N]
    prices = []
    for i in range(len(tt)):
        prices.append(black_scholes(tt.iloc[i]['flag'], tt.iloc[i]['S'], tt.iloc[i]['K'], tt.iloc[i]['t'], tt.iloc[i]['R'], tt.iloc[i]['v']))
    tt['price'] = prices
    ivs = []
    for i in range(len(tt)):
        ivs.append(implied_volatility(tt.iloc[i]['price'], tt.iloc[i]['S'], tt.iloc[i]['K'], tt.iloc[i]['t'], tt.iloc[i]['R'], tt.iloc[i]['flag']))

def vectorized(N):
    import py_vollib_vectorized
    tt = big_test_df.iloc[:N]
    prices = py_vollib_vectorized.vectorized_black_scholes(tt['flag'], tt['S'], tt['K'], tt['t'], tt['R'], tt['v'])
    tt['price'] = prices
    py_vollib_vectorized.vectorized_implied_volatility(tt['price'], tt['S'], tt['K'], tt['t'], tt['R'], tt['flag'])

# 预热 Numba JIT
print('预热 Numba JIT...')
vectorized(10)

n_contracts = [100, 1000, 10000, 100000]
print(f'\n{"N":>10s}  {"for_loop":>12s}  {"vectorized":>12s}  {"speedup":>8s}')
print('-' * 50)
for N in n_contracts:
    t_for = timeit(for_loop, N)
    t_vec = timeit(vectorized, N)
    speedup = t_for / t_vec if t_vec > 0 else float('inf')
    print(f'{N:10d}  {t_for:10.4f}s  {t_vec:10.4f}s  {speedup:7.1f}x')
