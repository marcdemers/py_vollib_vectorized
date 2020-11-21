from py_vollib_vectorized.entrypoints import *

from py_vollib_vectorized import implied_volatility_vectorized

import pandas as pd

data = pd.read_csv("fake_data.csv").iloc[:100]

data_repeated = pd.concat([data for _ in range(int(1e6 // 100))])
data_repeated = data_repeated.sample(frac=1)  # shuffle

# np.random.seed(0)
# data_repeated["Px"] += np.random.normal(0, 5, len(data_repeated))
# data_repeated["MidPx"] += np.random.normal(0, 5, len(data_repeated))
# data_repeated["Strike"] += np.random.normal(0, 10, len(data_repeated))
#
# data_repeated["Interest Free Rate"] += np.random.normal(0, 0.01, len(data_repeated))
# data_repeated["Annualized Time To Expiration"] += np.random.normal(0, 0.01, len(data_repeated))

data_repeated[["Px", "MidPx", "Strike", "Interest Free Rate", "Annualized Time To Expiration"]] = \
    data_repeated[["Px", "MidPx", "Strike", "Interest Free Rate", "Annualized Time To Expiration"]].abs()

from py_vollib_vectorized.jit_helper import use_jit, use_cache, force_nopython

import time

times = []
times_greeks = []
for run in range(10):
    start_time = time.time()
    ivs = implied_volatility_vectorized(
        price=data_repeated["MidPx"].values,  # current option price
        S=data_repeated["Px"].values,  # underlying asset price
        K=data_repeated["Strike"].values,  # strike
        t=data_repeated["Annualized Time To Expiration"].values,  # normalized days to expiration
        r=data_repeated["Interest Free Rate"].values,  # interest free rate
        flag=data_repeated["Flag"].values,  # call or put
    )
    times.append(time.time() - start_time)


    start_time = time.time()
    greeks = get_all_greeks(
        sigma=ivs.values,  # current option price
        S=data_repeated["Px"].values,  # underlying asset price
        K=data_repeated["Strike"].values,  # strike
        t=data_repeated["Annualized Time To Expiration"].values,  # normalized days to expiration
        r=data_repeated["Interest Free Rate"].values,  # interest free rate
        flag=data_repeated["Flag"].values,  # call or put
    )
    times_greeks.append(time.time() - start_time)

print(times)
print(times_greeks)
