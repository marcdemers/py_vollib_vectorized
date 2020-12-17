import json
import signal
from functools import wraps
from time import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.simplefilter("ignore")

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)

def timeit(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time()
        signal.alarm(60)
        try:
            result = fn(*args, **kwargs)
        except TimeoutException:
            pass
        else:
            signal.alarm(60)

        return time() - start

    return wrapper


with open("tests/test_data_py_vollib.json", "rb") as f:
    d = json.load(f)
    test_df = pd.DataFrame(d["data"], index=d["index"], columns=d["columns"])
    test_df_calls = test_df.copy()
    test_df_calls["flag"] = "c"
    test_df_calls["q"] = 0
    test_df_puts = test_df.copy()
    test_df_puts["flag"] = "p"
    test_df_puts["q"] = 0

test_df = pd.concat((test_df_calls.iloc[:100], test_df_puts.iloc[:100]), 0)
big_test_df = pd.concat([test_df for _ in range(500)]).sample(frac=1., random_state=0)


@timeit
def pd_apply(N):
    from py_vollib.black_scholes.implied_volatility import implied_volatility
    from py_vollib.black_scholes import black_scholes

    tt = big_test_df.iloc[:N]
    tt["price"] = tt.apply(
        lambda row: black_scholes(row["flag"], row["S"], row["K"], row["t"], row["R"], row["v"]), axis=1)
    tt.apply(lambda row: implied_volatility(row["price"], row["S"], row["K"], row["t"], row["R"], row["flag"]), axis=1)


@timeit
def for_loop(N):
    from py_vollib.black_scholes.implied_volatility import implied_volatility
    from py_vollib.black_scholes import black_scholes

    tt = big_test_df.iloc[:N]

    prices = []
    for i in range(len(tt)):
        prices.append(black_scholes(tt.iloc[i]["flag"], tt.iloc[i]["S"], tt.iloc[i]["K"], tt.iloc[i]["t"],
                                    tt.iloc[i]["R"], tt.iloc[i]["v"]))

    tt["price"] = prices

    ivs = []
    for i in range(len(tt)):
        ivs.append(
            implied_volatility(tt.iloc[i]["price"], tt.iloc[i]["S"], tt.iloc[i]["K"], tt.iloc[i]["t"], tt.iloc[i]["R"],
                               tt.iloc[i]["flag"])
        )


@timeit
def iterrows(N):
    from py_vollib.black_scholes.implied_volatility import implied_volatility
    from py_vollib.black_scholes import black_scholes

    tt = big_test_df.iloc[:N]

    prices = []
    for _, row in tt.iterrows():
        prices.append(black_scholes(row["flag"], row["S"], row["K"], row["t"],
                                    row["R"], row["v"]))

    tt["price"] = prices

    ivs = []
    for _, row in tt.iterrows():
        ivs.append(
            implied_volatility(row["price"], row["S"], row["K"], row["t"], row["R"],
                               row["flag"])
        )


@timeit
def listcomp(N):
    from py_vollib.black_scholes.implied_volatility import implied_volatility
    from py_vollib.black_scholes import black_scholes

    tt = big_test_df.iloc[:N]

    prices = [black_scholes(tt.iloc[i]["flag"], tt.iloc[i]["S"], tt.iloc[i]["K"], tt.iloc[i]["t"],
                            tt.iloc[i]["R"], tt.iloc[i]["v"]) for i in range(len(tt))]

    tt["price"] = prices

    ivs = [implied_volatility(tt.iloc[i]["price"], tt.iloc[i]["S"], tt.iloc[i]["K"], tt.iloc[i]["t"], tt.iloc[i]["R"],
                              tt.iloc[i]["flag"]) for i in range(len(tt))]


@timeit
def vectorized(N):
    import py_vollib_vectorized

    tt = big_test_df.iloc[:N]
    prices = py_vollib_vectorized.vectorized_black_scholes(tt["flag"], tt["S"], tt["K"], tt["t"],
                                                           tt["R"], tt["v"])

    tt["price"] = prices
    py_vollib_vectorized.vectorized_implied_volatility(tt["price"], tt["S"], tt["K"], tt["t"], tt["R"],
                                                       tt["flag"])


n_contracts = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
timings = []

for N in n_contracts:
    # time_apply, time_forloop, time_iterrows, time_listcomp, time_vectorized = [], [], [], [], []
    for repetition in range(10):
        time_apply = (pd_apply(N))
        time_forloop =(for_loop(N))
        time_iterrows = (iterrows(N))
        time_listcomp = (listcomp(N))
        time_vectorized = (vectorized(N))
        timings.append((N, time_apply, time_forloop, time_iterrows, time_listcomp, time_vectorized))

df = pd.DataFrame(timings, columns=["N", "apply", "forloop", "iterrows", "listcomp", "vectorized (this library)"])
g = sns.lineplot(data=df.melt(id_vars="N", var_name="method"), x="N", y="value", hue="method", markers=True)
g.set_xscale("log")

plt.title("Time required to price up to 10 million contracts\n(10 runs, capped at 60s)")
plt.xlabel("Number of contracts")
plt.ylabel("Time (s)")
plt.tight_layout()
plt.savefig("docs/_static/benchmark.png")
plt.show()

# Tabulation
df.groupby("N").mean().T.to_markdown(open("docs/_static/benchmark_table.rst", "w"), index=True,
                                     tablefmt="grid")