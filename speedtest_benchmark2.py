#TODO compare no numba at all, dataframe apply, our solution.
# from py_vollib_vectorized import implied_volatility_vectorized
import json
import pandas as pd

with open("tests/test_data_py_vollib.json", "rb") as f:
    d = json.load(f)
    test_df = pd.DataFrame(d["data"], index=d["index"], columns=d["columns"])
    test_df_calls = test_df.copy()
    test_df_calls["flag"] = "c"
    test_df_calls["q"] = 0
    test_df_puts = test_df.copy()
    test_df_puts["flag"] = "p"
    test_df_puts["q"] = 0

# ivs = implied_volatility_vectorized(
#     price=test_df_puts["bs_put"].iloc[72],  # current option price
#     S=test_df_puts["S"].iloc[72],  # underlying asset price
#     K=test_df_puts["K"].iloc[72],  # strike
#     t=test_df_puts["t"].iloc[72],  # normalized days to expiration
#     r=test_df_puts["R"].iloc[72],  # interest free rate
#     flag=test_df_puts["flag"].iloc[72],  # call or put
# )
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes import black_scholes

value = test_df_puts.iloc[72]
implied_volatility(value["bs_put"], value["S"], value["K"], value["t"], value["R"], "p")

print("Done")
