from py_vollib_vectorized.iv_models import *



import pandas as pd

data = pd.read_csv("fake_data.csv")

print(data)

ivs = implied_volatility_vectorized(
    price=data["MidPx"].values,  # current option price
    S=data["Px"].values,  # underlying asset price
    K=data["Strike"].values,  # strike
    t=data["Annualized Time To Expiration"].values,  # nroamlized days to expiration
    r=data["Interest Free Rate"].values,  # interest free rate
    flag=data["Flag"].values,  # call or put
)

# print(ivs)


data["IV"] = ivs


#### greeks

from py_vollib_vectorized.entrypoints import delta as my_delta, theta as my_theta, gamma as my_gamma, rho as my_rho, vega as my_vega

from py_vollib.black_scholes.greeks.numerical import delta as original_delta, gamma as original_gamma, rho as original_rho, theta as original_theta, vega as original_vega
print(len(data))

from time import time
run_time = []
checks = []
for run in range(10):
    tic = time()
    i = 11
    for i in range(len(data)):

        d = my_delta(
            flag=data["Flag"].iloc[i:i+1].values,
            S=data["Px"].iloc[i:i + 1].values,
            K=data["Strike"].iloc[i:i + 1].values,
            t=data["Annualized Time To Expiration"].iloc[i:i + 1].values,
            r=data["Interest Free Rate"].iloc[i:i + 1].values,
            sigma=data["IV"].iloc[i:i + 1].values,
        )
        t = my_theta(
            flag=data["Flag"].iloc[i:i + 1].values,
            S=data["Px"].iloc[i:i + 1].values,
            K=data["Strike"].iloc[i:i + 1].values,
            t=data["Annualized Time To Expiration"].iloc[i:i + 1].values,
            r=data["Interest Free Rate"].iloc[i:i + 1].values,
            sigma=data["IV"].iloc[i:i + 1].values,
        )
        r = my_rho(
            flag=data["Flag"].iloc[i:i + 1].values,
            S=data["Px"].iloc[i:i + 1].values,
            K=data["Strike"].iloc[i:i + 1].values,
            t=data["Annualized Time To Expiration"].iloc[i:i + 1].values,
            r=data["Interest Free Rate"].iloc[i:i + 1].values,
            sigma=data["IV"].iloc[i:i + 1].values,
        )
        v = my_vega(
            flag=data["Flag"].iloc[i:i + 1].values,
            S=data["Px"].iloc[i:i + 1].values,
            K=data["Strike"].iloc[i:i + 1].values,
            t=data["Annualized Time To Expiration"].iloc[i:i + 1].values,
            r=data["Interest Free Rate"].iloc[i:i + 1].values,
            sigma=data["IV"].iloc[i:i + 1].values,
        )
        g = my_gamma(
            flag=data["Flag"].iloc[i:i + 1].values,
            S=data["Px"].iloc[i:i + 1].values,
            K=data["Strike"].iloc[i:i + 1].values,
            t=data["Annualized Time To Expiration"].iloc[i:i + 1].values,
            r=data["Interest Free Rate"].iloc[i:i + 1].values,
            sigma=data["IV"].iloc[i:i + 1].values,
        )

        print(i)
        # print(d)
        # print(t)
        print(r)
        orig_d = original_delta(
            flag=data["Flag"].iloc[i],
            S=data["Px"].iloc[i],
            K=data["Strike"].iloc[i],
            t=data["Annualized Time To Expiration"].iloc[i],
            r=data["Interest Free Rate"].iloc[i],
            sigma=data["IV"].iloc[i],
        )
        orig_t = original_theta(
            flag=data["Flag"].iloc[i],
            S=data["Px"].iloc[i],
            K=data["Strike"].iloc[i],
            t=data["Annualized Time To Expiration"].iloc[i],
            r=data["Interest Free Rate"].iloc[i],
            sigma=data["IV"].iloc[i],
        )
        orig_r = original_rho(
            flag=data["Flag"].iloc[i],
            S=data["Px"].iloc[i],
            K=data["Strike"].iloc[i],
            t=data["Annualized Time To Expiration"].iloc[i],
            r=data["Interest Free Rate"].iloc[i],
            sigma=data["IV"].iloc[i],
        )
        orig_v = original_vega(
            flag=data["Flag"].iloc[i],
            S=data["Px"].iloc[i],
            K=data["Strike"].iloc[i],
            t=data["Annualized Time To Expiration"].iloc[i],
            r=data["Interest Free Rate"].iloc[i],
            sigma=data["IV"].iloc[i],
        )
        orig_g = original_gamma(
            flag=data["Flag"].iloc[i],
            S=data["Px"].iloc[i],
            K=data["Strike"].iloc[i],
            t=data["Annualized Time To Expiration"].iloc[i],
            r=data["Interest Free Rate"].iloc[i],
            sigma=data["IV"].iloc[i],
        )

        print(d[0], orig_d)
        # print(t == orig_t)
        checks.append(r[0] == orig_r and v[0] == orig_v and g[0] == orig_g and d[0] == orig_d and t[0] == orig_t)
        print("**"*10)

    toc = time()
    run_time.append(toc-tic)

print("check verification:", np.all(checks))
print("times", run_time)