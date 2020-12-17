from unittest import TestCase
import json
import pandas as pd, numpy as np
from numpy.testing import assert_array_almost_equal

from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility

from py_vollib_vectorized.models import vectorized_black_scholes, vectorized_black_scholes_merton

from py_vollib_vectorized.api import get_all_greeks, price_dataframe

class Test(TestCase):
    def setUp(self) -> None:
        with open("test_data_py_vollib.json", "rb") as f:
            d = json.load(f)
            self.test_df = pd.DataFrame(d["data"], index=d["index"], columns=d["columns"])
            self.test_df_calls = self.test_df.copy()
            self.test_df_calls["flag"] = "c"
            self.test_df_calls["q"] = 0
            self.test_df_puts = self.test_df.copy()
            self.test_df_puts["flag"] = "p"
            self.test_df_puts["q"] = 0

    def test_implied_volatility_vectorized(self):
        # Call
        ivs = vectorized_implied_volatility(
            price=vectorized_black_scholes(self.test_df_calls["flag"],
                                           self.test_df_calls["S"],
                                           self.test_df_calls["K"],
                                           self.test_df_calls["t"],
                                           self.test_df_calls["R"],
                                           self.test_df_calls["v"]),
            # price=self.test_df_puts["bs_put"].values,  # current option price
            S=self.test_df_calls["S"].values,  # underlying asset price
            K=self.test_df_calls["K"].values,  # strike
            t=self.test_df_calls["t"].values,  # normalized days to expiration
            r=self.test_df_calls["R"].values,  # interest free rate
            flag=self.test_df_calls["flag"].values,  # call or put
        )

        true_sigmas = self.test_df_calls["v"].values.ravel()
        pred_sigmas = ivs.values.ravel()
        test_array = np.isclose(true_sigmas, pred_sigmas, atol=1e-2)
        test_array[~test_array] = pred_sigmas[~test_array] == 0
        self.assertTrue(all(test_array))

        # Put

        ivs = vectorized_implied_volatility(
            price=vectorized_black_scholes(self.test_df_puts["flag"],
                                           self.test_df_puts["S"],
                                           self.test_df_puts["K"],
                                           self.test_df_puts["t"],
                                           self.test_df_puts["R"],
                                           self.test_df_puts["v"]).values,
            # price=self.test_df_puts["bs_put"].values,  # current option price
            S=self.test_df_puts["S"].values,  # underlying asset price
            K=self.test_df_puts["K"].values,  # strike
            t=self.test_df_puts["t"].values,  # normalized days to expiration
            r=self.test_df_puts["R"].values,  # interest free rate
            flag=self.test_df_puts["flag"].values,  # call or put
        )

        true_sigmas = self.test_df_puts["v"].values.ravel()
        pred_sigmas = ivs.values.ravel()
        test_array = np.isclose(true_sigmas, pred_sigmas, atol=1e-2)
        test_array[~test_array] = pred_sigmas[~test_array] == 0
        self.assertTrue(all(test_array))

    def test_black_scholes_vectorized(self):
        prices = vectorized_black_scholes(
            sigma=self.test_df_calls["v"].values,  # current option price
            S=self.test_df_calls["S"].values,  # underlying asset price
            K=self.test_df_calls["K"].values,  # strike
            t=self.test_df_calls["t"].values,  # normalized days to expiration
            r=self.test_df_calls["R"].values,  # interest free rate
            flag=self.test_df_calls["flag"].values,  # call or put
        )

        # returns None when conditions matches
        self.assertIsNone(
            assert_array_almost_equal(self.test_df_calls["bs_call"].values.ravel(), prices.values.ravel()))

        prices = vectorized_black_scholes(
            sigma=self.test_df_puts["v"].values,  # current option price
            S=self.test_df_puts["S"].values,  # underlying asset price
            K=self.test_df_puts["K"].values,  # strike
            t=self.test_df_puts["t"].values,  # normalized days to expiration
            r=self.test_df_puts["R"].values,  # interest free rate
            flag=self.test_df_puts["flag"].values,  # call or put
        )

        # returns None when conditions matches
        self.assertIsNone(assert_array_almost_equal(self.test_df_puts["bs_put"].values.ravel(), prices.values.ravel()))

    def test_black_scholes_merton_vectorized(self):
        prices = vectorized_black_scholes_merton(
            sigma=self.test_df_calls["v"].values,  # current option price
            S=self.test_df_calls["S"].values,  # underlying asset price
            K=self.test_df_calls["K"].values,  # strike
            t=self.test_df_calls["t"].values,  # normalized days to expiration
            r=self.test_df_calls["R"].values,  # interest free rate
            flag=self.test_df_calls["flag"].values,  # call or put
            q=self.test_df_calls["q"].values
        )

        # returns None when conditions matches
        self.assertIsNone(
            assert_array_almost_equal(self.test_df_calls["bs_call"].values.ravel(), prices.values.ravel()))

        prices = vectorized_black_scholes_merton(
            sigma=self.test_df_puts["v"].values,  # current option price
            S=self.test_df_puts["S"].values,  # underlying asset price
            K=self.test_df_puts["K"].values,  # strike
            t=self.test_df_puts["t"].values,  # normalized days to expiration
            r=self.test_df_puts["R"].values,  # interest free rate
            flag=self.test_df_puts["flag"].values,  # call or put
            q=self.test_df_puts["q"].values
        )

        # returns None when conditions matches
        self.assertIsNone(assert_array_almost_equal(self.test_df_puts["bs_put"].values.ravel(), prices.values.ravel()))


    def test_api_get_all_greeks(self):
        # Calls
        greeks_dataframe = get_all_greeks(
            sigma=self.test_df_calls["v"].values,  # current option price
            S=self.test_df_calls["S"].values,  # underlying asset price
            K=self.test_df_calls["K"].values,  # strike
            t=self.test_df_calls["t"].values,  # normalized days to expiration
            r=self.test_df_calls["R"].values,  # interest free rate
            flag=self.test_df_calls["flag"].values,  # call or put
        )

        self.assertIsNone(assert_array_almost_equal(greeks_dataframe["delta"], self.test_df_calls["CD"]))
        self.assertIsNone(assert_array_almost_equal(greeks_dataframe["gamma"], self.test_df_calls["CG"]))
        self.assertIsNone(assert_array_almost_equal(greeks_dataframe["vega"], self.test_df_calls["CV"] * .01, decimal=3))

        # Puts
        greeks_dataframe = get_all_greeks(
            sigma=self.test_df_puts["v"].values,  # current option price
            S=self.test_df_puts["S"].values,  # underlying asset price
            K=self.test_df_puts["K"].values,  # strike
            t=self.test_df_puts["t"].values,  # normalized days to expiration
            r=self.test_df_puts["R"].values,  # interest free rate
            flag=self.test_df_puts["flag"].values,  # call or put
        )

        self.assertIsNone(assert_array_almost_equal(greeks_dataframe["delta"], self.test_df_puts["PD"]))
        self.assertIsNone(assert_array_almost_equal(greeks_dataframe["gamma"], self.test_df_puts["PG"]))
        self.assertIsNone(assert_array_almost_equal(greeks_dataframe["vega"], self.test_df_puts["PV"] * .01, decimal=3))

    def test_api_price_dataframe(self):
        for model in ["black", "black_scholes", "black_scholes_merton"]:
            for what_to_test in ["sigma", "price", "sigma_price"]:
                resulting_df = price_dataframe(self.test_df_puts,
                                               flag_col='flag',
                                               underlying_price_col='S',
                                               strike_col='K',
                                               annualized_tte_col='t',
                                               riskfree_rate_col='R',
                                               sigma_col='v' if "sigma" in what_to_test else None,
                                               model=model,
                                               inplace=False,
                                               dividend_col='q',
                                               price_col='bs_put' if "price" in what_to_test else None
                                               )
                if "sigma" not in what_to_test:
                    self.assertTrue("IV" in resulting_df.columns)
                if "price" not in what_to_test:
                    self.assertTrue("Price" in resulting_df.columns)
                self.assertTrue("delta" in resulting_df.columns)
                self.assertTrue("gamma" in resulting_df.columns)
                self.assertTrue("theta" in resulting_df.columns)
                self.assertTrue("vega" in resulting_df.columns)
                self.assertTrue("rho" in resulting_df.columns)

    def test_validity_greeks(self):
        from py_vollib_vectorized.greeks import delta as my_delta, theta as my_theta, gamma as my_gamma, \
            rho as my_rho, \
            vega as my_vega

        from py_vollib.black_scholes.greeks.numerical import delta as original_delta, gamma as original_gamma, \
            rho as original_rho, theta as original_theta, vega as original_vega

        data = pd.read_csv("fake_data.csv")
        ivs = vectorized_implied_volatility(
            price=data["MidPx"].values,  # current option price
            S=data["Px"].values,  # underlying asset price
            K=data["Strike"].values,  # strike
            t=data["Annualized Time To Expiration"].values,  # nroamlized days to expiration
            r=data["Interest Free Rate"].values,  # interest free rate
            flag=data["Flag"].values,  # call or put
        )

        data["IV"] = ivs

        my_deltas = my_delta(
            flag=data["Flag"],
            S=data["Px"],
            K=data["Strike"],
            t=data["Annualized Time To Expiration"],
            r=data["Interest Free Rate"],
            sigma=data["IV"],
        )
        my_thetas = my_theta(
            flag=data["Flag"],
            S=data["Px"],
            K=data["Strike"],
            t=data["Annualized Time To Expiration"],
            r=data["Interest Free Rate"],
            sigma=data["IV"],
        )
        my_rhos = my_rho(
            flag=data["Flag"],
            S=data["Px"],
            K=data["Strike"],
            t=data["Annualized Time To Expiration"],
            r=data["Interest Free Rate"],
            sigma=data["IV"],
        )
        my_vegas = my_vega(
            flag=data["Flag"],
            S=data["Px"],
            K=data["Strike"],
            t=data["Annualized Time To Expiration"],
            r=data["Interest Free Rate"],
            sigma=data["IV"],
        )
        my_gammas = my_gamma(
            flag=data["Flag"],
            S=data["Px"],
            K=data["Strike"],
            t=data["Annualized Time To Expiration"],
            r=data["Interest Free Rate"],
            sigma=data["IV"],
        )
        orig_ds, orig_ts, orig_rs, orig_vs, orig_gs = [], [], [], [], []
        for i in range(len(data)):
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
            orig_ds.append(orig_d.iloc[0])
            orig_ts.append(orig_t.iloc[0])
            orig_rs.append(orig_r.iloc[0])
            orig_vs.append(orig_v.iloc[0])
            orig_gs.append(orig_g.iloc[0])

        orig_ds = np.array(orig_ds)
        orig_ts = np.array(orig_ts)
        orig_rs = np.array(orig_rs)
        orig_vs = np.array(orig_vs)
        orig_gs = np.array(orig_gs)

        self.assertIsNone(assert_array_almost_equal(my_deltas, orig_ds))
        self.assertIsNone(assert_array_almost_equal(my_gammas, orig_gs))
        self.assertIsNone(assert_array_almost_equal(my_rhos, orig_rs))
        self.assertIsNone(assert_array_almost_equal(my_vegas, orig_vs))
        self.assertIsNone(assert_array_almost_equal(my_thetas, orig_ts))

    def test_implied_volatility_broadcasting(self):
        for i in range(6):
            ivs = vectorized_implied_volatility(
                price=np.repeat([0.10], 2) if i == 0 else 0.10,
                S=np.repeat([30], 2) if i == 1 else 30,  # underlying asset price
                K=np.repeat([30], 2) if i == 2 else 30,  # strike
                t=np.repeat([10/365.], 2) if i == 3 else 10/365.,  # normalized days to expiration
                r=np.repeat([0.05], 2) if i == 4 else 0.05,  # interest free rate
                flag=np.repeat(["c"], 2) if i == 5 else "c",  # call or put
                return_as="numpy"
            )

            self.assertTrue(ivs[0] == ivs[1])

    def test_greeks_broadcasting(self):
        for i in range(6):
            greeks_dataframe = get_all_greeks(
                sigma=np.repeat([0.10], 2) if i == 0 else 0.10,
                S=np.repeat([30], 2) if i == 1 else 30,  # underlying asset price
                K=np.repeat([30], 2) if i == 2 else 30,  # strike
                t=np.repeat([10 / 365.], 2) if i == 3 else 10 / 365.,  # normalized days to expiration
                r=np.repeat([0.05], 2) if i == 4 else 0.05,  # interest free rate
                flag=np.repeat(["c"], 2) if i == 5 else "c",  # call or put
                return_as="numpy"
            )