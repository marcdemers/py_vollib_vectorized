import unittest
from unittest import TestCase

class Test(TestCase):
    def test_02_vectorized_imports(self):
        print("test02")
        import py_vollib.black.implied_volatility
        import py_vollib.black_scholes.implied_volatility
        import py_vollib.black_scholes_merton.implied_volatility

        import py_vollib.black
        import py_vollib.black_scholes
        import py_vollib.black_scholes_merton

        import py_vollib.black.greeks.numerical
        import py_vollib.black_scholes.greeks.numerical
        import py_vollib.black_scholes_merton.greeks.numerical

        import py_vollib_vectorized
        #IVs
        self.assertTrue(
            py_vollib.black.implied_volatility.implied_volatility.__module__ == "py_vollib_vectorized.implied_volatility")
        self.assertTrue(
            py_vollib.black_scholes.implied_volatility.implied_volatility.__module__ == "py_vollib_vectorized.implied_volatility")
        self.assertTrue(
            py_vollib.black_scholes_merton.implied_volatility.implied_volatility.__module__ == "py_vollib_vectorized.implied_volatility")

        #Models
        self.assertTrue(
            py_vollib.black.black.__module__ == "py_vollib_vectorized.models")
        self.assertTrue(
            py_vollib.black_scholes.black_scholes.__module__ == "py_vollib_vectorized.models")
        self.assertTrue(
            py_vollib.black_scholes_merton.black_scholes_merton.__module__ == "py_vollib_vectorized.models")

        #Greeks
        self.assertTrue(
            py_vollib.black.greeks.numerical.delta.__module__ == "py_vollib_vectorized.greeks")
        self.assertTrue(
            py_vollib.black_scholes.greeks.numerical.delta.__module__ == "py_vollib_vectorized.greeks")
        self.assertTrue(
            py_vollib.black_scholes_merton.greeks.numerical.delta.__module__ == "py_vollib_vectorized.greeks")

