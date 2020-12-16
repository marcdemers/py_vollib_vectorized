from unittest import TestCase
import json
import pandas as pd, numpy as np

# from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility
# from py_vollib_vectorized.models import vectorized_black_scholes, vectorized_black_scholes_merton
# from py_vollib_vectorized.api import get_all_greeks

def delete_module(modname, paranoid=None):
    from sys import modules
    try:
        thismod = modules[modname]
    except KeyError:
        raise ValueError(modname)
    these_symbols = dir(thismod)
    if paranoid:
        try:
            paranoid[:]  # sequence support
        except:
            raise ValueError('must supply a finite list for paranoid')
        else:
            these_symbols = paranoid[:]
    del modules[modname]
    for mod in modules.values():
        try:
            delattr(mod, modname)
        except AttributeError:
            pass
        if paranoid:
            for symbol in these_symbols:
                if symbol[:2] == '__':  # ignore special symbols
                    continue
                try:
                    delattr(mod, symbol)
                except AttributeError:
                    pass


class Test(TestCase):
    def test_imports_and_monkeypatches_models(self):
        from py_vollib.black import black
        from py_vollib.black_scholes import black_scholes
        from py_vollib.black_scholes_merton import black_scholes_merton

        self.assertTrue(black.__module__ == "py_vollib.black")
        self.assertTrue(black_scholes.__module__ == "py_vollib.black_scholes")
        self.assertTrue(black_scholes_merton.__module__ == "py_vollib.black_scholes_merton")

        import py_vollib_vectorized

        self.assertTrue(black.__module__ == "py_vollib_vectorized.black")
        self.assertTrue(black_scholes.__module__ == "py_vollib_vectorized.black_scholes")
        self.assertTrue(black_scholes_merton.__module__ == "py_vollib_vectorized.black_scholes_merton")

    def test_imports_and_monkeypatches_ivs_black(self):
        #TODO fix imports
        from py_vollib.black.implied_volatility import implied_volatility
        self.assertTrue(implied_volatility.__module__ == "py_vollib.black.implied_volatility")

        import py_vollib_vectorized
        self.assertTrue(implied_volatility.__module__ == "py_vollib_vectorized.implied_volatility")

    def test_imports_and_monkeypatches_ivs_black_scholes(self):

        from py_vollib.black_scholes.implied_volatility import implied_volatility
        self.assertTrue(implied_volatility.__module__ == "py_vollib.implied_volatility")

        import py_vollib_vectorized
        self.assertTrue(implied_volatility.__module__ == "py_vollib_vectorized.implied_volatility")

    def test_imports_and_monkeypatches_ivs_black_scholes_merton(self):

        from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
        self.assertTrue(implied_volatility.__module__ == "py_vollib.implied_volatility")

        import py_vollib_vectorized
        self.assertTrue(implied_volatility.__module__ == "py_vollib_vectorized.implied_volatility")


