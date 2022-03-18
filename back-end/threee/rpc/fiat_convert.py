import datetime
import logging
from typing import Dict, List

from cachetools import TTLCache
from pycoingecko import CoinGeckoAPI
from requests.exceptions import RequestException

from threee.constants import SUPPORTED_FIAT


coingecko_mapping = {
    'eth': 'ethereum',
    'bnb': 'binancecoin',
    'sol': 'solana',
    'usdt': 'tether',
}


class CryptoToFiatConverter:
    """
    btc에서 usdt로 변환
    """
    __instance = None
    _coingekko: CoinGeckoAPI = None
    _coinlistings: List[Dict] = []
    _backoff: float = 0.0

    def __new__(cls):
        if CryptoToFiatConverter.__instance is None:
            CryptoToFiatConverter.__instance = object.__new__(cls)
            try:
                CryptoToFiatConverter._coingekko = CoinGeckoAPI()
            except BaseException:
                CryptoToFiatConverter._coingekko = None
        return CryptoToFiatConverter.__instance

    def __init__(self) -> None:

        self._pair_price: TTLCache = TTLCache(maxsize=500, ttl=6 * 60 * 60)

        self._load_cryptomap()

    def _load_cryptomap(self) -> None:
        try:
            self._coinlistings = [x for x in self._coingekko.get_coins_list()]
        except RequestException as request_exception:
            if "429" in str(request_exception):
                return

        except (Exception) as exception:
            pass
    def _get_gekko_id(self, crypto_symbol):
        if not self._coinlistings:
            if self._backoff <= datetime.datetime.now().timestamp():
                self._load_cryptomap()
                if not self._coinlistings:
                    return None
            else:
                return None
        found = [x for x in self._coinlistings if x['symbol'] == crypto_symbol]

        if crypto_symbol in coingecko_mapping.keys():
            found = [x for x in self._coinlistings if x['id'] == coingecko_mapping[crypto_symbol]]

        if len(found) == 1:
            return found[0]['id']

        if len(found) > 0:
            return None

    def convert_amount(self, crypto_amount: float, crypto_symbol: str, fiat_symbol: str) -> float:
        if crypto_symbol == fiat_symbol:
            return float(crypto_amount)
        price = self.get_price(crypto_symbol=crypto_symbol, fiat_symbol=fiat_symbol)
        return float(crypto_amount) * float(price)

    def get_price(self, crypto_symbol: str, fiat_symbol: str) -> float:

        crypto_symbol = crypto_symbol.lower()
        fiat_symbol = fiat_symbol.lower()
        inverse = False

        if crypto_symbol == 'usd':

            crypto_symbol = fiat_symbol
            fiat_symbol = 'usd'
            inverse = True

        symbol = f"{crypto_symbol}/{fiat_symbol}"
        if not self._is_supported_fiat(fiat=fiat_symbol):
            raise ValueError(f'The fiat {fiat_symbol} is not supported.')

        price = self._pair_price.get(symbol, None)

        if not price:
            price = self._find_price(
                crypto_symbol=crypto_symbol,
                fiat_symbol=fiat_symbol
            )
            if inverse and price != 0.0:
                price = 1 / price
            self._pair_price[symbol] = price

        return price

    def _is_supported_fiat(self, fiat: str) -> bool:

        return fiat.upper() in SUPPORTED_FIAT

    def _find_price(self, crypto_symbol: str, fiat_symbol: str) -> float:

        if not self._is_supported_fiat(fiat=fiat_symbol):
            raise ValueError(f'The fiat {fiat_symbol} is not supported.')


        if crypto_symbol == fiat_symbol:
            return 1.0

        _gekko_id = self._get_gekko_id(crypto_symbol)

        if not _gekko_id:
            return 0.0

        try:
            return float(
                self._coingekko.get_price(
                    ids=_gekko_id,
                    vs_currencies=fiat_symbol
                )[_gekko_id][fiat_symbol]
            )
        except Exception as exception:
            return 0.0
