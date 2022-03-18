class FreqtradeException(Exception):
    pass


class OperationalException(FreqtradeException):
    pass


class DependencyException(FreqtradeException):
    pass


class PricingError(DependencyException):
    pass


class ExchangeError(DependencyException):
    pass


class InvalidOrderException(ExchangeError):
    pass


class RetryableOrderError(InvalidOrderException):
    pass


class InsufficientFundsError(InvalidOrderException):
    pass


class TemporaryError(ExchangeError):
    pass


class DDosProtection(TemporaryError):
    pass


class StrategyError(FreqtradeException):
    pass
