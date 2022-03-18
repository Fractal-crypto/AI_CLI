import logging
from copy import deepcopy

from threee.exceptions import StrategyError



def strategy_safe_wrapper(f, message: str = "", default_retval=None, supress_error=False):
    def wrapper(*args, **kwargs):
        try:
            if 'trade' in kwargs:
                kwargs['trade'] = deepcopy(kwargs['trade'])
            return f(*args, **kwargs)
        except ValueError as error:
            pass
            
            if default_retval is None and not supress_error:
                raise StrategyError(str(error)) from error
            return default_retval
        except Exception as error:
            pass
            if default_retval is None and not supress_error:
                raise StrategyError(str(error)) from error
            return default_retval

    return wrapper
