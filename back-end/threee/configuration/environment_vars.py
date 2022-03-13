import logging
import os
from typing import Any, Dict

from threee.constants import ENV_VAR_PREFIX
from threee.misc import deep_merge_dicts


def get_var_typed(val):
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            if val.lower() in ('t', 'true'):
                return True
            elif val.lower() in ('f', 'false'):
                return False
    return val


def flat_vars_to_nested_dict(env_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Os 정보와 각 유저데이터 환경값 저장
    """
    no_convert = ['CHAT_ID']
    relevant_vars: Dict[str, Any] = {}

    for env_var, val in sorted(env_dict.items()):
        if env_var.startswith(prefix):

            key = env_var.replace(prefix, '')
            for k in reversed(key.split('__')):
                val = {k.lower(): get_var_typed(val)
                       if type(val) != dict and k not in no_convert else val}
            relevant_vars = deep_merge_dicts(val, relevant_vars)
    return relevant_vars


def enironment_vars_to_dict() -> Dict[str, Any]:
    # 환경값 불러오기
    return flat_vars_to_nested_dict(os.environ.copy(), ENV_VAR_PREFIX)
