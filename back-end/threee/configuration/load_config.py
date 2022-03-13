"""
This module contain functions to load the configuration file
"""
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict

import rapidjson

from threee.exceptions import OperationalException


CONFIG_PARSE_MODE = rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS


def load_file(path: Path) -> Dict[str, Any]:
    try:
        with path.open('r') as file:
            config = rapidjson.load(file, parse_mode=CONFIG_PARSE_MODE)
    except FileNotFoundError:
        raise OperationalException(f'File "{path}" not found!')
    return config


def load_config_file(path: str) -> Dict[str, Any]:
    """
    주어진 경로로 가서 config파일 정보 불러오기
    """
    try:
        # Read config from stdin if requested in the options
        with open(path) if path != '-' else sys.stdin as file:
            config = rapidjson.load(file, parse_mode=CONFIG_PARSE_MODE)
    except FileNotFoundError:
        raise OperationalException("경로 재지정")
    except rapidjson.JSONDecodeError as e:
        err_range = log_config_error_range(path, str(e))
        raise OperationalException("딕셔너리 형태로 넣어주시오")

    return config
