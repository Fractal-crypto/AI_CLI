import logging
from typing import Any, Dict

from threee.enums import RunMode

from .config_validation import validate_config_consistency
from .configuration import Configuration

def setup_utils_configuration(args: Dict[str, Any], method: RunMode) -> Dict[str, Any]:
    """
    보조 cli커멘드  config 파일에서 불러오기

    """
    configuration = Configuration(args, method)
    config = configuration.get_config()
    config['dry_run'] = True
    validate_config_consistency(config)
    return config
