import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from threee.constants import USER_DATA_FILES
from threee.exceptions import OperationalException




def create_datadir(config: Dict[str, Any], datadir: Optional[str] = None) -> Path:

    folder = Path(datadir) if datadir else Path(f"{config['user_data_dir']}/data")
    if not datadir:
        # 데이터 생성
        exchange_name = config.get('exchange', {}).get('name').lower()
        folder = folder.joinpath(exchange_name)

    if not folder.is_dir():
        folder.mkdir(parents=True)

    return folder


def chown_user_directory(directory: Path) -> None:
    """
    도커에서 새로운 config파일 생성
    """
    import os
    if os.environ.get('FT_APP_ENV') == 'docker':
        try:
            import subprocess
            subprocess.check_output(
                ['sudo', 'chown', '-R', 'ftuser:', str(directory.resolve())])
        except Exception:
            None

def create_userdata_dir(directory: str, create_dir: bool = False) -> Path:
    """
    유저 각각 필요한 정보 생성
    """
    sub_dirs = ["backtest_results", "data", "hyperopts", "hyperopt_results", "logs",
                "notebooks", "plot", "strategies", ]
    folder = Path(directory)
    chown_user_directory(folder)
    if not folder.is_dir():
        if create_dir:
            folder.mkdir(parents=True)
        else:
            None
    for f in sub_dirs:
        subfolder = folder / f
        if not subfolder.is_dir():
            subfolder.mkdir(parents=False)
    return folder
