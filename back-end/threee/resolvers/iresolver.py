import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

from threee.exceptions import OperationalException


class IResolver:
    """
    사용자 정의 클래스를 로드
    """
    object_type: Type[Any]
    object_type_str: str
    user_subdir: Optional[str] = None
    initial_search_path: Optional[Path]

    @classmethod
    def build_search_paths(cls, config: Dict[str, Any], user_subdir: Optional[str] = None,
                           extra_dir: Optional[str] = None) -> List[Path]:

        abs_paths: List[Path] = []
        if cls.initial_search_path:
            abs_paths.append(cls.initial_search_path)

        if user_subdir:
            abs_paths.insert(0, config['user_data_dir'].joinpath(user_subdir))

        if extra_dir:
            # Add extra directory to the top of the search paths
            abs_paths.insert(0, Path(extra_dir).resolve())

        return abs_paths

    @classmethod
    def _get_valid_object(cls, module_path: Path, object_name: Optional[str],
                          enum_failed: bool = False) -> Iterator[Any]:
        spec = importlib.util.spec_from_file_location(object_name or "", str(module_path))
        if not spec:
            return iter([None])

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except (ModuleNotFoundError, SyntaxError, ImportError, NameError) as err:
            pass
            if enum_failed:
                return iter([None])

        valid_objects_gen = (
            (obj, inspect.getsource(module)) for
            name, obj in inspect.getmembers(
                module, inspect.isclass) if ((object_name is None or object_name == name)
                                             and issubclass(obj, cls.object_type)
                                             and obj is not cls.object_type)
        )
        return valid_objects_gen

    @classmethod
    def _search_object(cls, directory: Path, *, object_name: str, add_source: bool = False
                       ) -> Union[Tuple[Any, Path], Tuple[None, None]]:
        for entry in directory.iterdir():

            if entry.suffix != '.py':

                continue
            if entry.is_symlink() and not entry.is_file():
                continue
            module_path = entry.resolve()

            obj = next(cls._get_valid_object(module_path, object_name), None)

            if obj:
                obj[0].__file__ = str(entry)
                if add_source:
                    obj[0].__source__ = obj[1]
                return (obj[0], module_path)
        return (None, None)

    @classmethod
    def _load_object(cls, paths: List[Path], *, object_name: str, add_source: bool = False,
                     kwargs: dict = {}) -> Optional[Any]:

        for _path in paths:
            try:
                (module, module_path) = cls._search_object(directory=_path,
                                                           object_name=object_name,
                                                           add_source=add_source)
                if module:
                    return module(**kwargs)
            except FileNotFoundError:
                pass

        return None

    @classmethod
    def load_object(cls, object_name: str, config: dict, *, kwargs: dict,
                    extra_dir: Optional[str] = None) -> Any:

        abs_paths = cls.build_search_paths(config,
                                           user_subdir=cls.user_subdir,
                                           extra_dir=extra_dir)

        found_object = cls._load_object(paths=abs_paths, object_name=object_name,
                                        kwargs=kwargs)
        if found_object:
            return found_object


    @classmethod
    def search_all_objects(cls, directory: Path,
                           enum_failed: bool) -> List[Dict[str, Any]]:
        objects = []
        for entry in directory.iterdir():
            if entry.suffix != '.py':

                continue
            module_path = entry.resolve()
            for obj in cls._get_valid_object(module_path, object_name=None,
                                             enum_failed=enum_failed):
                objects.append(
                    {'name': obj[0].__name__ if obj is not None else '',
                     'class': obj[0] if obj is not None else None,
                     'location': entry,
                     })
        return objects
