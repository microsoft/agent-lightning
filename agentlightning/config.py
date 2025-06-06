"""
This file is not carefully reviewed.
It might contain unintentional bugs and issues.
Please always review the parsed construction arguments before using them.
"""

from __future__ import annotations

import argparse
import inspect
import logging
from typing import (
    Any,
    List,
    Type,
    TypeVar,
    Union,
    _GenericAlias,  # type: ignore
    get_origin,
    get_args,
    Tuple,
    Callable,
    overload,
    Dict,
    get_type_hints,
)

CliConfigurable = Any

logger = logging.getLogger(__name__)

# TypeVars for precise return type hinting with overloads
_C = TypeVar("_C", bound=CliConfigurable)
_C1 = TypeVar("_C1", bound=CliConfigurable)
_C2 = TypeVar("_C2", bound=CliConfigurable)
_C3 = TypeVar("_C3", bound=CliConfigurable)
_C4 = TypeVar("_C4", bound=CliConfigurable)


# Custom type for CLI arguments that can be string or None
def nullable_str(value: str) -> str | None:
    """Converts specific string values (case-insensitive) to None, otherwise returns the string."""
    if value.lower() in ["none", "null", "~", "nil"]:  # Define keywords for None
        return None
    return value


def nullable_int(value: str) -> int | None:
    """Converts specific string values (case-insensitive) to None, otherwise returns the integer."""
    if value.lower() in ["none", "null", "~", "nil"]:  # Define keywords for None
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: '{value}'")


def nullable_float(value: str) -> float | None:
    """Converts specific string values (case-insensitive) to None, otherwise returns the float."""
    if value.lower() in ["none", "null", "~", "nil"]:  # Define keywords for None
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{value}'")


def _str_to_bool(v: str) -> bool:
    """Converts common string representations of bool to Python bool (case-insensitive)."""
    if isinstance(v, bool):  # Allow passing bools directly if used programmatically
        return v
    lowered_v = v.lower()
    if lowered_v in ("yes", "true", "t", "y", "1"):
        return True
    elif lowered_v in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected (e.g., 'true', 'false', 'yes', 'no'), got '{v}'")


def _get_param_type_details(param_annotation: Any) -> Tuple[Any, bool, bool]:
    """
    Determines the core type, if it's Optional, and if it's a List.
    Returns: (core_type, is_optional, is_list)
    - For Optional[T]: (T, True, is_list_status_of_T)
    - For List[T]: (List[T], is_optional_status_of_List, True)
    - For Optional[List[T]]: (List[T], True, True)
    """
    is_optional = False
    is_list = False
    current_type = param_annotation

    # Check for outer Optional
    origin = get_origin(current_type)
    if origin is Union:
        union_args = get_args(current_type)
        if len(union_args) == 2 and type(None) in union_args:
            is_optional = True
            current_type = next(arg for arg in union_args if arg is not type(None))  # Unwrap Optional

    # Check if the (potentially unwrapped) type is a List
    origin = get_origin(current_type)  # Re-check origin after potential unwrap
    if origin is list or (isinstance(current_type, _GenericAlias) and current_type.__origin__ is list):
        is_list = True

    return current_type, is_optional, is_list


def _determine_argparse_type(param_type: Any) -> Callable[[str], Any]:
    """Determines the type for argparse based on parameter type details."""
    core_type, is_optional, _ = _get_param_type_details(param_type)
    if core_type is str and is_optional:
        return nullable_str  # Special handling for Optional[str]
    elif core_type is int and is_optional:
        return nullable_int
    elif core_type is float and is_optional:
        return nullable_float
    elif core_type is bool:
        return _str_to_bool  # Special handling for bool
    elif core_type in (int, float, str):
        return core_type
    return str  # Default to str if no specific type is provided (including empty)


def _determine_argparse_type_and_nargs(
    core_param_type: Any, is_param_list: bool  # The type after unwrapping an outer Optional
) -> Dict[str, Any]:
    """Determines the 'type' and 'nargs' for argparse based on parameter type details."""
    kwargs: Dict[str, Any] = {}

    if is_param_list:
        kwargs["nargs"] = "*"  # Allows zero or more arguments for lists
        list_item_annotations = get_args(core_param_type)  # For List[T], core_param_type is List[T]

        if list_item_annotations and list_item_annotations[0] is not Any:
            item_ann = list_item_annotations[0]
            # Check if the list item itself is, e.g., Optional[str] or bool
            kwargs["type"] = _determine_argparse_type(item_ann)
        else:
            kwargs["type"] = str
    else:  # Not a list
        kwargs["type"] = _determine_argparse_type(core_param_type)
    return kwargs


def _build_help_string(cls_name: str, param_name: str, core_type: Any, is_optional: bool, is_list: bool) -> str:
    """Constructs a descriptive help string for a CLI argument."""
    type_display_name = "Any"
    if core_type is not inspect.Parameter.empty:
        type_display_name = getattr(core_type, "__name__", str(core_type))

    if is_list:
        list_item_args = get_args(core_type)  # core_type is List[T] here
        item_name = "Any"
        if list_item_args and list_item_args[0] is not Any:
            inner_item_core_type, inner_item_optional, _ = _get_param_type_details(list_item_args[0])
            item_name = getattr(inner_item_core_type, "__name__", str(inner_item_core_type))
            if inner_item_optional:  # e.g. List[Optional[str]]
                item_name = f"Optional[{item_name}]"
        type_display_name = f"List[{item_name}]"

    full_type_display = f"Optional[{type_display_name}]" if is_optional and not is_list else type_display_name
    if is_optional and is_list:  # e.g. Optional[List[str]]
        full_type_display = f"Optional[{type_display_name}]"

    help_str = f"For {cls_name}: '{param_name}'. Inferred type: {full_type_display}."
    return help_str


def _add_argument_for_parameter(
    parser: argparse.ArgumentParser,
    cls: Type[CliConfigurable],
    param_name: str,
    param_obj: inspect.Parameter,
    dest_name: str,
    resolved_param_annotation: Any = None,
) -> None:
    """Configures and adds a single CLI argument for an __init__ parameter."""
    if resolved_param_annotation is None:
        param_type_annotation = param_obj.annotation
    else:
        param_type_annotation = resolved_param_annotation

    # core_type is the main type (e.g., int, str, List[str]), after unwrapping the outermost Optional.
    # is_overall_optional indicates if the parameter itself can be None (e.g. param: Optional[T] = None)
    # is_list indicates if core_type is a List.
    core_type, is_overall_optional, is_list = _get_param_type_details(param_type_annotation)

    has_init_default = param_obj.default is not inspect.Parameter.empty
    init_default_value = param_obj.default if has_init_default else None

    argparse_kwargs = _determine_argparse_type_and_nargs(core_type if is_list else param_type_annotation, is_list)

    if has_init_default:
        argparse_kwargs["default"] = init_default_value
    elif is_overall_optional:  # Parameter is Optional (e.g. Optional[int]) and no explicit default in __init__
        argparse_kwargs["default"] = None  # So, if not provided on CLI, it becomes None.

    argparse_kwargs["help"] = _build_help_string(cls.__name__, param_name, core_type, is_overall_optional, is_list)

    if not has_init_default and not is_overall_optional:  # Required if no __init__ default AND not Optional
        argparse_kwargs["required"] = True
        if "default" in argparse_kwargs:  # Should not happen if logic is correct
            del argparse_kwargs["default"]

    cli_arg_name = f"--{cls.__name__.lower()}.{param_name.replace('_', '-')}"
    parser.add_argument(cli_arg_name, dest=dest_name, **argparse_kwargs)


def _add_arguments_for_class(
    parser: argparse.ArgumentParser,
    cls: Type[CliConfigurable],
    class_arg_configs_maps: Dict[Type[CliConfigurable], Dict[str, str]],  # Maps cls to {param_name: dest_name}
) -> None:
    """Adds all relevant CLI arguments for a given class by processing its __init__ parameters."""
    cls_name_lower = cls.__name__.lower()
    sig = inspect.signature(cls.__init__)

    try:
        # Resolve string annotations to actual types using get_type_hints.
        # For methods, get_type_hints automatically uses obj.__globals__ for globalns.
        resolved_hints = get_type_hints(cls.__init__)
    except Exception as e:
        logger.warning(
            f"Could not resolve type hints for {cls.__name__}.__init__ using get_type_hints: {e}. "
            f"CLI argument parsing for this class might be based on string annotations, "
            "which could be unreliable for complex types."
        )
        resolved_hints = {} # Fallback to an empty dict if resolution fails

    if cls not in class_arg_configs_maps:  # Ensure the class entry exists
        class_arg_configs_maps[cls] = {}

    for param_name, param_obj in sig.parameters.items():
        if param_name == "self":  # Skip 'self'
            continue

        dest_name = f"{cls_name_lower}_{param_name}"  # Unique destination for argparse
        class_arg_configs_maps[cls][param_name] = dest_name  # Store mapping for later instantiation

        # Use the resolved hint if available, otherwise fallback to param_obj.annotation (which might be a string)
        actual_param_annotation = resolved_hints.get(param_name, param_obj.annotation)
        _add_argument_for_parameter(parser, cls, param_name, param_obj, dest_name, actual_param_annotation)


def _create_argument_parser() -> argparse.ArgumentParser:
    """Creates and returns the main ArgumentParser with default settings."""
    return argparse.ArgumentParser(
        description="CLI configurator for application components.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Automatically shows default values in help
    )


def _instantiate_classes(
    parsed_args: argparse.Namespace,
    classes: Tuple[Type[CliConfigurable], ...],
    class_arg_configs_maps: Dict[Type[CliConfigurable], Dict[str, str]],
) -> Tuple[CliConfigurable, ...]:
    """Instantiates classes using the parsed CLI arguments and the stored mappings."""
    instances_list: List[CliConfigurable] = []
    for cls in classes:
        constructor_args: Dict[str, Any] = {}
        # Get the {__init__ param_name: argparse_dest_name} map for the current class
        param_to_dest_map = class_arg_configs_maps.get(cls, {})

        sig = inspect.signature(cls.__init__)
        for param_name_in_sig, _ in sig.parameters.items():
            if param_name_in_sig == "self":
                continue

            dest_name_for_arg = param_to_dest_map.get(param_name_in_sig)
            if dest_name_for_arg and hasattr(parsed_args, dest_name_for_arg):
                value = getattr(parsed_args, dest_name_for_arg)
                constructor_args[param_name_in_sig] = value
            # If an argument was required by argparse, parse_args() would have exited if missing.
            # If not required and not provided, its default value (set by argparse) is used.

        try:
            logger.info("Instantiating %s with args: %s", cls.__name__, constructor_args)
            instances_list.append(cls(**constructor_args))
        except Exception as e:
            parsed_args_for_cls = {k: getattr(parsed_args, v) for k, v in param_to_dest_map.items() if hasattr(parsed_args, v)}
            logger.error(
                f"Error instantiating {cls.__name__} with resolved args {constructor_args}. "
                f"Parsed args for class: "
                f"{parsed_args_for_cls}. "
                f"Error: {e}"
            )
            raise

    return tuple(instances_list)


@overload
def lightning_cli(cls1: Type[_C1]) -> _C1: ...
@overload
def lightning_cli(cls1: Type[_C1], cls2: Type[_C2]) -> Tuple[_C1, _C2]: ...
@overload
def lightning_cli(cls1: Type[_C1], cls2: Type[_C2], cls3: Type[_C3]) -> Tuple[_C1, _C2, _C3]: ...
@overload
def lightning_cli(cls1: Type[_C1], cls2: Type[_C2], cls3: Type[_C3], cls4: Type[_C4]) -> Tuple[_C1, _C2, _C3, _C4]: ...
@overload  # Fallback for more than 4 or a dynamic number of classes
def lightning_cli(*classes: Type[CliConfigurable]) -> Tuple[CliConfigurable, ...]: ...


def lightning_cli(*classes: Type[CliConfigurable]) -> CliConfigurable | Tuple[CliConfigurable, ...]:
    """
    Parses command-line arguments to configure and instantiate provided CliConfigurable classes.

    Args:
        *classes: One or more classes that inherit from CliConfigurable. Each class's
                  __init__ parameters will be exposed as command-line arguments.

    Returns:
        A tuple of instantiated objects, corresponding to the input classes in order.
    """
    if not classes:
        return tuple()  # Return an empty tuple if no classes are provided

    parser = _create_argument_parser()

    # This map will store {cls: {init_param_name: argparse_dest_name}}
    class_arg_configs_maps: Dict[Type[CliConfigurable], Dict[str, str]] = {}

    for cls in classes:
        _add_arguments_for_class(parser, cls, class_arg_configs_maps)

    parsed_args = parser.parse_args()  # Uses sys.argv[1:] by default

    # Correctly handle single class case for return type matching overloads
    instances = _instantiate_classes(parsed_args, classes, class_arg_configs_maps)
    if len(classes) == 1:
        return instances[0]
    return instances
