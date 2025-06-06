"""
This file is not carefully reviewed.
It is to ensure the *somewhat* correctness of the code in agentlightning/config.py.
It can contain logically erroroneous expected values.
Please do not use the file as a reference for the expected behavior of config.
"""

import pytest
import argparse
from typing import (
    Any,
    List,
    Type,
    Union,
    Optional,
    Tuple,
    Callable,
    Dict,
    TypeVar,  # Added for completeness if testing TypeVars directly
    _GenericAlias,  # type: ignore
    get_origin,
    get_args,
)
import inspect
from unittest import mock  # For mock.patch.object, mock.call, MagicMock etc.
import sys

from agentlightning import config

# TypeVar as used in the original code
CliConfigurable = Any
_C = TypeVar("_C", bound=CliConfigurable)


# --- Helper classes for testing ---
class SimpleConfig:
    def __init__(self, name: str, value: int = 10):
        self.name = name
        self.value = value


class ComplexConfig:
    def __init__(
        self,
        text: str,
        is_active: bool,
        count: Optional[int] = None,
        items: List[str] = [],  # Default mutable is usually discouraged, but for testing CLI it's okay.
        maybe_text: Optional[str] = "default_text",
        optional_list: Optional[List[int]] = None,
        list_of_optionals: List[Optional[str]] = [],
        any_param: Any = None,
        untyped_param="untyped_default",  # Parameter without type hint but with a default
    ):
        self.text = text
        self.is_active = is_active
        self.count = count
        self.items = items
        self.maybe_text = maybe_text
        self.optional_list = optional_list
        self.list_of_optionals = list_of_optionals
        self.any_param = any_param
        self.untyped_param = untyped_param


class NoInitParamsConfig:
    def __init__(self):
        pass


class OnlySelfConfig:  # For completeness, though __init__ without params is similar
    def __init__(self):
        pass


class RequiredOnlyConfig:
    def __init__(self, req_str: str, req_int: int):
        self.req_str = req_str
        self.req_int = req_int


class OptionalNoDefaultConfig:
    def __init__(self, opt_val: Optional[str]):  # No __init__ default
        self.opt_val = opt_val


# --- Tests for nullable_str ---
@pytest.mark.parametrize(
    "input_val, expected_output",
    [
        ("none", None),
        ("None", None),
        ("NONE", None),
        ("null", None),
        ("Null", None),
        ("NULL", None),
        ("~", None),
        ("nil", None),
        ("Nil", None),
        ("NIL", None),
        ("actual_string", "actual_string"),
        ("", ""),
        (" ", " "),
        ("NoneValue", "NoneValue"),  # Should not convert if not exact keyword
    ],
)
def test_nullable_str(input_val, expected_output):
    """Tests the nullable_str function for various inputs."""
    assert config.nullable_str(input_val) == expected_output


# --- Tests for _str_to_bool ---
@pytest.mark.parametrize(
    "input_val, expected_output",
    [
        ("yes", True),
        ("Yes", True),
        ("YES", True),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("t", True),
        ("T", True),
        ("y", True),
        ("Y", True),
        ("1", True),
        (True, True),  # Direct bool pass-through
        ("no", False),
        ("No", False),
        ("NO", False),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("f", False),
        ("F", False),
        ("n", False),
        ("N", False),
        ("0", False),
        (False, False),  # Direct bool pass-through
    ],
)
def test_str_to_bool_valid(input_val, expected_output):
    """Tests _str_to_bool with valid boolean string representations."""
    assert config._str_to_bool(input_val) == expected_output


@pytest.mark.parametrize("invalid_input", ["maybe", "2", "", " ", "trueish", "falsey"])
def test_str_to_bool_invalid(invalid_input):
    """Tests _str_to_bool with invalid inputs, expecting ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError):
        config._str_to_bool(invalid_input)


# --- Tests for _get_param_type_details ---
@pytest.mark.parametrize(
    "annotation, expected_core_type, expected_is_optional, expected_is_list",
    [
        (str, str, False, False),
        (int, int, False, False),
        (bool, bool, False, False),
        (Any, Any, False, False),
        (Optional[str], str, True, False),
        (Union[str, None], str, True, False),  # Equivalent to Optional[str]
        (Optional[int], int, True, False),
        (List[str], List[str], False, True),  # core_type is List[str]
        (List[int], List[int], False, True),
        (Optional[List[str]], List[str], True, True),  # core_type is List[str], outer is Optional
        (Union[List[str], None], List[str], True, True),
        (List[Optional[str]], List[Optional[str]], False, True),  # core_type is List[Optional[str]]
        (Optional[List[Optional[int]]], List[Optional[int]], True, True),
        (inspect.Parameter.empty, inspect.Parameter.empty, False, False),  # No annotation
        (Union[int, str], Union[int, str], False, False),  # Non-optional Union
        (Tuple[str, int], Tuple[str, int], False, False),  # Tuple is not a List
        (Callable[[int], str], Callable[[int], str], False, False),
        (Dict[str, int], Dict[str, int], False, False),
        (List[Any], List[Any], False, True),
        (Optional[List[Any]], List[Any], True, True),
    ],
)
def test_get_param_type_details(annotation, expected_core_type, expected_is_optional, expected_is_list):
    """Tests _get_param_type_details for various type annotations."""
    core_type, is_optional, is_list = config._get_param_type_details(annotation)

    # Handle comparison for complex types like Union by comparing origin and args
    if get_origin(expected_core_type) is Union and get_origin(core_type) is Union:
        assert get_origin(core_type) == get_origin(expected_core_type)
        # Sort args for comparison as order doesn't matter in Union and get_args might not preserve input order
        assert sorted(map(str, get_args(core_type))) == sorted(map(str, get_args(expected_core_type)))
    else:
        assert core_type == expected_core_type
    assert is_optional == expected_is_optional
    assert is_list == expected_is_list


# --- Tests for _determine_argparse_type_and_nargs ---
@pytest.mark.parametrize(
    "core_param_type, is_param_list, expected_kwargs",
    [
        # List cases
        (List[str], True, {"nargs": "*", "type": str}),
        (List[int], True, {"nargs": "*", "type": int}),
        (List[float], True, {"nargs": "*", "type": float}),
        (List[bool], True, {"nargs": "*", "type": config._str_to_bool}),
        (List[Optional[str]], True, {"nargs": "*", "type": config.nullable_str}),
        (List[Any], True, {"nargs": "*", "type": str}),  # Defaults to str for List[Any] items
        # Non-list cases
        (str, False, {"type": str}),
        (int, False, {"type": int}),
        (float, False, {"type": float}),
        (bool, False, {"type": config._str_to_bool}),
        (Optional[str], False, {"type": config.nullable_str}),  # Handled by re-checking details
        (Union[str, None], False, {"type": config.nullable_str}),  # Equivalent to Optional[str]
        (Any, False, {"type": str}),  # No specific argparse type for Any
        (inspect.Parameter.empty, False, {"type": str}),  # No specific type for unannotated
        # List of complex types (e.g. List[Dict]), type of item is treated as string
        (List[Dict[str, int]], True, {"nargs": "*", "type": str}),
    ],
)
def test_determine_argparse_type_and_nargs(core_param_type, is_param_list, expected_kwargs):
    """Tests _determine_argparse_type_and_nargs for type and nargs mapping."""
    assert config._determine_argparse_type_and_nargs(core_param_type, is_param_list) == expected_kwargs


# --- Tests for _build_help_string ---
@pytest.mark.parametrize(
    "cls_name, param_name, core_type, is_optional, is_list, expected_help",
    [
        ("C", "p", str, False, False, "For C: 'p'. Inferred type: str."),
        ("C", "p", int, True, False, "For C: 'p'. Inferred type: Optional[int]."),
        ("C", "p_list", List[str], False, True, "For C: 'p_list'. Inferred type: List[str]."),
        ("C", "p_opt_list", List[int], True, True, "For C: 'p_opt_list'. Inferred type: Optional[List[int]]."),
        (
            "C",
            "p_list_opt",
            List[Optional[str]],
            False,
            True,
            "For C: 'p_list_opt'. Inferred type: List[Optional[str]].",
        ),
        ("C", "p_any", Any, False, False, "For C: 'p_any'. Inferred type: Any."),
        ("C", "p_opt_any", Any, True, False, "For C: 'p_opt_any'. Inferred type: Optional[Any]."),
        ("C", "p_list_any", List[Any], False, True, "For C: 'p_list_any'. Inferred type: List[Any]."),
        ("C", "p_empty", inspect.Parameter.empty, False, False, "For C: 'p_empty'. Inferred type: Any."),
    ],
)
def test_build_help_string(cls_name, param_name, core_type, is_optional, is_list, expected_help):
    """Tests _build_help_string for generating correct help messages."""
    assert config._build_help_string(cls_name, param_name, core_type, is_optional, is_list) == expected_help


# --- Tests for _add_argument_for_parameter ---
@pytest.fixture
def mock_parser():
    """Fixture to create a mock ArgumentParser."""
    parser = mock.MagicMock(spec=argparse.ArgumentParser)
    parser.add_argument = mock.MagicMock()
    return parser


def get_param_obj(cls, param_name):
    """Helper to get an inspect.Parameter object from a class's __init__."""
    sig = inspect.signature(cls.__init__)
    return sig.parameters[param_name]


@pytest.mark.parametrize(
    "param_name, cls, expected_cli_name_part, expected_argparse_kwargs_subset",
    [
        ("name", SimpleConfig, "simpleconfig.name", {"type": str, "required": True}),
        ("value", SimpleConfig, "simpleconfig.value", {"type": int, "default": 10}),
        ("text", ComplexConfig, "complexconfig.text", {"type": str, "required": True}),
        ("is_active", ComplexConfig, "complexconfig.is-active", {"type": config._str_to_bool, "required": True}),
        ("count", ComplexConfig, "complexconfig.count", {"type": int, "default": None}),  # Optional[int]=None
        ("items", ComplexConfig, "complexconfig.items", {"type": str, "nargs": "*", "default": []}),
        (
            "maybe_text",
            ComplexConfig,
            "complexconfig.maybe-text",
            {"type": config.nullable_str, "default": "default_text"},
        ),
        ("optional_list", ComplexConfig, "complexconfig.optional-list", {"type": int, "nargs": "*", "default": None}),
        (
            "list_of_optionals",
            ComplexConfig,
            "complexconfig.list-of-optionals",
            {"type": config.nullable_str, "nargs": "*", "default": []},
        ),
        (
            "any_param",
            ComplexConfig,
            "complexconfig.any-param",
            {"default": None},
        ),  # Type Any implies no specific argparse type
        (
            "untyped_param",
            ComplexConfig,
            "complexconfig.untyped-param",
            {"type": str, "default": "untyped_default"},
        ),  # No type hint
        (
            "opt_val",
            OptionalNoDefaultConfig,
            "optionalnodefaultconfig.opt-val",
            {"type": config.nullable_str, "default": None},
        ),
    ],
)
def test_add_argument_for_parameter(
    mock_parser, param_name, cls, expected_cli_name_part, expected_argparse_kwargs_subset
):
    """Tests _add_argument_for_parameter for correct argument configuration."""
    param_obj = get_param_obj(cls, param_name)
    dest_name = f"{cls.__name__.lower()}_{param_name}"
    config._add_argument_for_parameter(mock_parser, cls, param_name, param_obj, dest_name)

    # Build expected CLI argument name (e.g., --simpleconfig.name)
    expected_cli_arg_name = f"--{expected_cli_name_part.replace('_', '-')}"

    # Check that add_argument was called
    assert mock_parser.add_argument.called

    # Get the actual call arguments
    actual_call_args, actual_call_kwargs = mock_parser.add_argument.call_args
    assert actual_call_args[0] == expected_cli_arg_name
    assert actual_call_kwargs["dest"] == dest_name

    # Check for the presence and correctness of specified kwargs
    for key, expected_value in expected_argparse_kwargs_subset.items():
        assert key in actual_call_kwargs, f"Key '{key}' not in argparse kwargs for {param_name}"
        assert actual_call_kwargs[key] == expected_value, f"Value for key '{key}' incorrect for {param_name}"

    # Check 'required' status carefully
    if expected_argparse_kwargs_subset.get("required", False):
        assert actual_call_kwargs.get("required") is True
    else:  # Not required or default makes it not required
        assert not actual_call_kwargs.get("required", False)  # Either 'required' is False or not present

    # Check help string was generated
    assert "help" in actual_call_kwargs
    assert isinstance(actual_call_kwargs["help"], str)


# --- Tests for _add_arguments_for_class ---
def test_add_arguments_for_class(mock_parser):
    """Tests _add_arguments_for_class by checking calls to _add_argument_for_parameter."""
    class_arg_configs_maps = {}
    with mock.patch("agentlightning.config._add_argument_for_parameter") as mock_add_param_func:
        config._add_arguments_for_class(mock_parser, ComplexConfig, class_arg_configs_maps)

        assert ComplexConfig in class_arg_configs_maps
        init_params = inspect.signature(ComplexConfig.__init__).parameters
        expected_calls = 0
        for p_name, p_obj in init_params.items():
            if p_name == "self":
                continue
            expected_calls += 1
            dest = f"complexconfig_{p_name}"
            assert class_arg_configs_maps[ComplexConfig][p_name] == dest
            # Check if mock_add_param_func was called with these specific arguments
            found_call = any(
                call_args[0] == mock_parser
                and call_args[1] == ComplexConfig
                and call_args[2] == p_name
                and call_args[3] == p_obj
                and call_args[4] == dest
                for call_args, _ in mock_add_param_func.call_args_list
            )
            assert found_call, f"Expected call for {p_name} not found or with wrong arguments."
        assert mock_add_param_func.call_count == expected_calls


def test_add_arguments_for_class_no_init_params(mock_parser):
    """Tests _add_arguments_for_class with a class having no __init__ parameters."""
    class_arg_configs_maps = {}
    with mock.patch("agentlightning.config._add_argument_for_parameter") as mock_add_param_func:
        config._add_arguments_for_class(mock_parser, NoInitParamsConfig, class_arg_configs_maps)
        mock_add_param_func.assert_not_called()
        assert class_arg_configs_maps[NoInitParamsConfig] == {}


def test_create_argument_parser():
    """Tests _create_argument_parser for basic parser properties."""
    parser = config._create_argument_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.description == "CLI configurator for application components."
    # The formatter_class is a type, so compare types
    assert type(parser.formatter_class) == type(argparse.ArgumentDefaultsHelpFormatter)


def test_instantiate_classes():
    """Tests _instantiate_classes with various argument types and defaults."""
    parsed_args = argparse.Namespace(
        simpleconfig_name="TestName",
        simpleconfig_value=99,
        complexconfig_text="Hello",
        complexconfig_is_active=True,
        complexconfig_count=None,  # Explicitly None
        complexconfig_items=["item1", "item2"],
        # maybe_text not provided, should use its default "default_text" from __init__
        # Argparse default for complexconfig_maybe_text would be "default_text"
        complexconfig_maybe_text="default_text",  # Simulate argparse providing the default
        complexconfig_optional_list=[1, 2, 3],
        complexconfig_list_of_optionals=["a", None, "b"],
        complexconfig_any_param=object(),
        complexconfig_untyped_param="custom_untyped",
    )
    # This simulates that 'complexconfig_maybe_text' was given a default by add_argument
    # and thus present in parsed_args.
    # If not provided on CLI and having __init__ default, argparse gives it default.

    class_arg_configs_maps = {
        SimpleConfig: {"name": "simpleconfig_name", "value": "simpleconfig_value"},
        ComplexConfig: {
            p: f"complexconfig_{p}" for p in inspect.signature(ComplexConfig.__init__).parameters if p != "self"
        },
    }

    instances = config._instantiate_classes(parsed_args, (SimpleConfig, ComplexConfig), class_arg_configs_maps)

    assert len(instances) == 2
    sc, cc = instances

    assert isinstance(sc, SimpleConfig)
    assert sc.name == "TestName"
    assert sc.value == 99

    assert isinstance(cc, ComplexConfig)
    assert cc.text == "Hello"
    assert cc.is_active is True
    assert cc.count is None
    assert cc.items == ["item1", "item2"]
    assert cc.maybe_text == "default_text"  # From __init__ via argparse default
    assert cc.optional_list == [1, 2, 3]
    assert cc.list_of_optionals == ["a", None, "b"]
    assert cc.any_param is parsed_args.complexconfig_any_param
    assert cc.untyped_param == "custom_untyped"


def test_instantiate_classes_error_handling(caplog):
    """Tests error logging during class instantiation failure."""

    class FailingConfig:
        def __init__(self, param):
            raise ValueError("Instantiaion failed")

    parsed_args = argparse.Namespace(failingconfig_param="value")
    class_arg_configs_maps = {FailingConfig: {"param": "failingconfig_param"}}

    with pytest.raises(ValueError, match="Instantiaion failed"):
        config._instantiate_classes(parsed_args, (FailingConfig,), class_arg_configs_maps)

    assert "Error instantiating FailingConfig" in caplog.text
    assert "{'param': 'value'}" in caplog.text  # constructor_args
    assert "Error: Instantiaion failed" in caplog.text


# --- Integration Tests for lightning_cli ---
def run_lightning_cli(classes_to_configure, cli_args_list):
    """Helper to run lightning_cli with mocked sys.argv."""
    # Prepend a dummy program name to cli_args_list for sys.argv
    with mock.patch.object(sys, "argv", ["test_program.py"] + cli_args_list):
        result = config.lightning_cli(*classes_to_configure)
        if not isinstance(result, tuple):
            return (result,)
        return result


def test_lightning_cli_no_classes():
    """Tests lightning_cli with no classes provided."""
    assert run_lightning_cli([], []) == tuple()


def test_lightning_cli_simple_config():
    """Tests lightning_cli with a simple class and various argument scenarios."""
    # Only required arg
    (sc1,) = run_lightning_cli([SimpleConfig], ["--simpleconfig.name", "MyName"])
    assert sc1.name == "MyName"
    assert sc1.value == 10  # Default from __init__

    # All args
    (sc2,) = run_lightning_cli([SimpleConfig], ["--simpleconfig.name", "Another", "--simpleconfig.value", "77"])
    assert sc2.name == "Another"
    assert sc2.value == 77


def test_lightning_cli_complex_config_types():
    """Tests lightning_cli with ComplexConfig, checking various type conversions."""
    cli_args = [
        "--complexconfig.text",
        "CLI Text",
        "--complexconfig.is-active",
        "yes",
        "--complexconfig.count",
        "123",
        "--complexconfig.items",
        "apple",
        "banana",
        "--complexconfig.maybe-text",
        "None",  # Test nullable_str
        "--complexconfig.optional-list",
        "10",
        "20",
        "30",
        "--complexconfig.list-of-optionals",
        "first",
        "nil",
        "third",
        "null",
        "--complexconfig.any-param",
        "AnythingGoes",
        "--complexconfig.untyped-param",
        "FromCLI",
    ]
    (cc,) = run_lightning_cli([ComplexConfig], cli_args)

    assert cc.text == "CLI Text"
    assert cc.is_active is True
    assert cc.count == 123
    assert cc.items == ["apple", "banana"]
    assert cc.maybe_text is None
    assert cc.optional_list == [10, 20, 30]
    assert cc.list_of_optionals == ["first", None, "third", None]
    assert cc.any_param == "AnythingGoes"
    assert cc.untyped_param == "FromCLI"


def test_lightning_cli_complex_config_defaults():
    """Tests that __init__ defaults are used if CLI args are not provided."""
    # Only provide required arguments for ComplexConfig
    cli_args = ["--complexconfig.text", "Minimal", "--complexconfig.is-active", "f"]
    (cc,) = run_lightning_cli([ComplexConfig], cli_args)

    assert cc.text == "Minimal"
    assert cc.is_active is False
    assert cc.count is None  # Default from __init__
    assert cc.items == []  # Default from __init__
    assert cc.maybe_text == "default_text"  # Default from __init__
    assert cc.optional_list is None  # Default from __init__
    assert cc.list_of_optionals == []  # Default from __init__
    assert cc.any_param is None  # Default from __init__
    assert cc.untyped_param == "untyped_default"  # Default from __init__


def test_lightning_cli_multiple_classes():
    """Tests configuring multiple classes simultaneously."""
    cli_args = [
        "--simpleconfig.name",
        "SC_Multi",
        "--simpleconfig.value",
        "5",
        "--complexconfig.text",
        "CC_Multi",
        "--complexconfig.is-active",
        "true",
        "--complexconfig.maybe-text",
        "NotNone",
    ]
    sc, cc = run_lightning_cli([SimpleConfig, ComplexConfig], cli_args)

    assert isinstance(sc, SimpleConfig)
    assert sc.name == "SC_Multi"
    assert sc.value == 5

    assert isinstance(cc, ComplexConfig)
    assert cc.text == "CC_Multi"
    assert cc.is_active is True
    assert cc.maybe_text == "NotNone"
    assert cc.items == []  # Default


def test_lightning_cli_missing_required_arg_exits(capsys):
    """Tests that argparse exits if a required argument is missing."""
    with pytest.raises(SystemExit) as e:
        run_lightning_cli([RequiredOnlyConfig], ["--requiredonlyconfig.req-int", "123"])
    assert e.value.code != 0  # Argparse exits with non-zero for error
    captured = capsys.readouterr()  # Capture stderr
    assert "the following arguments are required: --requiredonlyconfig.req-str" in captured.err


def test_lightning_cli_optional_no_default_behavior():
    """Tests Optional parameter without __init__ default."""
    # Not provided: should become None
    (cfg1,) = run_lightning_cli([OptionalNoDefaultConfig], [])
    assert cfg1.opt_val is None

    # Provided as "None" via nullable_str
    (cfg2,) = run_lightning_cli([OptionalNoDefaultConfig], ["--optionalnodefaultconfig.opt-val", "None"])
    assert cfg2.opt_val is None

    # Provided with a value
    (cfg3,) = run_lightning_cli([OptionalNoDefaultConfig], ["--optionalnodefaultconfig.opt-val", "ActualValue"])
    assert cfg3.opt_val == "ActualValue"
