# Copyright (c) Microsoft. All rights reserved.

import pytest

from agentlightning.trainer.init_utils import build_component, instantiate_component, instantiate_from_spec, load_class

from .sample_components import (
    ComponentWithKwargs,
    ComponentWithoutOptional,
    SampleBase,
    SampleComponent,
    SimpleComponent,
    Unrelated,
)


def test_load_class_resolves_fully_qualified_name() -> None:
    resolved = load_class("tests.trainer.sample_components.SampleComponent")
    assert resolved is SampleComponent


def test_instantiate_component_uses_provided_kwargs() -> None:
    component = instantiate_component(SampleComponent, {"required": 3})

    assert isinstance(component, SampleComponent)
    assert component.required == 3
    assert component.optional == 7


def test_instantiate_component_applies_optional_defaults_callable() -> None:
    component = instantiate_component(
        SampleComponent,
        {"required": 2},
        {"optional": lambda: 11},
    )

    assert component.optional == 11


def test_instantiate_component_ignores_unknown_optional_defaults() -> None:
    component = instantiate_component(
        ComponentWithoutOptional,
        {"value": 5},
        {"missing": lambda: pytest.fail("should not be called")},
    )

    assert isinstance(component, ComponentWithoutOptional)
    assert component.value == 5


def test_instantiate_from_spec_with_string_path() -> None:
    instance = instantiate_from_spec("tests.trainer.sample_components.SimpleComponent", spec_name="component")

    assert isinstance(instance, SimpleComponent)
    assert instance.created


def test_instantiate_from_spec_with_dict_type() -> None:
    instance = instantiate_from_spec(
        {"type": "tests.trainer.sample_components.SampleComponent", "required": 4},
        spec_name="component",
    )

    assert isinstance(instance, SampleComponent)
    assert instance.required == 4


def test_instantiate_from_spec_missing_type_uses_default_cls() -> None:
    instance = instantiate_from_spec(
        {"required": 8},
        spec_name="adapter",
        dict_requires_type=False,
        dict_default_cls=SampleComponent,
    )

    assert isinstance(instance, SampleComponent)
    assert instance.required == 8


def test_instantiate_from_spec_missing_type_raises_when_required() -> None:
    with pytest.raises(ValueError, match="component dict must have a 'type' key"):
        instantiate_from_spec({}, spec_name="component")


def test_build_component_returns_existing_instance() -> None:
    existing = SampleComponent(required=3)
    result = build_component(
        existing,
        expected_type=SampleComponent,
        spec_name="component",
    )

    assert result is existing


def test_build_component_uses_default_factory_for_none() -> None:
    result = build_component(
        None,
        expected_type=SampleComponent,
        spec_name="component",
        default_factory=lambda: SampleComponent(required=5),
    )

    assert isinstance(result, SampleComponent)
    assert result.required == 5


def test_build_component_returns_none_when_allowed() -> None:
    assert (
        build_component(
            None,
            expected_type=SampleComponent,
            spec_name="component",
            allow_none=True,
        )
        is None
    )


def test_build_component_from_string_spec() -> None:
    result = build_component(
        "tests.trainer.sample_components.SimpleComponent",
        expected_type=SampleBase,
        spec_name="component",
    )

    assert isinstance(result, SimpleComponent)


def test_build_component_from_registry_key_string_spec() -> None:
    result = build_component(
        "simple",
        expected_type=SampleBase,
        spec_name="component",
        registry={"simple": "tests.trainer.sample_components.SimpleComponent"},
    )

    assert isinstance(result, SimpleComponent)


def test_build_component_from_dict_spec_with_optional_defaults() -> None:
    result = build_component(
        {"type": "tests.trainer.sample_components.SampleComponent", "required": 10},
        expected_type=SampleComponent,
        spec_name="component",
        optional_defaults={"optional": lambda: 42},
    )

    assert result is not None
    assert result.optional == 42


def test_build_component_from_type_spec() -> None:
    result = build_component(
        SimpleComponent,
        expected_type=SampleBase,
        spec_name="component",
    )

    assert isinstance(result, SimpleComponent)


def test_build_component_from_callable_spec() -> None:
    result = build_component(
        lambda: SampleComponent(required=12),
        expected_type=SampleComponent,
        spec_name="component",
    )

    assert isinstance(result, SampleComponent)
    assert result.required == 12


def test_build_component_invalid_type_error() -> None:
    with pytest.raises(
        ValueError, match="Invalid component type: <class 'int'>. Expected SampleComponent, str, dict, or None."
    ):
        build_component(
            1,
            expected_type=SampleComponent,
            spec_name="component",
            invalid_spec_error_fmt="Invalid component type: {actual_type}. Expected SampleComponent, str, dict, or None.",
        )


def test_build_component_type_error_fmt() -> None:
    with pytest.raises(TypeError, match="Custom type error message"):
        build_component(
            "tests.trainer.sample_components.SimpleComponent",
            expected_type=ComponentWithoutOptional,
            spec_name="component",
            type_error_fmt="Custom type error message",
        )


def test_instantiate_component_applies_literal_optional_default() -> None:
    component = instantiate_component(SampleComponent, {"required": 9}, {"optional": 13})

    assert component.optional == 13


def test_instantiate_from_spec_invalid_type_error_message() -> None:
    with pytest.raises(TypeError, match="component spec must be a string or dict"):
        instantiate_from_spec(42, spec_name="component")  # type: ignore[arg-type]


def test_build_component_from_string_type_mismatch() -> None:
    with pytest.raises(
        TypeError,
        match="component factory returned <class 'tests.trainer.sample_components.SimpleComponent'>, which is not a SampleComponent subclass.",
    ):
        build_component(
            "tests.trainer.sample_components.SimpleComponent",
            expected_type=SampleComponent,
            spec_name="component",
            invalid_spec_error_fmt="Invalid component type: {actual_type}. Expected SampleComponent, str, dict, or None.",
        )


def test_build_component_none_without_factory_raises_default_message() -> None:
    with pytest.raises(ValueError, match="component cannot be None."):
        build_component(
            None,
            expected_type=SampleComponent,
            spec_name="component",
        )


def test_build_component_with_type_not_subclass_raises() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported component type <class 'type'> for component",
    ):
        build_component(
            Unrelated,
            expected_type=SampleComponent,
            spec_name="component",
            invalid_spec_error_fmt="Unsupported component type {actual_type} for component",
        )


def test_build_component_callable_returning_wrong_type_raises() -> None:
    with pytest.raises(
        TypeError,
        match="component factory returned <class 'tests.trainer.sample_components.Unrelated'>, which is not a SampleComponent subclass.",
    ):
        build_component(
            lambda: Unrelated(),
            expected_type=SampleComponent,
            spec_name="component",
        )


def test_build_component_optional_defaults_skipped_when_not_supported() -> None:
    component = build_component(
        {"type": "tests.trainer.sample_components.ComponentWithKwargs"},
        expected_type=ComponentWithKwargs,
        spec_name="component",
        optional_defaults={"missing": lambda: pytest.fail("should not be invoked")},
    )

    assert isinstance(component, ComponentWithKwargs)
    assert component.kwargs == {}


def test_build_component_dict_spec_uses_registry_type_lookup() -> None:
    component = build_component(
        {"type": "simple"},
        expected_type=SampleBase,
        spec_name="component",
        registry={"simple": "tests.trainer.sample_components.SimpleComponent"},
    )

    assert isinstance(component, SimpleComponent)


def test_build_component_dict_spec_supports_name_registry_lookup() -> None:
    component = build_component(
        {"name": "simple"},
        expected_type=SampleBase,
        spec_name="component",
        dict_requires_type=False,
        registry={"simple": "tests.trainer.sample_components.SimpleComponent"},
    )

    assert isinstance(component, SimpleComponent)
