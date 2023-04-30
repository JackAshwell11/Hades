"""Tests all functions in game_objects/events.py."""
from __future__ import annotations

# Builtin
import re

# Pip
import pytest

# Custom
from hades.game_objects.events import (
    RA,
    EventError,
    add_event_handler,
    dispatch_event,
    get_handlers_for_event_name,
)

__all__ = ()


@add_event_handler()
def on_basic_event() -> None:
    """Simulate a basic event for testing."""
    assert True


@add_event_handler()
def on_arguments(test: str) -> None:
    """Simulate an event with arguments for testing.

    Parameters
    ----------
    test: str
        A testing argument.
    """
    assert test == "test"


@add_event_handler(event_name="on_event_parameter")
def event_parameter() -> None:
    """Simulate an event created using the parameter name."""
    assert True


@add_event_handler(return_value=True)
def on_return_value(return_args: dict[str, RA]) -> None:
    """Simulate an event which returns a value for testing.

    Parameters
    ----------
    return_args: dict[str, RA]
        The testing return arguments.
    """
    return_args["test"] = "test"


@add_event_handler()
def on_return_value_no_bool(return_args: dict[str, RA]) -> None:
    """Simulate an event which should return a value but does not initialise it as so.

    Parameters
    ----------
    return_args: dict[str, RA]
        The testing return arguments.
    """
    return_args["test"] = "test"


@add_event_handler(return_value=True)
def on_return_value_no_parameter() -> None:
    """Simulate an event which does not provide a parameter for return_args."""


@add_event_handler()
def on_unexpected_return_arg(return_args: dict[str, RA]) -> None:
    """Simulate an event which unexpectedly provides a return_args parameter.

    Parameters
    ----------
    return_args: dict[str, RA]
        The testing return arguments.
    """
    return_args["test"] = "test"


@add_event_handler(event_name="on_multiple_handlers")
def on_multiple_handlers_one() -> None:
    """Simulate an event that is dispatched to multiple handlers."""
    assert True


@add_event_handler(event_name="on_multiple_handlers")
def on_multiple_handlers_two() -> None:
    """Simulate an event that is dispatched to multiple handlers."""
    assert True


class ClassStaticEventHandler:
    """Simulates an event which does not take in a self argument."""

    @staticmethod
    @add_event_handler()
    def on_class_static() -> None:
        """Simulate an event registered to a staticmethod."""
        assert True


class ClassEventHandler:
    """Simulates an event which will raise an error when dispatched."""

    @add_event_handler()
    def on_class_state(self: ClassEventHandler) -> None:
        """Simulate an event bound to a class which fails."""


def test_raise_event_error() -> None:
    """Test that EventError is raised correctly."""
    with pytest.raises(
        expected_exception=EventError,
        match="The event `test` is not registered.",
    ):
        raise EventError(event_name="test", error="is not registered")


def test_dispatch_event_basic() -> None:
    """Test dispatching a basic event is successful."""
    dispatch_event("on_basic_event")


def test_dispatch_event_arguments() -> None:
    """Test dispatching an event with arguments is successful."""
    dispatch_event("on_arguments", event_args={"test": "test"})


def test_dispatch_event_arguments_not_provided() -> None:
    """Test dispatching an event without the required arguments is not successful."""
    with pytest.raises(expected_exception=TypeError):
        dispatch_event("on_arguments")


def test_dispatch_event_arguments_wrong_type() -> None:
    """Test dispatching an event with invalid required arguments is not successful."""
    with pytest.raises(expected_exception=TypeError):
        dispatch_event("on_arguments", "test")  # type: ignore[arg-type]


def test_dispatch_event_parameter() -> None:
    """Test dispatching a parameter event is successful."""
    dispatch_event("on_event_parameter")


def test_dispatch_event_return() -> None:
    """Test dispatching an event which returns a value is successful."""
    result = {}
    dispatch_event("on_return_value", return_args=result)
    assert result["test"] == "test"


def test_dispatch_event_return_no_args() -> None:
    """Test dispatching an event without a return_args parameter when it is expected."""
    with pytest.raises(
        expected_exception=EventError,
        match="The event `on_return_value` expects a return args dictionary.",
    ):
        dispatch_event("on_return_value")


def test_dispatch_event_return_no_bool() -> None:
    """Test dispatching an event which does not initialise return_value."""
    with pytest.raises(expected_exception=TypeError):
        dispatch_event("on_return_value_no_bool", return_args={})


def test_dispatch_event_return_no_parameter() -> None:
    """Test dispatching an event which does not provide a parameter for return_args."""
    with pytest.raises(expected_exception=TypeError):
        dispatch_event("on_return_value_no_parameter", return_args={})


def test_dispatch_event_unexpected_return_arg() -> None:
    """Test dispatching an event which unexpectedly provides a return_args parameter."""
    with pytest.raises(expected_exception=TypeError):
        dispatch_event("on_unexpected_return_arg", return_args={})


def test_dispatch_event_multiple_handlers() -> None:
    """Test dispatching an event to multiple handlers is successful."""
    dispatch_event("on_multiple_handlers")


def test_dispatch_event_staticmethod_event() -> None:
    """Test dispatching an event to a staticmethod handler is successful."""
    dispatch_event("on_class_static")


def test_dispatch_event_unregistered_event() -> None:
    """Test dispatching an unregistered event raises an error."""
    with pytest.raises(
        expected_exception=EventError,
        match="The event `test` is not registered.",
    ):
        dispatch_event("test")


def test_dispatch_event_class_event() -> None:
    """Test dispatching an event to a class handler."""
    with pytest.raises(
        expected_exception=TypeError,
        match=re.escape(
            (
                "ClassEventHandler.on_class_state() missing 1 required positional"
                " argument: 'self'"
            ),
        ),
    ):
        dispatch_event("on_class_state")


def test_get_handlers_for_event_name_registered_event_name() -> None:
    """Test that getting handlers for a registered event name is successful."""
    assert get_handlers_for_event_name("on_basic_event") == {(on_basic_event, False)}


def test_get_handlers_for_event_name_unregistered_event_name() -> None:
    """Test that getting handlers for an unregistered event name is not successful."""
    with pytest.raises(
        expected_exception=EventError,
        match="The event `test` is not registered.",
    ):
        get_handlers_for_event_name("test")
