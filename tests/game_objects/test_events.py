"""Tests all functions in game_objects/events.py."""
from __future__ import annotations

# Builtin
import re
from typing import TypeVar

# Pip
import pytest

# Custom
from hades.game_objects.events import (
    EA,
    EventError,
    add_event_handler,
    dispatch_event,
    get_handlers_for_event_name,
)

__all__ = ()


# Define a generic type for the kwargs
T = TypeVar("T")


@add_event_handler()
def on_no_kwargs(_: dict[str, EA]) -> None:
    """Simulate a non-kwarg event for testing."""
    assert True


@add_event_handler()
def on_kwargs(event_args: dict[str, EA]) -> None:
    """Simulate a kwarg event for testing.

    Parameters
    ----------
    event_args: dict[str, EA]
        The testing event arguments.
    """
    assert event_args["test"] == "test"


@add_event_handler()
def on_expected_kwarg(event_args: dict[str, EA]) -> None:
    """Simulate a kwarg event for testing.

    Parameters
    ----------
    event_args: dict[str, EA]
        The testing event arguments.
    """
    with pytest.raises(expected_exception=KeyError):
        assert event_args["test"] == "test"


@add_event_handler()
def on_return_value(event_args: dict[str, EA]) -> None:
    """Simulate an event which returns a value for testing.

    Parameters
    ----------
    event_args: dict[str, EA]
        The testing event arguments.
    """
    event_args["test"] = "test"


@add_event_handler("on_multiple_handlers")
def on_multiple_handlers_one(_: dict[str, EA]) -> None:
    """Simulate an event that is dispatched to multiple handlers."""
    assert True


@add_event_handler("on_multiple_handlers")
def on_multiple_handlers_two(_: dict[str, EA]) -> None:
    """Simulate an event that is dispatched to multiple handlers."""
    assert True


class ClassStaticEventHandler:
    """Simulates an event which does not take in a self argument."""

    @staticmethod
    @add_event_handler()
    def on_class_static(_: dict[str, EA]) -> None:
        """Simulate an event registered to a staticmethod."""
        assert True


class ClassEventHandler:
    """Simulates an event which will raise an error when dispatched."""

    @add_event_handler()
    def on_class_state(self: ClassEventHandler, _: dict[str, EA]) -> None:
        """Simulate an event bound to a class which fails."""


def test_raise_event_error() -> None:
    """Test that EventError is raised correctly."""
    with pytest.raises(
        expected_exception=EventError,
        match="The event `test` is not registered.",
    ):
        raise EventError(event_name="test")


def test_dispatch_event_no_kwargs() -> None:
    """Test dispatching an event with no keyword arguments is successful."""
    dispatch_event("on_no_kwargs", {})


def test_dispatch_event_kwargs() -> None:
    """Test dispatching an event with keyword arguments is successful."""
    dispatch_event("on_kwargs", {"test": "test"})


def test_dispatch_event_return() -> None:
    """Test dispatching an event which returns a value is successful."""
    result = {}
    dispatch_event("on_return_value", result)
    assert result["test"] == "test"


def test_dispatch_event_multiple_handlers() -> None:
    """Test dispatching an event to multiple handlers is successful."""
    dispatch_event("on_multiple_handlers", {})


def test_dispatch_event_staticmethod_event() -> None:
    """Test dispatching an event to a staticmethod handler is successful."""
    dispatch_event("on_class_static", {})


def test_dispatch_event_unregistered_event() -> None:
    """Test dispatching an unregistered event raises an error."""
    with pytest.raises(
        expected_exception=EventError,
        match="The event `test` is not registered.",
    ):
        dispatch_event("test", {})


def test_dispatch_event_class_event() -> None:
    """Test dispatching an event to a class handler."""
    with pytest.raises(
        expected_exception=TypeError,
        match=re.escape(
            "ClassEventHandler.on_class_state() missing 1 required positional"
            " argument: '_'"
        ),
    ):
        dispatch_event("on_class_state", {})


def test_get_handlers_for_event_name_registered_event_name() -> None:
    """Test that getting handlers for a registered event name is successful."""
    assert get_handlers_for_event_name("on_no_kwargs") == {on_no_kwargs}


def test_get_handlers_for_event_name_unregistered_event_name() -> None:
    """Test that getting handlers for an unregistered event name is not successful."""
    with pytest.raises(
        expected_exception=EventError,
        match="The event `test` is not registered.",
    ):
        get_handlers_for_event_name("test")
