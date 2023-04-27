"""Tests all functions in game_objects/events.py."""
from __future__ import annotations

# Builtin
from typing import TypeVar

# Pip
import pytest

# Custom
from hades.game_objects.events import (
    EventError,
    add_event_handler,
    dispatch_event,
    get_handlers_for_event_name,
)

__all__ = ()


# Define a generic type for the kwargs
T = TypeVar("T")


@add_event_handler("on_no_kwargs")
def on_no_kwargs() -> None:
    """Simulate a non-kwarg event for testing."""
    assert True


@add_event_handler("on_kwargs")
def on_kwargs(x: str, **_: T) -> None:
    """Simulate a kwarg event for testing.

    Parameters
    ----------
    x: str
        A testing variable.
    """
    assert x == "test"


@add_event_handler("on_multiple_handlers")
def on_multiple_handlers_one() -> None:
    """Simulate an event that is dispatched to multiple handlers."""
    assert True


@add_event_handler("on_multiple_handlers")
def on_multiple_handlers_two() -> None:
    """Simulate an event that is dispatched to multiple handlers."""
    assert True


def test_raise_event_error() -> None:
    """Test that EventError is raised correctly."""
    with pytest.raises(
        expected_exception=EventError,
        match="The event `test` is not registered.",
    ):
        raise EventError(event_name="test")


def test_dispatch_event_no_kwargs() -> None:
    """Test dispatching an event with no keyword arguments is successful."""
    dispatch_event("on_no_kwargs")


def test_dispatch_event_kwargs() -> None:
    """Test dispatching an event with keyword arguments is successful."""
    dispatch_event("on_kwargs", x="test")


def test_dispatch_event_multiple_handlers() -> None:
    """Test dispatching an event to multiple handlers is successful."""
    dispatch_event("on_multiple_handlers")


def test_dispatch_event_unregistered_event() -> None:
    """Test dispatching an unregistered event raises an error."""
    with pytest.raises(
        expected_exception=EventError,
        match="The event `test` is not registered.",
    ):
        dispatch_event("test")


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
