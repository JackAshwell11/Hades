"""Provides an event system to dispatch events to various handlers."""
from __future__ import annotations

# Builtin
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = (
    "EventError",
    "add_event_handler",
    "dispatch_event",
    "get_handlers_for_event_name",
)


# Define some generic types for the keyword arguments
KW = TypeVar("KW")
R = TypeVar("R")


# Store all the event handlers
_event_handlers: defaultdict[str, set[Callable[[KW], R]]] = defaultdict(set)


class EventError(Exception):
    """Raised when an event is not registered."""

    def __init__(self: EventError, *, event_name: str) -> None:
        """Initialise the object.

        Parameters
        ----------
        event_name: str
            The event name that is not registered.
        """
        super().__init__(f"The event `{event_name}` is not registered.")


def add_event_handler(event_name: str) -> Callable[[KW], R]:
    """Add an event handler to the event system.

    Parameters
    ----------
    event_name: str
        The event name.

    Returns
    -------
    Callable[[KW], R]
        The event handler now added to the event system.
    """

    def wrapper(func: Callable[[KW], R]) -> Callable[[KW], R]:
        _event_handlers[event_name].add(func)
        return func

    return wrapper


def dispatch_event(event_name: str, **kwargs: KW) -> None:
    """Dispatch an event with keyword arguments.

    Parameters
    ----------
    event_name: str
        The event name to dispatch an event for.

    Raises
    ------
    EventError
        The event is not registered.
    """
    # Check if the event is registered or not
    if event_name not in _event_handlers:
        raise EventError(event_name=event_name)

    # Dispatch the event to all handlers
    for handler in _event_handlers[event_name]:
        handler(**kwargs)


def get_handlers_for_event_name(event_name: str) -> set[Callable[[KW], R]]:
    """Get an event's handlers.

    Parameters
    ----------
    event_name: str
        The event name to get handlers for.

    Raises
    ------
    EventError
        The event is not registered.

    Returns
    -------
    set[Callable[[KW], R]]
        The event's handlers.
    """
    # Check if the event is registered or not
    if event_name not in _event_handlers:
        raise EventError(event_name=event_name)

    # Return the event's handlers
    return _event_handlers[event_name]


# TODO: Dispatch_event could possibly return results (need to see if good idea) by
#  returning collection containing result for each handler (could sort by component name
#  to remove Nones). It could also become a global function and global private variable

# TODO: Figure out how to remove events
