"""Provides an event system to dispatch events to various handlers."""
from __future__ import annotations

# Builtin
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = (
    "EA",
    "EventError",
    "add_event_handler",
    "dispatch_event",
    "get_handlers_for_event_name",
)

# Define a generic type for the event arguments
EA = TypeVar("EA")

# Store all the event handlers
_event_handlers: defaultdict[str, set[Callable[[dict[str, EA]], None]]] = defaultdict(
    set
)


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


def add_event_handler(
    event_name: str = "",
) -> Callable[[Callable[[dict[str, EA]], None]], Callable[[dict[str, EA]], None]]:
    """Add an event handler to the event system.

    This event must be a global function (or a static method that does not access the
    class's state) due to the event system not supporting bound methods.

    Parameters
    ----------
    event_name: str
        The event name.

    Examples
    --------
    An event handler can be easily added using this decorator with the event name as the
    function name. For example, an event handler which uses the function name:

    >>> @add_event_handler()
    ... def event_function_name(event_args: dict[str, EA]) -> None:
    ...     pass

    Or an event handler that uses the parameter variable:

    >>> @add_event_handler("event_parameter_name")
    ... def event_example(event_args: dict[str, EA]) -> None:
    ...     pass

    Returns
    -------
    Callable[[Callable[[dict[str, EA]], None]], Callable[[dict[str, EA]], None]]
        The event handler now added to the event system.
    """

    def wrapper(
        func: Callable[[dict[str, EA]], None]
    ) -> Callable[[dict[str, EA]], None]:
        _event_handlers[event_name if event_name else func.__name__].add(func)
        return func

    return wrapper


def dispatch_event(event_name: str, event_args: dict[str, EA]) -> None:
    """Dispatch an event with keyword arguments.

    Parameters
    ----------
    event_name: str
        The event name to dispatch an event for.
    event_args: dict[str, EA]
        The event arguments to pass through to the event.

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
        handler(event_args)


def get_handlers_for_event_name(
    event_name: str,
) -> set[Callable[[dict[str, EA]], None]]:
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
    set[Callable[[dict[str, EA]], None]]
        The event's handlers.
    """
    # Check if the event is registered or not
    if event_name not in _event_handlers:
        raise EventError(event_name=event_name)

    # Return the event's handlers
    return _event_handlers[event_name]
