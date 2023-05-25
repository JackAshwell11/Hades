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
    "RA",
    "add_event_handler",
    "dispatch_event",
    "get_handlers_for_event_name",
)

# Define a generic type for the return and event arguments
RA = TypeVar("RA")
EA = TypeVar("EA")

# Store all the event handlers
_event_handlers: defaultdict[
    str,
    set[tuple[Callable[[dict[str, RA] | None, dict[str, EA] | None], None], bool]],
] = defaultdict(
    set,
)


class EventError(Exception):
    """Raised when an event is not registered or encounters a problem."""

    def __init__(self: EventError, *, event_name: str, error: str) -> None:
        """Initialise the object.

        Args:
            event_name: The event name that is not registered.
            error: The error related to the event.
        """
        super().__init__(f"The event `{event_name}` {error}.")


def add_event_handler(
    *,
    event_name: str = "",
    return_value: bool = False,
) -> Callable[
    [Callable[[dict[str, RA] | None, dict[str, EA] | None], None]],
    Callable[[dict[str, RA] | None, dict[str, EA] | None], None],
]:
    """Add an event handler to the event system.

    This event must be a global function (or a static method that does not access the
    class's state) due to the event system not supporting bound methods.

    Args:
        event_name: The event name.
        return_value: Whether the event returns values or not.

    Returns:
        The event handler now added to the event system.

    Examples:
        An event handler can be easily added using this decorator with the event name as
        the function name. For example, an event handler which uses the function name
        and takes no parameters:

        >>> @add_event_handler()
        ... def event_function_name(**_) -> None:
        ...     pass

        Or an event handler that uses the parameter variable and takes a single
        argument:

        >>> @add_event_handler(event_name="event_parameter_name")
        ... def event_parameter_example(arg: int, **_) -> None:
        ...     pass

        Or another event handler that uses the function name but also returns a value:

        >>> @add_event_handler(return_value=True)
        ... def event_return(return_args: dict[str, RA], **_) -> None:
        ...     pass

        Or even an event handler that uses the parameter variable, takes multiple
        arguments, and returns a value:

        >>> @add_event_handler(event_name="event_all", return_value=True)
        ... def event_all_example(return_args: dict[str, RA], arg: int, **_) -> None:
        ...     pass
    """

    def wrapper(
        func: Callable[[dict[str, RA] | None, dict[str, EA] | None], None],
    ) -> Callable[[dict[str, RA] | None, dict[str, EA] | None], None]:
        _event_handlers[event_name if event_name else func.__name__].add(
            (func, return_value),
        )
        return func

    return wrapper


def dispatch_event(
    event_name: str,
    *,
    event_args: dict[str, EA] | None = None,
    return_args: dict[str, RA] | None = None,
) -> None:
    """Dispatch an event with keyword arguments.

    Args:
        event_name: The event name to dispatch an event for.
        event_args: The event arguments to pass through to the event.
        return_args: The return arguments to allow the event to return arguments.

    Raises:
        EventError: The event is not registered or expects a return args dictionary.
    """
    # Check if the event is registered or not
    if event_name not in _event_handlers:
        raise EventError(event_name=event_name, error="is not registered")

    # Dispatch the event to all handlers
    event_args = event_args if event_args else {}
    for handler, do_return in _event_handlers[event_name]:
        if do_return:
            if return_args is None:
                raise EventError(
                    event_name=event_name,
                    error="expects a return args dictionary",
                )
            handler(return_args, **event_args)  # type: ignore[call-arg]
        else:
            handler(**event_args)  # type: ignore[call-arg]


def get_handlers_for_event_name(
    event_name: str,
) -> set[tuple[Callable[[dict[str, RA] | None, dict[str, EA] | None], None], bool]]:
    """Get an event's handlers.

    Args:
        event_name: The event name to get handlers for.

    Returns:
        The event's handlers.

    Raises:
        EventError: The event is not registered.
    """
    # Check if the event is registered or not
    if event_name not in _event_handlers:
        raise EventError(event_name=event_name, error="is not registered")

    # Return the event's handlers
    return _event_handlers[event_name]
