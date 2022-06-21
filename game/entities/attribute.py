from __future__ import annotations


class EntityAttribute:
    """"""

    __slots__ = ("_value",)

    def __init__(self, value) -> None:
        self._value = value

    @property
    def value(self):
        return self._value


class UpgradableAttribute(EntityAttribute):
    pass


class StatusEffectAttribute(EntityAttribute):
    pass


class VariableAttribute(EntityAttribute):
    pass
