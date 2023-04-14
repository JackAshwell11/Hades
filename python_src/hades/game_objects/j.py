#class InstantEffect(GameObject):
class InstantEffect:
    def apply(self, target): # GameObject):
        match self.type:
            case health:
                target.health.apply_instant
            case armour:
                target.armour.apply_instant


#class StatusEffect(GameObject):
class StatusEffect:
    def apply(self, target): # GameObject):
        match self.type:
            case health:
                target.health.apply_status_effect
            case armour:
                target.armour.apply_status_effect



class EntityAttribute:
    """Represents an attribute that is part of an entity.

    Attributes
    ----------
    applied_status_effect: StatusEffect | None
        The currently applied status effect.
    """

    __slots__ = (
        "_value",
        "_max_value",
        "applied_status_effect",
        "attribute_data",
        "variable",
        "maximum",
    )

    def __init__(
            self: EntityAttribute,
            name: str,
            attribute_data: EntityAttributeData,
    ) -> None:
        """Initialise the object.

        Parameters
        ----------
        level: int
            The level to initialise the attribute at. These should start at 0 and
            increase over the game time.
        attribute_data: EntityAttributeData
            The data for this attribute.
        """
        print(name, attribute_data)
