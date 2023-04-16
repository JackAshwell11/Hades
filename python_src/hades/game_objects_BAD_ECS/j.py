# class InstantEffect:
#     def apply(self, target): # GameObject):
#         match self.type:
#             case health:
#                 target.health.apply_instant
#             case armour:
#                 target.armour.apply_instant
#
#
# class StatusEffect:
#     def apply(self, target): # GameObject):
#         match self.type:
#             case health:
#                 target.health.apply_status_effect
#             case armour:
#                 target.armour.apply_status_effect
#
#
#
# class EntityAttribute:
#     """Represents an attribute that is part of an entity.
#
#     Attributes
#     ----------
#         The currently applied status effect.
#     """
#
#         "_value",
#         "_max_value",
#         "applied_status_effect",
#         "attribute_data",
#         "variable",
#         "maximum",
#
#     def __init__(
#             self: EntityAttribute,
#             name: str,
#             attribute_data: EntityAttributeData,
#     ) -> None:
#         """Initialise the object.
#
#         Parameters
#         ----------
#             The level to initialise the attribute at. These should start at 0 and
#             increase over the game time.
#             The data for this attribute.
#         """

from hades.game_objects.characters import Player

f = Player(**{"x": 5, "y": 6})
print(f)
print(f.health)
print(f.armour)
