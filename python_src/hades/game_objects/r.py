# Builtin
from hades.game_objects.objects import GameObject
from hades.game_objects.enums import GameObjectData, EntityAttributeData, ComponentType

class Temp(GameObject):
    def __init__(self, data: GameObjectData) -> None:
        super().__init__(5, 5, data)
        print(self.attributes)


f = Temp(GameObjectData(name="temp", textures={}, component_data={
    ComponentType.HEALTH: EntityAttributeData(value=5, maximum=True, variable=True),
}))
print(f.status)
print(f.health)
print(f.armour)
