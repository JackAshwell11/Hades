// Local headers
#include "common.hpp"
#include "ecs/registry.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/shop.hpp"

void bind_ecs(const pybind11::module_& module) {
  // Add the global constants, functions, and base classes
  module.attr("SPRITE_SCALE") = SPRITE_SCALE;
  module.attr("SPRITE_SIZE") = SPRITE_SIZE;
  const pybind11::class_<SystemBase, std::shared_ptr<SystemBase>> system_base(module, "SystemBase",
                                                                              "The base class for all systems.");

  // Add the enums
  pybind11::enum_<GameObjectType>(module, "GameObjectType", "Stores the different types of game objects available.")
      .value("Bullet", GameObjectType::Bullet)
      .value("Enemy", GameObjectType::Enemy)
      .value("Floor", GameObjectType::Floor)
      .value("Player", GameObjectType::Player)
      .value("Wall", GameObjectType::Wall)
      .value("Goal", GameObjectType::Goal)
      .value("HealthPotion", GameObjectType::HealthPotion)
      .value("Chest", GameObjectType::Chest);
  pybind11::enum_<EffectType>(module, "EffectType", "Stores the different types of effects available.")
      .value("Regeneration", EffectType::Regeneration)
      .value("Poison", EffectType::Poison);

  // Add the registry class
  register_exception<RegistryError>(module, "RegistryError");
  pybind11::class_<Registry, std::shared_ptr<Registry>>(
      module, "Registry", "Manages game objects, components, and systems that are registered.")
      .def(
          "get_system",
          [](const Registry& registry, const pybind11::object& system_type) -> std::shared_ptr<SystemBase> {
            if (system_type.is(pybind11::type::of<InventorySystem>())) {
              return registry.get_system<InventorySystem>();
            }
            if (system_type.is(pybind11::type::of<ShopSystem>())) {
              return registry.get_system<ShopSystem>();
            }
            throw std::runtime_error("Invalid system type provided.");
          },
          pybind11::arg("system_type"),
          "Get a system from the registry.\n\n"
          "Args:\n"
          "    system_type: The type of system to find.\n\n"
          "Raises:\n"
          "    RegistryError: If the system type is not registered.\n"
          "    RuntimeError: If the system type is invalid..\n\n"
          "Returns:\n"
          "    The system from the registry.");
}
