// Local headers
#include "common.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/effects.hpp"

void bind_ecs(const pybind11::module_ &module) {
  // Add the global constants, functions, and base classes
  module.attr("SPRITE_SCALE") = SPRITE_SCALE;
  module.attr("SPRITE_SIZE") = SPRITE_SIZE;
  const pybind11::class_<ComponentBase, std::shared_ptr<ComponentBase>> component_base(
      module, "ComponentBase", "The base class for all components.");
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
      .value("Chest", GameObjectType::Chest)
      .value("Shop", GameObjectType::Shop);
  pybind11::enum_<EffectType>(module, "EffectType", "Stores the different types of effects available.")
      .value("Regeneration", EffectType::Regeneration)
      .value("Poison", EffectType::Poison);

  // Add the registry class
  register_exception<RegistryError>(module, "RegistryError");
  pybind11::class_<Registry, std::shared_ptr<Registry>>(
      module, "Registry", "Manages game objects, components, and systems that are registered.")
      .def(
          "has_component",
          [](const Registry &registry, const GameObjectID game_object_id, const pybind11::handle &component_type) {
            return registry.has_component(game_object_id, get_type_index(component_type));
          },
          pybind11::arg("game_object_id"), pybind11::arg("component_type"),
          "Checks if a game object has a given component or not.\n\n"
          "Args:\n"
          "    game_object_id: The game object ID.\n"
          "    component_type: The type of component to check for.\n\n"
          "Raises:\n"
          "    RuntimeError: If the component type is invalid.\n\n"
          "Returns:\n"
          "    Whether the game object has the component or not.")
      .def(
          "get_component",
          [](const Registry &registry, const GameObjectID game_object_id, const pybind11::handle &component_type) {
            return registry.get_component(game_object_id, get_type_index(component_type));
          },
          pybind11::arg("game_object_id"), pybind11::arg("component_type"),
          "Get a component from the registry.\n\n"
          "Args:\n"
          "    game_object_id: The game object ID.\n"
          "    component_type: The type of component to get.\n\n"
          "Raises:\n"
          "    RegistryError: If the game object is not registered or if the game object does not have the "
          "component."
          "   RuntimeError: If the component type is invalid.\n\n"
          "Returns:\n"
          "    The component from the registry.")
      .def("get_game_object_type", &Registry::get_game_object_type, pybind11::arg("game_object_id"),
           "Get the type of a game object.\n\n"
           "Args:\n"
           "    game_object_id: The game object ID.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object is not registered.\n\n"
           "Returns:\n"
           "    The type of the game object.")
      .def(
          "get_system",
          [](const Registry &registry, const pybind11::object &system_type) {
            // Get all the system types and check if the given system type exists
            const auto &system_types = get_system_types();
            const auto iter = system_types.find(system_type);
            if (iter == system_types.end()) {
              throw std::runtime_error("Invalid system type provided.");
            }

            // Return the system from the registry
            return iter->second(registry);
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
