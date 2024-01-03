// External headers
#include <pybind11/stl.h>

// Local headers
#include "game_objects/stats.hpp"
#include "game_objects/systems/armour_regen.hpp"
#include "game_objects/systems/attacks.hpp"
#include "game_objects/systems/effects.hpp"
#include "game_objects/systems/inventory.hpp"
#include "game_objects/systems/movements.hpp"
#include "game_objects/systems/upgrade.hpp"
#include "generation/map.hpp"

// ----- STRUCTURES ------------------------------------------
/// The hash function for a pybind11 handle.
struct py_handle_hash {
  /// Calculate the hash of a pybind11 handle.
  ///
  /// @param handle - The handle to calculate the hash of.
  /// @return The hash of the handle.
  auto operator()(const pybind11::handle &handle) const -> std::size_t { return hash(handle); }
};

/// The equality function for a pybind11 handle.
struct py_handle_equal {
  /// Check if two pybind11 handles are equal.
  ///
  /// @param handle_one - The first handle to compare.
  /// @param handle_two - The second handle to compare.
  /// @return Whether the two handles are equal or not.
  auto operator()(const pybind11::handle &handle_one, const pybind11::handle &handle_two) const noexcept -> bool {
    return handle_one.is(handle_two);
  }
};

// ----- FUNCTIONS -------------------------------------------
/// Get the system from the registry.
///
/// @tparam T - The type of system to find.
/// @param registry - The registry that manages the game objects, components, and systems.
/// @throws RegistryError - If the system is not registered.
/// @return The system from the registry.
template <typename T>
auto get_system_impl(const Registry &registry) -> std::shared_ptr<SystemBase> {
  return registry.get_system<T>();
}

/// Make the component types mapping.
///
/// @return The component types mapping.
template <typename... Ts>
auto make_component_types() -> std::unordered_map<pybind11::handle, std::type_index, py_handle_hash, py_handle_equal> {
  return {{pybind11::type::of<Ts>(), typeid(Ts)}...};
}

/// Make the system types mapping
///
/// @return The system types mapping.
template <typename... Ts>
auto make_system_types()
    -> std::unordered_map<pybind11::handle, std::function<std::shared_ptr<SystemBase>(const Registry &)>,
                          py_handle_hash, py_handle_equal> {
  return {{pybind11::type::of<Ts>(), get_system_impl<Ts>}...};
}

/// Get the type index for a given component type.
///
/// @param component_type - The component type.
/// @throws std::runtime_error - If the component type is invalid.
/// @return The type index for the component type.
inline auto get_type_index(const pybind11::handle &component_type) -> std::type_index {
  static const auto &component_types{
      make_component_types<Armour, ArmourRegen, Attacks, EffectApplier, Footprints, Health, Inventory, KeyboardMovement,
                           Money, MovementForce, StatusEffectData, StatusEffects, SteeringMovement, Upgrades>()};
  const auto iter{component_types.find(component_type)};
  if (iter == component_types.end()) {
    throw std::runtime_error("Invalid component type provided.");
  }
  return iter->second;
}

/// Make a C++ action function from a pybind11 function.
///
/// @param py_func - The pybind11 function.
/// @return The C++ action function.
auto make_action_function(const pybind11::function &py_func) -> ActionFunction {
  // NOLINTNEXTLINE(bugprone-exception-escape)
  return [py_func](int level) { return py_func(level).cast<double>(); };
}

// ----- PYTHON MODULE CREATION ------------------------------
PYBIND11_MODULE(hades_extensions, module) {  // NOLINT
  // Add the module docstring and the custom converters
  module.doc() = "Manages the various C++ extension modules for the game.";

  // Create the generation module
  pybind11::module generation{
      module.def_submodule("generation", "Generates the dungeon and places game objects in it.")};
  pybind11::enum_<TileType>(generation, "TileType")
      .value("Empty", TileType::Empty)
      .value("Floor", TileType::Floor)
      .value("Wall", TileType::Wall)
      .value("Obstacle", TileType::Obstacle)
      .value("Player", TileType::Player)
      .value("Potion", TileType::Potion);
  pybind11::class_<LevelConstants>(generation, "LevelConstants", "Holds the constants for a specific level.")
      .def_readonly("level", &LevelConstants::level)
      .def_readonly("width", &LevelConstants::width)
      .def_readonly("height", &LevelConstants::height);
  generation.def("create_map", &create_map, pybind11::arg("level"), pybind11::arg("seed") = pybind11::none(),
                 "Generate the game map for a given game level.\n\n"
                 "Args:\n"
                 "    level: The game level to generate a map for.\n"
                 "    seed: The seed to initialise the random generator.\n\n"
                 "Returns:\n"
                 "    A tuple containing the generated map and the level constants.");

  // Create the game objects, game_objects/systems, and game_objects/components modules
  pybind11::module game_objects = module.def_submodule(
      "game_objects", "Contains the registry and the various components and systems that can be used with it.");
  const pybind11::module systems =
      game_objects.def_submodule("systems", "Contains the systems which manage the game objects.");
  const pybind11::module components =
      game_objects.def_submodule("components", "Contains the components which can be added to game objects.");

  // Add the global constants and the base classes
  game_objects.attr("SPRITE_SCALE") = SPRITE_SCALE;
  game_objects.attr("SPRITE_SIZE") = SPRITE_SIZE;
  const pybind11::class_<ComponentBase, std::shared_ptr<ComponentBase>> component_base(
      game_objects, "ComponentBase", "The base class for all components.");
  pybind11::class_<SystemBase, std::shared_ptr<SystemBase>>(game_objects, "SystemBase",
                                                            "The base class for all systems.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    registry: The registry that manages the game objects, components, and systems.")
      .def("get_registry", &SystemBase::get_registry,
           "Get the registry that manages the game objects, components, and systems.\n\n"
           "Returns:\n"
           "    The registry.")
      .def("update", &SystemBase::update, pybind11::arg("delta_time"),
           "Process update logic for a system.\n\n"
           "Args:\n"
           "    delta_time: The time interval since the last time the function was called.");

  // Add the steering structures
  pybind11::class_<Vec2d>(game_objects, "Vec2d", "Represents a 2D vector.")
      .def(pybind11::init<double, double>(), pybind11::arg("x"), pybind11::arg("y"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    x: The x value of the vector.\n"
           "    y: The y value of the vector.")
      .def_readwrite("x", &Vec2d::x)
      .def_readwrite("y", &Vec2d::y)
      .def("__eq__", &Vec2d::operator==,
           "Check if this vector is equal to another vector.\n\n"
           "Args:\n"
           "    other: The other vector to compare to.\n\n"
           "Returns:\n"
           "    Whether the vectors are equal or not.")
      .def("__ne__", &Vec2d::operator!=,
           "Check if this vector is not equal to another vector.\n\n"
           "Args:\n"
           "    other: The other vector to compare to.\n\n"
           "Returns:\n"
           "    Whether the vectors are not equal or not.")
      .def("__add__", &Vec2d::operator+,
           "Add another vector to this vector.\n\n"
           "Args:\n"
           "    other: The other vector to add.\n\n"
           "Returns:\n"
           "    The sum of the two vectors.")
      .def("__iadd__", &Vec2d::operator+=,
           "Add another vector to this vector.\n\n"
           "Args:\n"
           "    other: The other vector to add.\n\n"
           "Returns:\n"
           "    This vector.")
      .def("__sub__", &Vec2d::operator-,
           "Subtract another vector from this vector.\n\n"
           "Args:\n"
           "    other: The other vector to subtract.\n\n"
           "Returns:\n"
           "    The difference of the two vectors.")
      .def("__mul__", &Vec2d::operator*,
           "Multiply this vector by a scalar.\n\n"
           "Args:\n"
           "    scalar: The scalar to multiply by.\n\n"
           "Returns:\n"
           "    The product of the vector and the scalar.")
      .def("__truediv__", &Vec2d::operator/,
           "Divide this vector by a scalar.\n\n"
           "Args:\n"
           "    scalar: The scalar to divide by.\n\n"
           "Returns:\n"
           "    The quotient of the vector and the scalar.")
      .def(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          "__iter__", [](const Vec2d &vec) { return pybind11::make_iterator(&vec.x, &vec.y + 1); },
          "Get an iterator to the vector.\n\n"
          "Returns:\n"
          "    An iterator to the vector.")
      .def("magnitude", &Vec2d::magnitude,
           "Calculate the magnitude of the vector.\n\n"
           "Returns:\n"
           "    The magnitude of the vector.")
      .def("normalised", &Vec2d::normalised,
           "Calculate the normalised vector.\n\n"
           "Returns:\n"
           "    The normalised vector.")
      .def("rotated", &Vec2d::rotated, pybind11::arg("angle"),
           "Rotate the vector by an angle.\n\n"
           "Args:\n"
           "    angle: The angle to rotate by.\n\n"
           "Returns:\n"
           "    The rotated vector.")
      .def("angle_between", &Vec2d::angle_between, pybind11::arg("other"),
           "Calculate the angle between this vector and another vector.\n\n"
           "Args:\n"
           "    other: The other vector to calculate the angle between.\n\n"
           "Returns:\n"
           "    The angle between the two vectors.")
      .def("distance_to", &Vec2d::distance_to, pybind11::arg("other"),
           "Calculate the distance between this vector and another vector.\n\n"
           "Args:\n"
           "    other: The other vector to calculate the distance between.\n\n"
           "Returns:\n"
           "    The distance between the two vectors.");
  pybind11::class_<KinematicObject, std::shared_ptr<KinematicObject>>(
      game_objects, "KinematicObject", "Stores various data about a game object for use in physics-related operations.")
      .def_readwrite("position", &KinematicObject::position)
      .def_readwrite("velocity", &KinematicObject::velocity)
      .def_readwrite("rotation", &KinematicObject::rotation);

  // Add the registry class
  register_exception<RegistryError>(game_objects, "RegistryError");
  pybind11::class_<Registry> registry_class(game_objects, "Registry",
                                            "Manages game objects, components, and systems that are registered.");
  registry_class.def(pybind11::init<>(), "Initialise the object.")
      .def("create_game_object", &Registry::create_game_object, pybind11::arg("components"),
           pybind11::arg("kinematic") = false,
           "Create a new game object.\n\n"
           "Args:\n"
           "    components: The components to add to the game object.\n"
           "    kinematic: Whether the game object should have a kinematic object or not.\n\n"
           "Returns:\n"
           "    The game object ID.")
      .def("delete_game_object", &Registry::delete_game_object, pybind11::arg("game_object_id"),
           "Delete a game object.\n\n"
           "Args:\n"
           "    game_object_id: The game object ID.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object is not registered.")
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
      .def(
          "add_systems",
          [](Registry &registry) {
            registry.add_system<ArmourRegenSystem>();
            registry.add_system<AttackSystem>();
            registry.add_system<DamageSystem>();
            registry.add_system<EffectSystem>();
            registry.add_system<InventorySystem>();
            registry.add_system<FootprintSystem>();
            registry.add_system<KeyboardMovementSystem>();
            registry.add_system<SteeringMovementSystem>();
            registry.add_system<UpgradeSystem>();
          },
          "Add all the systems into the registry.\n\n"
          "Raises:\n"
          "    RegistryError: If one of the systems is already registered.")
      .def(
          "get_system",
          [](const Registry &registry, const pybind11::object &system_type) {
            // Get all the system types and check if the given system type exists
            static const auto &system_types =
                make_system_types<ArmourRegenSystem, AttackSystem, DamageSystem, EffectSystem, InventorySystem,
                                  FootprintSystem, KeyboardMovementSystem, SteeringMovementSystem, UpgradeSystem>();
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
          "    The system from the registry.")
      .def("update", &Registry::update, pybind11::arg("delta_time"),
           "Update all systems in the registry.\n\n"
           "Args:\n"
           "    delta_time: The time interval since the last time the function was called.")
      .def("get_kinematic_object", &Registry::get_kinematic_object, pybind11::arg("game_object_id"),
           "Get the kinematic object for a game object.\n\n"
           "Args:\n"
           "    game_object_id: The game object ID.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object is not registered or if the game object does not have a "
           "kinematic object.\n\n"
           "Returns:\n"
           "    The kinematic object.")
      .def("add_wall", &Registry::add_wall, pybind11::arg("wall"),
           "Add a wall to the registry.\n\n"
           "Args:\n"
           "    wall: The wall to add to the registry.")
      .def("get_walls", &Registry::get_walls,
           "Get the walls from the registry.\n\n"
           "Returns:\n"
           "    The walls in the registry.");

  // Add the stat components
  pybind11::class_<Stat, ComponentBase, std::shared_ptr<Stat>>(
      components, "Stat", "Represents a component that has a variable value and maximum value.")
      .def(pybind11::init<double, int>(), pybind11::arg("value"), pybind11::arg("maximum_level"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    value: The initial and maximum value of the stat.\n"
           "    maximum_level: The maximum level of the stat.")
      .def("get_value", &Stat::get_value,
           "Get the value of the stat.\n\n"
           "Returns:\n"
           "    The value of the stat.")
      .def("set_value", &Stat::set_value,
           "Set the value of the stat.\n\n"
           "Args:\n"
           "    new_value: The new value of the stat.")
      .def("get_max_value", &Stat::get_max_value,
           "Get the maximum value of the stat.\n\n"
           "Returns:\n"
           "    The maximum value of the stat.")
      .def("add_to_max_value", &Stat::add_to_max_value,
           "Add a value to the maximum value of the stat.\n\n"
           "Args:\n"
           "    value: The value to add to the maximum value of the stat.")
      .def("get_current_level", &Stat::get_current_level,
           "Get the current level of the stat.\n\n"
           "Returns:\n"
           "    The current level of the stat.")
      .def("increment_current_level", &Stat::increment_current_level, "Increment the current level of the stat.")
      .def("get_maximum_level", &Stat::get_maximum_level,
           "Get the maximum level of the stat.\n\n"
           "Returns:\n"
           "    The maximum level of the stat.");
  pybind11::class_<Armour, Stat, std::shared_ptr<Armour>>(components, "Armour",
                                                          "Allows a game object to have an armour stat.")
      .def(pybind11::init<double, int>(), pybind11::arg("value"), pybind11::arg("maximum_level"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    value: The initial and maximum value of the armour stat.\n"
           "    maximum_level: The maximum level of the armour stat.");
  pybind11::class_<ArmourRegen, Stat, std::shared_ptr<ArmourRegen>>(components, "ArmourRegen",
                                                                    "Allows a game object to regenerate armour.")
      .def(pybind11::init<double, int>(), pybind11::arg("value"), pybind11::arg("maximum_level"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    value: The initial and maximum value of the armour regen stat.\n"
           "    maximum_level: The maximum level of the armour regen stat.");
  pybind11::class_<Health, Stat, std::shared_ptr<Health>>(components, "Health",
                                                          "Allows a game object to have a health stat.")
      .def(pybind11::init<double, int>(), pybind11::arg("value"), pybind11::arg("maximum_level"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    value: The initial and maximum value of the health stat.\n"
           "    maximum_level: The maximum level of the health stat.");
  pybind11::class_<MovementForce, Stat, std::shared_ptr<MovementForce>>(
      components, "MovementForce", "Allows a game object to determine how fast it can move.")
      .def(pybind11::init<double, int>(), pybind11::arg("value"), pybind11::arg("maximum_level"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    value: The initial and maximum value of the movement force stat.\n"
           "    maximum_level: The maximum level of the movement force stat.");

  // Add the armour regen system as well as relevant structures/components
  pybind11::class_<ArmourRegenSystem, SystemBase, std::shared_ptr<ArmourRegenSystem>>(
      systems, "ArmourRegenSystem", "Provides facilities to manipulate armour regen components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    registry: The registry that manages the game objects, components, and systems.")
      .def("update", &ArmourRegenSystem::update, pybind11::arg("delta_time"),
           "Process update logic for an armour regeneration component.\n\n"
           "Args:\n"
           "    delta_time: The time interval since the last time the function was called.");

  // Add the attack system and the damage system as well as relevant structures/components
  pybind11::enum_<AttackAlgorithm>(game_objects, "AttackAlgorithm",
                                   "Stores the different types of attack algorithms available.")
      .value("AreaOfEffect", AttackAlgorithm::AreaOfEffect)
      .value("Melee", AttackAlgorithm::Melee)
      .value("Ranged", AttackAlgorithm::Ranged);
  pybind11::class_<Attacks, ComponentBase, std::shared_ptr<Attacks>>(
      components, "Attacks", "Allows a game object to attack other game objects.")
      .def(pybind11::init<std::vector<AttackAlgorithm>>(), pybind11::arg("attack_algorithms"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    attack_algorithms: The attack algorithms the game object can use.")
      .def_readwrite("attack_algorithms", &Attacks::attack_algorithms)
      .def_readwrite("attack_state", &Attacks::attack_state);
  pybind11::class_<AttackSystem, SystemBase, std::shared_ptr<AttackSystem>>(
      systems, "AttackSystem", "Provides facilities to manipulate attack components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    registry: The registry that manages the game objects, components, and systems.")
      .def("do_attack", &AttackSystem::do_attack, pybind11::arg("game_object_id"), pybind11::arg("targets"),
           "Perform the currently selected attack algorithm.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to perform the attack for.\n"
           "    targets: The targets to attack.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have an attack component.\n\n"
           "Returns:\n"
           "    The result of the attack.")
      .def("previous_attack", &AttackSystem::previous_attack, pybind11::arg("game_object_id"),
           "Select the previous attack algorithm.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to select the previous attack algorithm for.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have an attack component.")
      .def("next_attack", &AttackSystem::next_attack, pybind11::arg("game_object_id"),
           "Select the next attack algorithm.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to select the next attack algorithm for.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have an attack component.");
  pybind11::class_<DamageSystem, SystemBase, std::shared_ptr<DamageSystem>>(
      systems, "DamageSystem", "Provides facilities to damage game objects.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    registry: The registry that manages the game objects, components, and systems.")
      .def("deal_damage", &DamageSystem::deal_damage,
           "Deal damage to a game object.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to deal damage to.\n"
           "    damage: The amount of damage to deal.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have a health component.");

  // Add the effect system as well as relevant structures/components
  pybind11::enum_<StatusEffectType>(game_objects, "StatusEffectType",
                                    "Stores the different types of status effects available.")
      .value("TEMP", StatusEffectType::TEMP)
      .value("TEMP2", StatusEffectType::TEMP2);
  pybind11::class_<StatusEffect>(components, "StatusEffect",
                                 "Represents a status effect that can be applied to a game object.")
      .def_readwrite("value", &StatusEffect::value)
      .def_readwrite("duration", &StatusEffect::duration)
      .def_readwrite("interval", &StatusEffect::interval)
      .def_readwrite("target_component", &StatusEffect::target_component);
  pybind11::class_<StatusEffectData>(components, "StatusEffectData",
                                     "Represents the data required to apply a status effect.")
      .def(pybind11::init([](const StatusEffectType status_effect_type, const pybind11::function &increase,
                             const pybind11::function &duration, const pybind11::function &interval) {
             return StatusEffectData(status_effect_type, make_action_function(increase), make_action_function(duration),
                                     make_action_function(interval));
           }),
           pybind11::arg("status_effect_type"), pybind11::arg("increase"), pybind11::arg("duration"),
           pybind11::arg("interval"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    status_effect_type: The type of status effect.\n"
           "    increase: The increase function to apply.\n"
           "    duration: The duration function to apply.\n"
           "    interval: The interval function to apply.")
      .def_readwrite("status_effect_type", &StatusEffectData::status_effect_type)
      .def_readwrite("increase", &StatusEffectData::increase)
      .def_readwrite("duration", &StatusEffectData::duration)
      .def_readwrite("interval", &StatusEffectData::interval);
  pybind11::class_<EffectApplier, ComponentBase, std::shared_ptr<EffectApplier>>(
      components, "EffectApplier", "Allows a game object to provide instant or status effects.")
      .def(pybind11::init([](const pybind11::dict &instant_effects, const pybind11::dict &status_effects) {
             // Create two mappings to hold the instant and status effects
             std::unordered_map<std::type_index, ActionFunction> target_instant_effects;
             std::unordered_map<std::type_index, StatusEffectData> target_status_effects;

             // Iterate through the instant effects and add them to the mapping
             for (const auto &[type, func] : instant_effects) {
               target_instant_effects.emplace(get_type_index(type),
                                              make_action_function(func.cast<pybind11::function>()));
             }

             // Iterate through the status effects and add them to the mapping
             for (const auto &[type, data] : status_effects) {
               auto data_dict = data.cast<pybind11::dict>();
               const auto status_effect_type = data_dict["status_effect_type"].cast<StatusEffectType>();
               const auto increase = make_action_function(data_dict["increase"].cast<pybind11::function>());
               const auto duration = make_action_function(data_dict["duration"].cast<pybind11::function>());
               const auto interval = make_action_function(data_dict["interval"].cast<pybind11::function>());
               target_status_effects.emplace(get_type_index(type),
                                             StatusEffectData{status_effect_type, increase, duration, interval});
             }

             // Initialise the object
             return std::make_shared<EffectApplier>(target_instant_effects, target_status_effects);
           }),
           pybind11::arg("instant_effects"), pybind11::arg("status_effects"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    instant_effects: The instant effects the game object provides.\n"
           "    status_effects: The status effects the game object provides.");
  pybind11::class_<StatusEffects, ComponentBase, std::shared_ptr<StatusEffects>>(
      components, "StatusEffects", "Allows a game object to have status effects applied to it.")
      .def(pybind11::init<>(), "Initialise the object.");
  pybind11::class_<EffectSystem, SystemBase, std::shared_ptr<EffectSystem>>(
      systems, "EffectSystem", "Provides facilities to manipulate instant and status effects.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    registry: The registry that manages the game objects, components, and systems.")
      .def("update", &EffectSystem::update, pybind11::arg("delta_time"),
           "Process update logic for a status effect component.\n\n"
           "Args:\n"
           "    delta_time: The time interval since the last time the function was called.")
      .def("apply_effects", &EffectSystem::apply_effects, pybind11::arg("game_object_id"),
           pybind11::arg("target_game_object_id"),
           "Apply effects to a game object..\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to get the effects from.\n"
           "    target_game_object_id: The ID of the game object to apply the effects to.\n\n"
           "Raises:\n"
           "    RegistryError: If either game object does not exist or does not have the required components.\n\n"
           "Returns:\n"
           "    Whether the effects were applied or not.");

  // Add the inventory system as well as relevant structures/components
  register_exception<InventorySpaceError>(game_objects, "InventorySpaceError");
  pybind11::class_<Inventory, ComponentBase, std::shared_ptr<Inventory>>(
      components, "Inventory", "Allows a game object to have a fixed size inventory.")
      .def(pybind11::init<int, int>(), pybind11::arg("width"), pybind11::arg("height"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    width: The width of the inventory.\n"
           "    height: The height of the inventory.")
      .def_readwrite("width", &Inventory::width)
      .def_readwrite("height", &Inventory::height)
      .def_readwrite("items", &Inventory::items)
      .def("get_capacity", &Inventory::get_capacity,
           "Get the capacity of the inventory.\n\n"
           "Returns:\n"
           "    The capacity of the inventory.");
  pybind11::class_<InventorySystem, SystemBase, std::shared_ptr<InventorySystem>>(
      systems, "InventorySystem", "Provides facilities to manipulate inventory components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    registry: The registry that manages the game objects, components, and systems.")
      .def("add_item_to_inventory", &InventorySystem::add_item_to_inventory, pybind11::arg("game_object_id"),
           pybind11::arg("item"),
           "Add an item to the inventory of a game object.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to add the item to.\n"
           "    item: The item to add to the inventory.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have an inventory component.\n"
           "    InventorySpaceError: If the inventory is full.")
      .def("remove_item_from_inventory", &InventorySystem::remove_item_from_inventory, pybind11::arg("game_object_id"),
           pybind11::arg("index"),
           "Remove an item from the inventory of a game object.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to remove the item from.\n"
           "    index: The index of the item to remove from the inventory.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have an inventory component.\n"
           "    InventorySpaceError: If the inventory is empty or if the index is out of bounds.\n\n"
           "Returns:\n"
           "    The item removed from the inventory.");

  // Add the footprint system, the keyboard movement system, and the steering movement system as well as relevant
  // structures/components
  pybind11::enum_<SteeringBehaviours>(game_objects, "SteeringBehaviours",
                                      "Stores the different types of steering behaviours available.")
      .value("Arrive", SteeringBehaviours::Arrive)
      .value("Evade", SteeringBehaviours::Evade)
      .value("Flee", SteeringBehaviours::Flee)
      .value("FollowPath", SteeringBehaviours::FollowPath)
      .value("ObstacleAvoidance", SteeringBehaviours::ObstacleAvoidance)
      .value("Pursue", SteeringBehaviours::Pursue)
      .value("Seek", SteeringBehaviours::Seek)
      .value("Wander", SteeringBehaviours::Wander);
  pybind11::enum_<SteeringMovementState>(game_objects, "SteeringMovementState",
                                         "Stores the different states the steering movement component can be in.")
      .value("Default", SteeringMovementState::Default)
      .value("Footprint", SteeringMovementState::Footprint)
      .value("Target", SteeringMovementState::Target);
  pybind11::class_<Footprints, ComponentBase, std::shared_ptr<Footprints>>(
      components, "Footprints", "Allows a game object to periodically leave footprints around the game map.")
      .def(pybind11::init<>(), "Initialise the object.")
      .def_readwrite("footprints", &Footprints::footprints);
  pybind11::class_<KeyboardMovement, ComponentBase, std::shared_ptr<KeyboardMovement>>(
      components, "KeyboardMovement", "Allows a game object's movement to be controlled by the keyboard.")
      .def(pybind11::init<>(), "Initialise the object.")
      .def_readwrite("moving_north", &KeyboardMovement::moving_north)
      .def_readwrite("moving_east", &KeyboardMovement::moving_east)
      .def_readwrite("moving_south", &KeyboardMovement::moving_south)
      .def_readwrite("moving_west", &KeyboardMovement::moving_west);
  pybind11::class_<SteeringMovement, ComponentBase, std::shared_ptr<SteeringMovement>>(
      components, "SteeringMovement", "Allows a game object's movement to be controlled by steering behaviours.")
      .def(pybind11::init<std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>>>(),
           pybind11::arg("behaviours"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    behaviours: The steering behaviours the game object can use.")
      .def_readwrite("behaviours", &SteeringMovement::behaviours)
      .def_readwrite("movement_state", &SteeringMovement::movement_state)
      .def_readwrite("target_id", &SteeringMovement::target_id)
      .def_readwrite("path_list", &SteeringMovement::path_list);
  pybind11::class_<FootprintSystem, SystemBase, std::shared_ptr<FootprintSystem>>(
      systems, "FootprintSystem", "Provides facilities to manipulate footprint components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    registry: The registry that manages the game objects, components, and systems.")
      .def("update", &FootprintSystem::update, pybind11::arg("delta_time"),
           "Process update logic for a footprint component.\n\n"
           "Args:\n"
           "    delta_time: The time interval since the last time the function was called.");
  pybind11::class_<KeyboardMovementSystem, SystemBase, std::shared_ptr<KeyboardMovementSystem>>(
      systems, "KeyboardMovementSystem", "Provides facilities to manipulate keyboard movement components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    registry: The registry that manages the game objects, components, and systems.")
      .def("calculate_force", &KeyboardMovementSystem::calculate_force, pybind11::arg("game_object_id"),
           "Calculate the new keyboard force to apply to the game object.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to calculate the keyboard force for.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have a keyboard movement "
           "component.\n\n"
           "Returns:\n"
           "    The new force to apply to the game object.");
  pybind11::class_<SteeringMovementSystem, SystemBase, std::shared_ptr<SteeringMovementSystem>>(
      systems, "SteeringMovementSystem", "Provides facilities to manipulate steering movement components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    registry: The registry that manages the game objects, components, and systems.")
      .def("calculate_force", &SteeringMovementSystem::calculate_force, pybind11::arg("game_object_id"),
           "Calculate the new steering force to apply to the game object.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to calculate the steering force for.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have a steering movement "
           "component.\n\n"
           "Returns:\n"
           "    The new force to apply to the game object.")
      .def("update_path_list", &SteeringMovementSystem::update_path_list, pybind11::arg("game_object_id"),
           pybind11::arg("path_list"),
           "Update the path lists for the game objects to follow.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to follow.\n"
           "    path_list: The list of footprints to follow.");

  // Add the upgrade system as well as relevant structures/components
  pybind11::class_<Money, ComponentBase, std::shared_ptr<Money>>(
      components, "Money", "Allows a game object to record the amount of money it has.")
      .def(pybind11::init<int>(), pybind11::arg("money"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    money: The amount of money the game object has.")
      .def_readwrite("money", &Money::money);
  pybind11::class_<Upgrades, ComponentBase, std::shared_ptr<Upgrades>>(components, "Upgrades",
                                                                       "Allows a game object to be upgraded.")
      .def(pybind11::init([](const pybind11::dict &upgrades) {
             // Create a mapping to hold the upgrades
             std::unordered_map<std::type_index, ActionFunction> target_upgrades;

             // Iterate through the upgrades and add them to the mapping
             for (const auto &[type, func] : upgrades) {
               target_upgrades.emplace(get_type_index(type), make_action_function(func.cast<pybind11::function>()));
             }

             // Initialise the object
             return std::make_shared<Upgrades>(target_upgrades);
           }),
           pybind11::arg("upgrades"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    upgrades: The upgrades the game object has.");
  pybind11::class_<UpgradeSystem, SystemBase, std::shared_ptr<UpgradeSystem>>(
      systems, "UpgradeSystem", "Provides facilities to manipulate game object upgrades.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    registry: The registry that manages the game objects, components, and systems.")
      .def("upgrade_component", &UpgradeSystem::upgrade_component, pybind11::arg("game_object_id"),
           pybind11::arg("target_component"),
           "Upgrade a component to the next level if possible.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to upgrade the component for.\n"
           "    target_component: The type of component to upgrade.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have the target component.\n\n"
           "Returns:\n"
           "    Whether the component upgrade was successful or not.");
}
