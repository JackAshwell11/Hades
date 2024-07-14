// External headers
#include <chipmunk/chipmunk_structs.h>
#ifdef Py_DEBUG
#undef Py_DEBUG
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#define Py_DEBUG
#else
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#endif

// Local headers
#include "game_objects/stats.hpp"
#include "game_objects/systems/armour_regen.hpp"
#include "game_objects/systems/attacks.hpp"
#include "game_objects/systems/effects.hpp"
#include "game_objects/systems/inventory.hpp"
#include "game_objects/systems/movements.hpp"
#include "game_objects/systems/physics.hpp"
#include "game_objects/systems/sprite.hpp"
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
      make_component_types<Armour, ArmourRegen, Attack, EffectApplier, Footprints, Health, Inventory, KeyboardMovement,
                           KinematicComponent, Money, MovementForce, PythonSprite, StatusEffectData, StatusEffect,
                           SteeringMovement, Upgrades>()};
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

  // Create the game_objects, game_objects/components, and game_objects/systems modules
  pybind11::module game_objects = module.def_submodule(
      "game_objects", "Contains the registry and the various components and systems that can be used with it.");
  const pybind11::module systems =
      game_objects.def_submodule("systems", "Contains the systems which manage the game objects.");
  const pybind11::module components =
      game_objects.def_submodule("components", "Contains the components which can be added to game objects.");

  // Add the global constants, functions, and base classes
  game_objects.attr("SPRITE_SCALE") = SPRITE_SCALE;
  game_objects.attr("SPRITE_SIZE") = SPRITE_SIZE;
  game_objects.def(
      "grid_pos_to_pixel",
      [](const cpVect &position) {
        auto [x, y] = grid_pos_to_pixel(position);
        return pybind11::make_tuple(x, y);
      },
      "Calculate the screen position based on a grid position.\n\n"
      "Args:\n"
      "    position: The position in the grid.\n\n"
      "Raises:\n"
      "    ValueError: If the position is negative.\n\n"
      "Returns:\n"
      "    The pixel position.");
  const pybind11::class_<ComponentBase, std::shared_ptr<ComponentBase>> component_base(
      game_objects, "ComponentBase", "The base class for all components.");
  const pybind11::class_<SystemBase, std::shared_ptr<SystemBase>> system_base(game_objects, "SystemBase",
                                                                              "The base class for all systems.");

  // Add the cpVect class
  pybind11::class_<cpVect>(game_objects, "Vec2d", "Represents a 2D vector.")
      .def(pybind11::init<float, float>(), pybind11::arg("x"), pybind11::arg("y"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    x: The x value of the vector.\n"
           "    y: The y value of the vector.")
      .def_readonly("x", &cpVect::x)
      .def_readonly("y", &cpVect::y)
      .def(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          "__iter__", [](const cpVect &vec) { return pybind11::make_iterator(&vec.x, &vec.y + 1); },
          "Get an iterator to the vector.\n\n"
          "Returns:\n"
          "    An iterator to the vector.")
      .def(
          "__mul__", [](const cpVect &vec, const float scalar) { return cpvmult(vec, scalar); },
          pybind11::arg("scalar"),
          "Multiply the vector by a scalar.\n\n"
          "Args:\n"
          "    scalar: The scalar to multiply the vector by.\n\n"
          "Returns:\n"
          "    The vector multiplied by the scalar.");

  // Add the enums
  pybind11::enum_<AttackAlgorithm>(game_objects, "AttackAlgorithm",
                                   "Stores the different types of attack algorithms available.")
      .value("AreaOfEffect", AttackAlgorithm::AreaOfEffect)
      .value("Melee", AttackAlgorithm::Melee)
      .value("Ranged", AttackAlgorithm::Ranged);
  pybind11::enum_<GameObjectType>(game_objects, "GameObjectType",
                                  "Stores the different types of game objects available.")
      .value("Bullet", GameObjectType::Bullet)
      .value("Enemy", GameObjectType::Enemy)
      .value("Floor", GameObjectType::Floor)
      .value("Player", GameObjectType::Player)
      .value("Potion", GameObjectType::Potion)
      .value("Wall", GameObjectType::Wall);
  pybind11::enum_<StatusEffectType>(game_objects, "StatusEffectType",
                                    "Stores the different types of status effects available.")
      .value("TEMP", StatusEffectType::TEMP)
      .value("TEMP2", StatusEffectType::TEMP2);
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
  pybind11::enum_<EventType>(game_objects, "EventType", "Stores the different types of events that can occur.")
      .value("BulletCreation", EventType::BulletCreation)
      .value("GameObjectDeath", EventType::GameObjectDeath)
      .value("InventoryUpdate", EventType::InventoryUpdate);

  // Add the registry class
  register_exception<RegistryError>(game_objects, "RegistryError");
  pybind11::class_<Registry>(game_objects, "Registry",
                             "Manages game objects, components, and systems that are registered.")
      .def(pybind11::init<>(), "Initialise the object.")
      .def("create_game_object", &Registry::create_game_object, pybind11::arg("game_object_type"),
           pybind11::arg("position"), pybind11::arg("components"),
           "Create a new game object.\n\n"
           "Args:\n"
           "    game_object_type: The type of game object to create.\n"
           "    position: The position of the game object.\n"
           "    components: The components to add to the game object.\n\n"
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
            registry.add_system<FootprintSystem>();
            registry.add_system<InventorySystem>();
            registry.add_system<KeyboardMovementSystem>();
            registry.add_system<PhysicsSystem>();
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
                make_system_types<ArmourRegenSystem, AttackSystem, DamageSystem, EffectSystem, FootprintSystem,
                                  InventorySystem, KeyboardMovementSystem, PhysicsSystem, SteeringMovementSystem,
                                  UpgradeSystem>();
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
      .def("add_callback", &Registry::add_callback, pybind11::arg("event_type"), pybind11::arg("callback"),
           "Add a callback to the registry to listen for events.\n\n"
           "Args:\n"
           "    event_type: The type of event to listen for.\n"
           "    callback: The callback to add.");

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
      .def("get_max_level", &Stat::get_max_level,
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

  // Add the components
  pybind11::class_<Attack, ComponentBase, std::shared_ptr<Attack>>(components, "Attack",
                                                                   "Allows a game object to attack other game objects.")
      .def(pybind11::init<std::vector<AttackAlgorithm>>(), pybind11::arg("attack_algorithms"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    attack_algorithms: The attack algorithms the game object can use.");
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
  pybind11::class_<Footprints, ComponentBase, std::shared_ptr<Footprints>>(
      components, "Footprints", "Allows a game object to periodically leave footprints around the game map.")
      .def(pybind11::init<>(), "Initialise the object.");
  pybind11::class_<Inventory, ComponentBase, std::shared_ptr<Inventory>>(
      components, "Inventory", "Allows a game object to have a fixed size inventory.")
      .def(pybind11::init<int, int>(), pybind11::arg("width"), pybind11::arg("height"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    width: The width of the inventory.\n"
           "    height: The height of the inventory.")
      .def("get_capacity", &Inventory::get_capacity,
           "Get the capacity of the inventory.\n\n"
           "Returns:\n"
           "    The capacity of the inventory.")
      .def_readonly("items", &Inventory::items);
  pybind11::class_<KeyboardMovement, ComponentBase, std::shared_ptr<KeyboardMovement>>(
      components, "KeyboardMovement", "Allows a game object's movement to be controlled by the keyboard.")
      .def(pybind11::init<>(), "Initialise the object.")
      .def_readwrite("moving_north", &KeyboardMovement::moving_north)
      .def_readwrite("moving_east", &KeyboardMovement::moving_east)
      .def_readwrite("moving_south", &KeyboardMovement::moving_south)
      .def_readwrite("moving_west", &KeyboardMovement::moving_west);
  pybind11::class_<KinematicComponent, ComponentBase, std::shared_ptr<KinematicComponent>>(
      components, "KinematicComponent", "Allows a game object to interact with the physics system.")
      .def(pybind11::init<std::vector<cpVect>>(), pybind11::arg("vertices"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    vertices: The vertices of the shape.")
      .def(pybind11::init<bool>(), pybind11::arg("is_static"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    is_static: Whether the game object is static or not.")
      .def(
          "get_position",
          [](const KinematicComponent &kinematic_component) {
            return pybind11::make_tuple(kinematic_component.body->p.x, kinematic_component.body->p.y);
          },
          "Get the position of the game object.\n\n"
          "Returns:\n"
          "    The position of the game object.")
      .def(
          "set_rotation",
          [](KinematicComponent &kinematic_component, const double angle) { kinematic_component.rotation = angle; },
          pybind11::arg("angle"),
          "Set the rotation of the game object.\n\n"
          "Args:\n"
          "    angle: The angle to set the game object to.");
  pybind11::class_<Money, ComponentBase, std::shared_ptr<Money>>(
      components, "Money", "Allows a game object to record the amount of money it has.")
      .def(pybind11::init<int>(), pybind11::arg("money"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    money: The amount of money the game object has.")
      .def_readonly("money", &Money::money);
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
           "    interval: The interval function to apply.");
  pybind11::class_<PythonSprite, ComponentBase, std::shared_ptr<PythonSprite>>(
      components, "PythonSprite", "Allows a game object to hold a reference to the Python sprite object.")
      .def(pybind11::init<>(),
           "Initialise the object.\n\n"
           "Args:\n"
           "    sprite: The reference to the Python sprite object.")
      .def_readwrite("sprite", &PythonSprite::sprite);
  pybind11::class_<StatusEffect, ComponentBase, std::shared_ptr<StatusEffect>>(
      components, "StatusEffect", "Allows a game object to have status effects applied to it.")
      .def(pybind11::init<>(), "Initialise the object.");
  pybind11::class_<SteeringMovement, ComponentBase, std::shared_ptr<SteeringMovement>>(
      components, "SteeringMovement", "Allows a game object's movement to be controlled by steering behaviours.")
      .def(pybind11::init<std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>>>(),
           pybind11::arg("behaviours"),
           "Initialise the object.\n\n"
           "Args:\n"
           "    behaviours: The steering behaviours the game object can use.")
      .def_readwrite("target_id", &SteeringMovement::target_id);
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

  // Add the systems
  const pybind11::class_<ArmourRegenSystem, SystemBase, std::shared_ptr<ArmourRegenSystem>> armour_regen_system(
      systems, "ArmourRegenSystem", "Provides facilities to manipulate armour regen components.");
  pybind11::class_<AttackSystem, SystemBase, std::shared_ptr<AttackSystem>>(
      systems, "AttackSystem", "Provides facilities to manipulate attack components.")
      .def("do_attack", &AttackSystem::do_attack, pybind11::arg("game_object_id"), pybind11::arg("targets"),
           "Perform the currently selected attack algorithm.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to perform the attack for.\n"
           "    targets: The targets to attack.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have an attack or kinematic component.")
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
      .def("deal_damage", &DamageSystem::deal_damage,
           "Deal damage to a game object.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to deal damage to.\n"
           "    damage: The amount of damage to deal.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have a health component.");
  pybind11::class_<EffectSystem, SystemBase, std::shared_ptr<EffectSystem>>(
      systems, "EffectSystem", "Provides facilities to manipulate instant and status effects.")
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
  const pybind11::class_<FootprintSystem, SystemBase, std::shared_ptr<FootprintSystem>> footprint_system(
      systems, "FootprintSystem", "Provides facilities to manipulate footprint components.");
  pybind11::class_<InventorySystem, SystemBase, std::shared_ptr<InventorySystem>>(
      systems, "InventorySystem", "Provides facilities to manipulate inventory components.")
      .def("add_item_to_inventory", &InventorySystem::add_item_to_inventory, pybind11::arg("game_object_id"),
           pybind11::arg("item"),
           "Add an item to the inventory of a game object.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to add the item to.\n"
           "    item: The item to add to the inventory.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have an inventory component.\n"
           "    RuntimeError: If the inventory is full.\n\n"
           "Returns:\n"
           "    Whether the item was added or not.")
      .def("remove_item_from_inventory", &InventorySystem::remove_item_from_inventory, pybind11::arg("game_object_id"),
           pybind11::arg("item_id"),
           "Remove an item from the inventory of a game object.\n\n"
           "Args:\n"
           "    game_object_id: The ID of the game object to remove the item from.\n"
           "    item_id: The ID of the item to remove from the inventory.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or does not have an inventory component.\n\n"
           "Returns:\n"
           "    Whether the item was removed or not.")
      .def("use_item", &InventorySystem::use_item, pybind11::arg("target_id"), pybind11::arg("item_id"),
           "Use an item from the inventory.\n\n"
           "Args:\n"
           "    target_id: The game object ID of the game object to use the item on.\n"
           "    item_id: The game object ID of the item to use.\n\n"
           "Raises:\n"
           "    RegistryError: If the game object does not exist or if the required systems is not registered.\n\n"
           "Returns:\n"
           "    Whether the item was used or not.");
  const pybind11::class_<KeyboardMovementSystem, SystemBase, std::shared_ptr<KeyboardMovementSystem>>
      keyboard_movement_system(systems, "KeyboardMovementSystem",
                               "Provides facilities to manipulate keyboard movement components.");
  const pybind11::class_<PhysicsSystem, SystemBase, std::shared_ptr<PhysicsSystem>> physics_system(
      systems, "PhysicsSystem", "Provides facilities to manipulate a game object's physics.");
  const pybind11::class_<SteeringMovementSystem, SystemBase, std::shared_ptr<SteeringMovementSystem>>
      steering_movement_system(systems, "SteeringMovementSystem",
                               "Provides facilities to manipulate steering movement components.");
  pybind11::class_<UpgradeSystem, SystemBase, std::shared_ptr<UpgradeSystem>>(
      systems, "UpgradeSystem", "Provides facilities to manipulate game object upgrades.")
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
