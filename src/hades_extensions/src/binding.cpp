// Std headers
#include <optional>

// External headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Local headers
#include "game_objects/stats.hpp"
#include "game_objects/steering.hpp"
#include "game_objects/systems/armour_regen.hpp"
#include "game_objects/systems/attacks.hpp"
#include "game_objects/systems/effects.hpp"
#include "game_objects/systems/inventory.hpp"
#include "game_objects/systems/movements.hpp"
#include "game_objects/systems/upgrade.hpp"
#include "generation/map.hpp"

// ----- FUNCTIONS ---------------------------------------
/// Binds the get_component method for a given number of arbitrary component types.
///
/// @tparam T - The types of components to bind the method for.
/// @param registry_class - The registry class to bind the method to.
template <typename... T>
void bind_components(pybind11::class_<Registry> &registry_class) {
  (registry_class.def("get_component", &Registry::get_component<T>, pybind11::arg("game_object_id"),
                      ("Get a component from the registry\n\n"
                       "Args:\n"
                       "    game_object_id: The game object ID.\n\n"
                       "Raises:\n"
                       "    RegistryException: If the game object is not registered or if the game object does not "
                       "have the component.\n\n"
                       "Returns:\n"
                       "    The component from the registry.")),
   ...);
}

/// Binds the find_system method for a given number of arbitrary system types.
///
/// @tparam T - The types of systems to bind the method for.
/// @param registry_class - The registry class to bind the method to.
template <typename... T>
void bind_systems(pybind11::class_<Registry> &registry_class) {
  (registry_class.def("add_system", &Registry::add_system<T>,
                      ("Add a system to the registry.\n\n"
                       "Raises:\n"
                       "    RegistryException: If the system is already registered.")),
   ...);
  (registry_class.def("find_system", &Registry::find_system<T>,
                      ("Find a system in the registry.\n\n"
                       "Raises:\n"
                       "    RegistryException: If the system is not registered.\n\n"
                       "Returns:\n"
                       "    The system.")),
   ...);
}

// ----- PYTHON MODULE CREATION ------------------------------
PYBIND11_MODULE(hades_extensions, module) {  // NOLINT
  // Add the module docstring
  module.doc() = "Manages the various C++ extension modules for the game.";

  // Create the generation module
  pybind11::module generation =
      module.def_submodule("generation", "Generates the dungeon and places game objects in it.");
  generation.def("create_map", &create_map, pybind11::arg("level"), pybind11::arg("seed") = pybind11::none(),
                 ("Generate the game map for a given game level.\n\n"
                  "Args:\n"
                  "    level: The game level to generate a map for.\n"
                  "    seed: The seed to initialise the random generator.\n\n"
                  "Returns:\n"
                  "    A tuple containing the generated map and the level constants."));
  pybind11::enum_<TileType>(generation, "TileType")
      .value("Empty", TileType::Empty)
      .value("Floor", TileType::Floor)
      .value("Wall", TileType::Wall)
      .value("Obstacle", TileType::Obstacle)
      .value("Player", TileType::Player)
      .value("Potion", TileType::Potion);

  // Create the game objects module and the game_objects/systems submodules
  pybind11::module game_objects = module.def_submodule(
      "game_objects", "Contains the registry and the various components and systems that can be used with it.");
  const pybind11::module systems =
      game_objects.def_submodule("systems", "Contains the systems which manage the game objects.");
  game_objects.attr("SPRITE_SCALE") = SPRITE_SCALE;
  game_objects.attr("SPRITE_SIZE") = SPRITE_SIZE;
  const pybind11::class_<ComponentBase> component_base(game_objects, "ComponentBase",
                                                       "The base class for all components.");
  pybind11::class_<SystemBase>(game_objects, "SystemBase", "The base class for all systems.")
      .def("update", &SystemBase::update,
           ("Process update logic for a system.\n\n"
            "Args:\n"
            "    delta_time: The time interval since the last time the function was called."));

  // Add the steering structures
  pybind11::class_<Vec2d>(game_objects, "Vec2d", "Represents a 2D vector.")
      .def(pybind11::init<double, double>(), pybind11::arg("x"), pybind11::arg("y"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    x: The x value of the vector.\n"
            "    y: The y value of the vector."))
      .def_readwrite("x", &Vec2d::x)
      .def_readwrite("y", &Vec2d::y)
      .def("__eq__", &Vec2d::operator==,
           ("Check if this vector is equal to another vector.\n\n"
            "Args:\n"
            "    other: The other vector to compare to.\n\n"
            "Returns:\n"
            "    Whether the vectors are equal or not."))
      .def("__ne__", &Vec2d::operator!=,
           ("Check if this vector is not equal to another vector.\n\n"
            "Args:\n"
            "    other: The other vector to compare to.\n\n"
            "Returns:\n"
            "    Whether the vectors are not equal or not."))
      .def("__add__", &Vec2d::operator+,
           ("Add another vector to this vector.\n\n"
            "Args:\n"
            "    other: The other vector to add.\n\n"
            "Returns:\n"
            "    The sum of the two vectors."))
      .def("__iadd__", &Vec2d::operator+=,
           ("Add another vector to this vector.\n\n"
            "Args:\n"
            "    other: The other vector to add.\n\n"
            "Returns:\n"
            "    This vector."))
      .def("__sub__", &Vec2d::operator-,
           ("Subtract another vector from this vector.\n\n"
            "Args:\n"
            "    other: The other vector to subtract.\n\n"
            "Returns:\n"
            "    The difference of the two vectors."))
      .def("__mul__", &Vec2d::operator*,
           ("Multiply this vector by a scalar.\n\n"
            "Args:\n"
            "    scalar: The scalar to multiply by.\n\n"
            "Returns:\n"
            "    The product of the vector and the scalar."))
      .def("__truediv__", &Vec2d::operator/,
           ("Divide this vector by a scalar.\n\n"
            "Args:\n"
            "    scalar: The scalar to divide by.\n\n"
            "Returns:\n"
            "    The quotient of the vector and the scalar."))
      .def("magnitude", &Vec2d::magnitude,
           ("Calculate the magnitude of the vector.\n\n"
            "Returns:\n"
            "    The magnitude of the vector."))
      .def("normalised", &Vec2d::normalised,
           ("Calculate the normalised vector.\n\n"
            "Returns:\n"
            "    The normalised vector."))
      .def("rotated", &Vec2d::rotated, pybind11::arg("angle"),
           ("Rotate the vector by an angle.\n\n"
            "Args:\n"
            "    angle: The angle to rotate by.\n\n"
            "Returns:\n"
            "    The rotated vector."))
      .def("angle_between", &Vec2d::angle_between, pybind11::arg("other"),
           ("Calculate the angle between this vector and another vector.\n\n"
            "Args:\n"
            "    other: The other vector to calculate the angle between.\n\n"
            "Returns:\n"
            "    The angle between the two vectors."))
      .def("distance_to", &Vec2d::distance_to, pybind11::arg("other"),
           ("Calculate the distance between this vector and another vector.\n\n"
            "Args:\n"
            "    other: The other vector to calculate the distance between.\n\n"
            "Returns:\n"
            "    The distance between the two vectors."));
  pybind11::class_<KinematicObject>(game_objects, "KinematicObject",
                                    "Stores various data about a game object for use in physics-related operations.")
      .def(pybind11::init<>(), "Initialise the object.")
      .def_readwrite("position", &KinematicObject::position)
      .def_readwrite("velocity", &KinematicObject::velocity)
      .def_readwrite("rotation", &KinematicObject::rotation);

  // Add the registry class
  register_exception<RegistryException>(game_objects, "RegistryException");
  pybind11::class_<Registry> registry_class(game_objects, "Registry",
                                            "Manages game objects, components, and systems that are registered.");
  registry_class.def(pybind11::init<>(), "Initialise the object.")
      .def("create_game_object", &Registry::create_game_object, pybind11::arg("kinematic") = false,
           ("Create a new game object.\n\n"
            "Args:\n"
            "    kinematic: Whether the game object should have a kinematic object or not.\n\n"
            "Returns:\n"
            "    The game object ID."))
      .def("delete_game_object", &Registry::delete_game_object, pybind11::arg("game_object_id"),
           ("Delete a game object.\n\n"
            "Args:\n"
            "    game_object_id: The game object ID.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object is not registered."))
      .def("has_component", &Registry::has_component, pybind11::arg("game_object_id"), pybind11::arg("component_type"),
           ("Checks if a game object has a given component or not.\n\n"
            "Args:\n"
            "    game_object_id: The game object ID.\n"
            "    component_type: The type of component to check for.\n\n"
            "Returns:\n"
            "    Whether the game object has the component or not."))
      .def("add_components", &Registry::add_components, pybind11::arg("game_object_id"), pybind11::arg("components"),
           ("Add multiple components to a game object.\n\n"
            "Args:\n"
            "    game_object_id: The game object ID.\n"
            "    components: The components to add to the game object.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object is not registered."))
      .def("update", &Registry::update, pybind11::arg("delta_time"),
           ("Update all systems in the registry.\n\n"
            "Args:\n"
            "    delta_time: The time interval since the last time the function was called."))
      .def("get_kinematic_object", &Registry::get_kinematic_object, pybind11::arg("game_object_id"),
           ("Get the kinematic object for a game object.\n\n"
            "Args:\n"
            "    game_object_id: The game object ID.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object is not registered or if the game object does not have a "
            "kinematic object.\n\n"
            "Returns:\n"
            "    The kinematic object."))
      .def("add_wall", &Registry::add_wall, pybind11::arg("wall"),
           ("Add a wall to the registry.\n\n"
            "Args:\n"
            "    wall: The wall to add to the registry."))
      .def("get_walls", &Registry::get_walls,
           ("Get the walls from the registry.\n\n"
            "Returns:\n"
            "    The walls in the registry."));
  bind_components<Armour, ArmourRegen, Attacks, EffectApplier, Footprints, Health, Inventory, KeyboardMovement, Money,
                  MovementForce, StatusEffects, SteeringMovement, Upgrades>(registry_class);
  bind_systems<ArmourRegenSystem, AttackSystem, DamageSystem, EffectSystem, FootprintSystem, InventorySystem,
               KeyboardMovementSystem, SteeringMovementSystem, UpgradeSystem>(registry_class);

  // Add the stat components
  pybind11::class_<Stat>(game_objects, "Stat", "Represents a component that has a variable value and maximum value.")
      .def(pybind11::init<double, int, bool>(), pybind11::arg("value"), pybind11::arg("maximum_level"),
           pybind11::arg("max_value") = true,
           ("Initialise the object.\n\n"
            "Args:\n"
            "    value: The initial and maximum value of the stat.\n"
            "    maximum_level: The maximum level of the stat.\n"
            "    max_value: Whether the stat has a maximum value or not."))
      .def_readwrite("max_value", &Stat::max_value)
      .def_readwrite("current_level", &Stat::current_level)
      .def_readwrite("maximum_level", &Stat::maximum_level)
      .def("get_value", &Stat::get_value,
           ("Get the value of the stat.\n\n"
            "Returns:\n"
            "    The value of the stat."))
      .def("set_value", &Stat::set_value,
           ("Set the value of the stat.\n\n"
            "Args:\n"
            "    new_value: The new value of the stat."));
  pybind11::class_<Armour, Stat>(game_objects, "Armour", "Allows a game object to have an armour stat.")
      .def(pybind11::init<double, int>(), pybind11::arg("value"), pybind11::arg("maximum_level"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    value: The initial and maximum value of the armour stat.\n"
            "    maximum_level: The maximum level of the armour stat."));
  pybind11::class_<ArmourRegen, Stat>(game_objects, "ArmourRegen", "Allows a game object to regenerate armour.")
      .def(pybind11::init<double, int>(), pybind11::arg("value"), pybind11::arg("maximum_level"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    value: The initial and maximum value of the armour regen stat.\n"
            "    maximum_level: The maximum level of the armour regen stat."))
      .def_readwrite("time_since_armour_regen", &ArmourRegen::time_since_armour_regen);
  pybind11::class_<Health, Stat>(game_objects, "Health", "Allows a game object to have a health stat.")
      .def(pybind11::init<double, int>(), pybind11::arg("value"), pybind11::arg("maximum_level"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    value: The initial and maximum value of the health stat.\n"
            "    maximum_level: The maximum level of the health stat."));
  pybind11::class_<MovementForce, Stat>(game_objects, "MovementForce",
                                        "Allows a game object to determine how fast it can move.")
      .def(pybind11::init<double, int>(), pybind11::arg("value"), pybind11::arg("maximum_level"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    value: The initial and maximum value of the movement force stat.\n"
            "    maximum_level: The maximum level of the movement force stat."));

  // Add the armour regen system as well as relevant structures/components
  pybind11::class_<ArmourRegenSystem, SystemBase>(systems, "ArmourRegenSystem",
                                                  "Provides facilities to manipulate armour regen components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    registry: The registry that manages the game objects, components, and systems."))
      .def("update", &ArmourRegenSystem::update, pybind11::arg("delta_time"),
           ("Process update logic for an armour regeneration component.\n\n"
            "Args:\n"
            "    delta_time: The time interval since the last time the function was called."));

  // Add the attack system and the damage system as well as relevant structures/components
  pybind11::enum_<AttackAlgorithms>(systems, "AttackAlgorithms",
                                    "Stores the different types of attack algorithms available.")
      .value("AreaOfEffect", AttackAlgorithms::AreaOfEffect)
      .value("Melee", AttackAlgorithms::Melee)
      .value("Ranged", AttackAlgorithms::Ranged);
  pybind11::class_<Attacks, ComponentBase>(systems, "Attacks", "Allows a game object to attack other game objects.")
      .def(pybind11::init<std::vector<AttackAlgorithms>>(), pybind11::arg("attack_algorithms"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    attack_algorithms: The attack algorithms the game object can use."))
      .def_readwrite("attack_algorithms", &Attacks::attack_algorithms)
      .def_readwrite("attack_state", &Attacks::attack_state);
  pybind11::class_<AttackSystem, SystemBase>(systems, "AttackSystem",
                                             "Provides facilities to manipulate attack components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    registry: The registry that manages the game objects, components, and systems."))
      .def("do_attack", &AttackSystem::do_attack, pybind11::arg("game_object_id"), pybind11::arg("targets"),
           ("Perform the currently selected attack algorithm.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to perform the attack for.\n"
            "    targets: The targets to attack.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object does not exist or does not have an attack component.\n\n"
            "Returns:\n"
            "    The result of the attack."))
      .def("previous_attack", &AttackSystem::previous_attack, pybind11::arg("game_object_id"),
           ("Select the previous attack algorithm.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to select the previous attack algorithm for.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object does not exist or does not have an attack component."))
      .def("next_attack", &AttackSystem::next_attack, pybind11::arg("game_object_id"),
           ("Select the next attack algorithm.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to select the next attack algorithm for.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object does not exist or does not have an attack component."));
  pybind11::class_<DamageSystem, SystemBase>(systems, "DamageSystem", "Provides facilities to damage game objects.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    registry: The registry that manages the game objects, components, and systems."))
      .def("deal_damage", &DamageSystem::deal_damage,
           ("Deal damage to a game object.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to deal damage to.\n"
            "    damage: The amount of damage to deal.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object does not exist or does not have a health component."));

  // Add the effect system as well as relevant structures/components
  pybind11::enum_<StatusEffectType>(systems, "StatusEffectType",
                                    "Stores the different types of status effects available.")
      .value("TEMP", StatusEffectType::TEMP)
      .value("TEMP2", StatusEffectType::TEMP2);
  pybind11::class_<StatusEffect>(systems, "StatusEffect",
                                 "Represents a status effect that can be applied to a game object.")
      .def(pybind11::init<double, double, double, std::type_index>(), pybind11::arg("value"), pybind11::arg("duration"),
           pybind11::arg("interval"), pybind11::arg("target_component"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    value: The value of the status effect.\n"
            "    duration: The duration of the status effect.\n"
            "    interval: The interval of the status effect.\n"
            "    target_component: The component the status effect should be applied to."))
      .def_readwrite("value", &StatusEffect::value)
      .def_readwrite("duration", &StatusEffect::duration)
      .def_readwrite("interval", &StatusEffect::interval)
      .def_readwrite("target_component", &StatusEffect::target_component)
      .def_readwrite("time_counter", &StatusEffect::time_counter)
      .def_readwrite("leftover_time", &StatusEffect::leftover_time);
  pybind11::class_<StatusEffectData>(systems, "StatusEffectData",
                                     "Represents the data required to apply a status effect.")
      .def(pybind11::init<StatusEffectType, ActionFunction, ActionFunction, ActionFunction>(),
           pybind11::arg("status_effect_type"), pybind11::arg("increase"), pybind11::arg("duration"),
           pybind11::arg("interval"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    status_effect_type: The type of status effect.\n"
            "    increase: The increase function to apply.\n"
            "    duration: The duration function to apply.\n"
            "    interval: The interval function to apply."))
      .def_readwrite("status_effect_type", &StatusEffectData::status_effect_type)
      .def_readwrite("increase", &StatusEffectData::increase)
      .def_readwrite("duration", &StatusEffectData::duration)
      .def_readwrite("interval", &StatusEffectData::interval);
  pybind11::class_<EffectApplier, ComponentBase>(systems, "EffectApplier",
                                                 "Allows a game object to provide instant or status effects.")
      .def(pybind11::init<std::unordered_map<std::type_index, ActionFunction>,
                          std::unordered_map<std::type_index, StatusEffectData>>(),
           pybind11::arg("instant_effects"), pybind11::arg("status_effects"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    instant_effects: The instant effects the game object provides.\n"
            "    status_effects: The status effects the game object provides."))
      .def_readwrite("instant_effects", &EffectApplier::instant_effects)
      .def_readwrite("status_effects", &EffectApplier::status_effects);
  pybind11::class_<StatusEffects, ComponentBase>(systems, "StatusEffects",
                                                 "Allows a game object to have status effects applied to it.")
      .def(pybind11::init<>(), "Initialise the object.")
      .def_readwrite("applied_effects", &StatusEffects::applied_effects);
  pybind11::class_<EffectSystem, SystemBase>(systems, "EffectSystem",
                                             "Provides facilities to manipulate instant and status effects.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    registry: The registry that manages the game objects, components, and systems."))
      .def("update", &EffectSystem::update, pybind11::arg("delta_time"),
           ("Process update logic for a status effect component.\n\n"
            "Args:\n"
            "    delta_time: The time interval since the last time the function was called."))
      .def("apply_instant_effect", &EffectSystem::apply_instant_effect, pybind11::arg("game_object_id"),
           pybind11::arg("target_component"), pybind11::arg("increase"), pybind11::arg("level"),
           ("Apply an instant effect to a game object.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to apply the effect to.\n"
            "    target_component: The component to apply the effect to.\n"
            "    increase: The increase function to apply.\n"
            "    level: The level of the effect to apply.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object does not exist or does not have the target component.\n\n"
            "Returns:\n"
            "    Whether the instant effect was applied or not."))
      .def("apply_status_effect", &EffectSystem::apply_status_effect, pybind11::arg("game_object_id"),
           pybind11::arg("target_component"), pybind11::arg("status_effect_data"), pybind11::arg("level"),
           ("Apply a status effect to a game object.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to apply the effect to.\n"
            "    target_component: The component to apply the effect to.\n"
            "    status_effect_data: The data required to apply the status effect.\n"
            "    level: The level of the effect to apply.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object does not exist or does not have the target component.\n\n"
            "Returns:\n"
            "    Whether the status effect was applied or not."));

  // Add the inventory system as well as relevant structures/components
  register_exception<InventorySpaceException>(systems, "InventorySpaceException");
  pybind11::class_<Inventory, ComponentBase>(systems, "Inventory",
                                             "Allows a game object to have a fixed size inventory.")
      .def(pybind11::init<int, int>(), pybind11::arg("width"), pybind11::arg("height"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    width: The width of the inventory.\n"
            "    height: The height of the inventory."))
      .def_readwrite("width", &Inventory::width)
      .def_readwrite("height", &Inventory::height)
      .def_readwrite("items", &Inventory::items)
      .def("get_capacity", &Inventory::get_capacity,
           ("Get the capacity of the inventory.\n\n"
            "Returns:\n"
            "    The capacity of the inventory."));
  pybind11::class_<InventorySystem, SystemBase>(systems, "InventorySystem",
                                                "Provides facilities to manipulate inventory components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    registry: The registry that manages the game objects, components, and systems."))
      .def("add_item_to_inventory", &InventorySystem::add_item_to_inventory, pybind11::arg("game_object_id"),
           pybind11::arg("item"),
           ("Add an item to the inventory of a game object.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to add the item to.\n"
            "    item: The item to add to the inventory.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object does not exist or does not have an inventory component.\n"
            "    InventorySpaceException: If the inventory is full."))
      .def("remove_item_from_inventory", &InventorySystem::remove_item_from_inventory, pybind11::arg("game_object_id"),
           pybind11::arg("index"),
           ("Remove an item from the inventory of a game object.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to remove the item from.\n"
            "    index: The index of the item to remove from the inventory.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object does not exist or does not have an inventory component.\n"
            "    InventorySpaceException: If the inventory is empty or if the index is out of bounds."));

  // Add the footprint system, the keyboard movement system, and the steering movement system as well as relevant
  // structures/components
  pybind11::enum_<SteeringBehaviours>(systems, "SteeringBehaviours",
                                      "Stores the different types of steering behaviours available.")
      .value("Arrive", SteeringBehaviours::Arrive)
      .value("Evade", SteeringBehaviours::Evade)
      .value("Flee", SteeringBehaviours::Flee)
      .value("FollowPath", SteeringBehaviours::FollowPath)
      .value("ObstacleAvoidance", SteeringBehaviours::ObstacleAvoidance)
      .value("Pursuit", SteeringBehaviours::Pursuit)
      .value("Seek", SteeringBehaviours::Seek)
      .value("Wander", SteeringBehaviours::Wander);
  pybind11::enum_<SteeringMovementState>(systems, "SteeringMovementState",
                                         "Stores the different states the steering movement component can be in.")
      .value("Default", SteeringMovementState::Default)
      .value("Footprint", SteeringMovementState::Footprint)
      .value("Target", SteeringMovementState::Target);
  pybind11::class_<Footprints, ComponentBase>(
      systems, "Footprints", "Allows a game object to periodically leave footprints around the game map.")
      .def(pybind11::init<>(), "Initialise the object.")
      .def_readwrite("footprints", &Footprints::footprints)
      .def_readwrite("time_since_last_footprint", &Footprints::time_since_last_footprint);
  pybind11::class_<KeyboardMovement, ComponentBase>(systems, "KeyboardMovement",
                                                    "Allows a game object's movement to be controlled by the keyboard.")
      .def(pybind11::init<>(), "Initialise the object.")
      .def_readwrite("moving_north", &KeyboardMovement::moving_north)
      .def_readwrite("moving_east", &KeyboardMovement::moving_east)
      .def_readwrite("moving_south", &KeyboardMovement::moving_south)
      .def_readwrite("moving_west", &KeyboardMovement::moving_west);
  pybind11::class_<SteeringMovement, ComponentBase>(
      systems, "SteeringMovement", "Allows a game object's movement to be controlled by steering behaviours.")
      .def(pybind11::init<std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>>>(),
           pybind11::arg("behaviours"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    behaviours: The steering behaviours the game object can use."))
      .def_readwrite("behaviours", &SteeringMovement::behaviours)
      .def_readwrite("movement_state", &SteeringMovement::movement_state)
      .def_readwrite("target_id", &SteeringMovement::target_id)
      .def_readwrite("path_list", &SteeringMovement::path_list);
  pybind11::class_<FootprintSystem, SystemBase>(systems, "FootprintSystem",
                                                "Provides facilities to manipulate footprint components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    registry: The registry that manages the game objects, components, and systems."))
      .def("update", &FootprintSystem::update, pybind11::arg("delta_time"),
           ("Process update logic for a footprint component.\n\n"
            "Args:\n"
            "    delta_time: The time interval since the last time the function was called."));
  pybind11::class_<KeyboardMovementSystem, SystemBase>(
      systems, "KeyboardMovementSystem", "Provides facilities to manipulate keyboard movement components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    registry: The registry that manages the game objects, components, and systems."))
      .def("calculate_keyboard_force", &KeyboardMovementSystem::calculate_keyboard_force,
           pybind11::arg("game_object_id"),
           ("Calculate the new keyboard force to apply to the game object.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to calculate the keyboard force for.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object does not exist or does not have a keyboard movement "
            "component.\n\n"
            "Returns:\n"
            "    The new force to apply to the game object."));
  pybind11::class_<SteeringMovementSystem, SystemBase>(
      systems, "SteeringMovementSystem", "Provides facilities to manipulate steering movement components.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    registry: The registry that manages the game objects, components, and systems."))
      .def("calculate_steering_force", &SteeringMovementSystem::calculate_steering_force,
           pybind11::arg("game_object_id"),
           ("Calculate the new steering force to apply to the game object.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to calculate the steering force for.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object does not exist or does not have a steering movement "
            "component.\n\n"
            "Returns:\n"
            "    The new force to apply to the game object."))
      .def("update_path_list", &SteeringMovementSystem::update_path_list, pybind11::arg("game_object_id"),
           pybind11::arg("path_list"),
           ("Update the path lists for the game objects to follow.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to follow.\n"
            "    path_list: The list of footprints to follow."));

  // Add the upgrade system as well as relevant structures/components
  pybind11::class_<Money, ComponentBase>(systems, "Money", "Allows a game object to record the amount of money it has.")
      .def(pybind11::init<int>(), pybind11::arg("money"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    money: The amount of money the game object has."))
      .def_readwrite("money", &Money::money);
  pybind11::class_<Upgrades, ComponentBase>(systems, "Upgrades", "Allows a game object to be upgraded.")
      .def(pybind11::init<std::unordered_map<std::type_index, ActionFunction>>(), pybind11::arg("upgrades"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    upgrades: The upgrades the game object has."))
      .def_readwrite("upgrades", &Upgrades::upgrades);
  pybind11::class_<UpgradeSystem, SystemBase>(systems, "UpgradeSystem",
                                              "Provides facilities to manipulate game object upgrades.")
      .def(pybind11::init<Registry *>(), pybind11::arg("registry"),
           ("Initialise the object.\n\n"
            "Args:\n"
            "    registry: The registry that manages the game objects, components, and systems."))
      .def("upgrade_component", &UpgradeSystem::upgrade_component, pybind11::arg("game_object_id"),
           pybind11::arg("target_component"),
           ("Upgrade a component to the next level if possible.\n\n"
            "Args:\n"
            "    game_object_id: The ID of the game object to upgrade the component for.\n"
            "    target_component: The type of component to upgrade.\n\n"
            "Raises:\n"
            "    RegistryException: If the game object does not exist or does not have the target component.\n\n"
            "Returns:\n"
            "    Whether the component upgrade was successful or not."));
}
