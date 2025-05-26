// Ensure this file is only included once
#pragma once

// External headers
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
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/physics.hpp"
#include "ecs/systems/sprite.hpp"
#include "ecs/systems/upgrade.hpp"

// The declarations for the binding functions
void bind_ecs(const pybind11::module &module);
void bind_components(const pybind11::module &module);
void bind_systems(const pybind11::module_ &module);

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
  return {{pybind11::type::of<Ts>(), [](const Registry &registry) { return registry.get_system<Ts>(); }}...};
}

/// Get the component types mapping.
///
/// @return The component types mapping.
inline auto get_component_types()
    -> const std::unordered_map<pybind11::handle, std::type_index, py_handle_hash, py_handle_equal> & {
  static const auto component_types{
      make_component_types<Armour, Health, KinematicComponent, Money, PythonSprite, Upgrades>()};
  return component_types;
}

/// Get the system types mapping.
///
/// @return The system types mapping.
inline auto get_system_types()
    -> const std::unordered_map<pybind11::handle, std::function<std::shared_ptr<SystemBase>(const Registry &)>,
                                py_handle_hash, py_handle_equal> & {
  static const auto system_types{make_system_types<PhysicsSystem, UpgradeSystem>()};
  return system_types;
}

/// Get the type index for a given component type.
///
/// @param component_type - The component type.
/// @throws std::runtime_error - If the component type is invalid.
/// @return The type index for the component type.
inline auto get_type_index(const pybind11::handle &component_type) -> std::type_index {
  const auto &component_types{get_component_types()};
  const auto iter{component_types.find(component_type)};
  if (iter == component_types.end()) {
    throw std::runtime_error("Invalid component type provided.");
  }
  return iter->second;
}

/// Get the Python type for a given component type index.
///
/// @param type_index - The type index.
/// @throws std::runtime_error - If the type index is invalid.
/// @return The Python type for the component type index.
inline auto get_python_type(const std::type_index type_index) -> pybind11::handle {
  for (const auto &component_types{get_component_types()}; const auto &[handle, index] : component_types) {
    if (index == type_index) {
      return handle;
    }
  }
  throw std::runtime_error("Invalid type index provided.");
}
