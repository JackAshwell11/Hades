// Ensure this file is only included once
#pragma once

// Std headers
#include <memory>
#include <unordered_map>
#include <vector>

// Local headers
#include "ecs/types.hpp"

/// Alias for a factory function that creates components for a game object.
using ComponentFactory = std::function<std::vector<std::shared_ptr<ComponentBase>>()>;

/// Get the map of game object types to their respective component factories.
///
/// @return The map of game object types to component factories.
auto get_factories() -> const std::unordered_map<GameObjectType, ComponentFactory>&;
