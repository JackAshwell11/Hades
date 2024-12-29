// Ensure this file is only included once
#pragma once

// Std headers
#include <memory>
#include <unordered_map>
#include <vector>

// Local headers
#include "ecs/types.hpp"

// Avoid having to include headers for this
struct cpVect;

/// Alias for a factory function that creates components for a game object with a given level.
using ComponentFactory = std::function<std::vector<std::shared_ptr<ComponentBase>>(int)>;

/// Load a hitbox for a given game object type.
///
/// @param game_object_type - The game object type.
/// @param hitbox - The hitbox to load.
/// @return Whether the hitbox was loaded or not.
auto load_hitbox(GameObjectType game_object_type, const std::vector<std::pair<double, double>> &hitbox) -> bool;

/// Clear all hitboxes.
void clear_hitboxes();

/// Get the map of game object types to their respective component factories.
///
/// @return The map of game object types to component factories.
auto get_factories() -> const std::unordered_map<GameObjectType, ComponentFactory> &;
