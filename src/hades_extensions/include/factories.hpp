// Ensure this file is only included once
#pragma once

// Std headers
#include <optional>
#include <stdexcept>
#include <vector>

// Local headers
#include "ecs/chipmunk.hpp"
#include "game_object.hpp"

// Forward declarations
struct ComponentBase;
class Registry;

/// Calculate the screen position based on a grid position.
///
/// @param position - The position in the grid.
/// @throws std::invalid_argument - If the position is negative.
/// @return The screen position of the grid position.
inline auto grid_pos_to_pixel(const cpVect& position) -> cpVect {
  if (position.x < 0 || position.y < 0) {
    throw std::invalid_argument("The position cannot be negative.");
  }
  return position * SPRITE_SIZE + SPRITE_SIZE / 2;
}

/// Load a hitbox for a given game object type.
///
/// @param game_object_type - The game object type.
/// @param hitbox - The hitbox to load.
/// @return Whether the hitbox was loaded or not.
auto load_hitbox(GameObjectType game_object_type, const std::vector<std::pair<double, double>>& hitbox) -> bool;

/// Clear all hitboxes.
void clear_hitboxes();

/// Create a game object if possible.
///
/// @param registry - The registry to create the game object in.
/// @param game_object_type - The type of game object to create.
/// @param position - The position to create the game object at.
/// @param level - The level of the game object, if applicable.
/// @return The ID of the created game object.
auto create_game_object(Registry* registry, GameObjectType game_object_type, const cpVect& position,
                        std::optional<int> level = {}) -> GameObjectID;
