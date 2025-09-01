// Ensure this file is only included once
#pragma once

// Std headers
#include <memory>
#include <vector>

// Forward declarations
struct ComponentBase;
enum class GameObjectType : std::uint16_t;

/// Load a hitbox for a given game object type.
///
/// @param game_object_type - The game object type.
/// @param hitbox - The hitbox to load.
/// @return Whether the hitbox was loaded or not.
auto load_hitbox(GameObjectType game_object_type, const std::vector<std::pair<double, double>>& hitbox) -> bool;

/// Clear all hitboxes.
void clear_hitboxes();

/// Get the components for a game object type.
///
/// @param game_object_type - The game object type.
/// @param level - The level of the game object, default is 0.
/// @return The components for the game object type.
auto get_game_object_components(GameObjectType game_object_type, int level = 0)
    -> std::vector<std::shared_ptr<ComponentBase>>;
