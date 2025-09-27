// Ensure this file is only included once
#pragma once

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/physics.hpp"
#include "game_state.hpp"

/// Get an item from the registry.
///
/// @param registry - The registry to search for the item.
/// @param game_object_type - The type of the item to find.
/// @return The game object ID of the item.
[[nodiscard]] inline auto get_item(const std::shared_ptr<Registry>& registry, const GameObjectType game_object_type)
    -> GameObjectID {
  return registry->get_game_object_ids(game_object_type).front();
}

/// Move the player to the position of an item.
///
/// @param registry - The registry containing the player and item components.
/// @param game_state - The game state to update after moving the player.
/// @param game_object_type - The type of the item to move the player to.
/// @param update - Whether to update the physics system after moving the player.
/// @return The game object ID of the item.
inline auto move_player_to_item(const std::shared_ptr<Registry>& registry, const std::shared_ptr<GameState>& game_state,
                                const GameObjectType game_object_type, const bool update = true) -> GameObjectID {
  const auto item_id{get_item(registry, game_object_type)};
  const auto item_pos{cpBodyGetPosition(*registry->get_component<KinematicComponent>(item_id)->body)};
  cpBodySetPosition(*registry->get_component<KinematicComponent>(0)->body, item_pos);
  if (update) {
    registry->get_system<PhysicsSystem>()->update(0);
    game_state->set_nearest_item(registry->get_system<PhysicsSystem>()->get_nearest_item(game_state->get_player_id()));
  }
  return item_id;
}
