// External headers
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/systems/level.hpp"

/// Implements the fixture for the ecs/systems/level.hpp tests.
class PlayerLevelFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    const auto game_object_id{registry.create_game_object(GameObjectType::Player)};
    registry.add_component<PlayerLevel>(game_object_id);
  }
};

/// Test that the player level component is serialised to a file correctly.
TEST_F(PlayerLevelFixture, TestPlayerLevelToFile) {
  const auto player_level{registry.get_component<PlayerLevel>(0)};
  player_level->level = 2;
  player_level->experience = 100.0;
  nlohmann::json json;
  player_level->to_file(json);
  ASSERT_EQ(json, nlohmann::json::parse(R"({"player_level":2,"experience":100.0})"));
}

/// Test that the player level component is deserialised from a file correctly.
TEST_F(PlayerLevelFixture, TestPlayerLevelFromFile) {
  const nlohmann::json json(nlohmann::json::parse(R"({"player_level":3,"experience":200.0})"));
  const auto player_level{registry.get_component<PlayerLevel>(0)};
  player_level->from_file(json);
  ASSERT_EQ(player_level->level, 3);
  ASSERT_EQ(player_level->experience, 200.0);
}
