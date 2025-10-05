// External headers
#include <gtest/gtest.h>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/armour_regen.hpp"

/// Implements the fixture for the ArmourRegenSystem tests.
class ArmourRegenSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    const auto game_object_id{registry.create_game_object(GameObjectType::Player)};
    registry.add_component<Armour>(game_object_id, 50);
    registry.add_component<ArmourRegen>(game_object_id);
    registry.add_system<ArmourRegenSystem>();
  }

  /// Get the armour regen system from the registry.
  ///
  /// @return The armour regen system.
  [[nodiscard]] auto get_armour_regen_system() const -> std::shared_ptr<ArmourRegenSystem> {
    return registry.get_system<ArmourRegenSystem>();
  }
};

/// Test that the armour regen component is updated correctly when armour is full.
TEST_F(ArmourRegenSystemFixture, TestArmourRegenSystemUpdateFullArmour) {
  get_armour_regen_system()->update(5);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 50);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 0);
}

/// Test that the armour regen component is updated with a small delta time.
TEST_F(ArmourRegenSystemFixture, TestArmourRegenSystemUpdateSmallDeltaTime) {
  const auto armour{registry.get_component<Armour>(0)};
  armour->set_value(40);
  get_armour_regen_system()->update(0.5);
  ASSERT_EQ(armour->get_value(), 40);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 0.5);
}

/// Test that the armour regen component is updated with a large delta time.
TEST_F(ArmourRegenSystemFixture, TestArmourRegenSystemUpdateLargeDeltaTime) {
  const auto armour{registry.get_component<Armour>(0)};
  armour->set_value(armour->get_value() - 10);
  get_armour_regen_system()->update(6);
  ASSERT_EQ(armour->get_value(), 41);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 0);
}

/// Test that the armour regen component is updated multiple times correctly.
TEST_F(ArmourRegenSystemFixture, TestArmourRegenSystemUpdateMultipleUpdates) {
  const auto armour{registry.get_component<Armour>(0)};
  armour->set_value(armour->get_value() - 10);
  get_armour_regen_system()->update(0.2);
  ASSERT_EQ(armour->get_value(), 40);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 0.2);
  get_armour_regen_system()->update(0.5);
  ASSERT_EQ(armour->get_value(), 40);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 0.7);
}

/// Test that the armour regen component is not updated if the game object does not have the required components.
TEST_F(ArmourRegenSystemFixture, TestArmourRegenSystemUpdateNoArmourRegen) {
  const auto game_object_id{registry.create_game_object(GameObjectType::Player)};
  registry.add_component<Armour>(game_object_id, 100);
  registry.get_component<Armour>(1)->set_value(50);
  get_armour_regen_system()->update(5);
  ASSERT_EQ(registry.get_component<Armour>(1)->get_value(), 50);
}
