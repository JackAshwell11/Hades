// External includes
#include "gtest/gtest.h"

// Custom includes
#include "game_objects/stats.hpp"
#include "game_objects/systems/armour_regen.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the ArmourRegenSystem tests.
class ArmourRegenSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    std::vector<std::unique_ptr<ComponentBase>> components;
    components.push_back(std::make_unique<Armour>(50, -1));
    components.push_back(std::make_unique<ArmourRegen>(4, -1));
    registry.create_game_object(false, std::move(components));
    registry.add_system(std::make_shared<ArmourRegenSystem>(registry));
  }

  /// Get the armour regen system from the registry.
  ///
  /// @return The armour regen system.
  std::shared_ptr<ArmourRegenSystem> get_armour_regen_system() {
    return registry.find_system<ArmourRegenSystem>();
  }
};

// ----- TESTS ----------------------------------
/// Test that the armour regen component is updated correctly when armour is full.
TEST_F(ArmourRegenSystemFixture, TestArmourRegenSystemUpdateFullArmour) {
  get_armour_regen_system()->update(5);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 50);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 0);
}

/// Test that the armour regen component is updated with a small delta time.
TEST_F(ArmourRegenSystemFixture, TestArmourRegenSystemUpdateSmallDeltaTime) {
  auto armour = registry.get_component<Armour>(0);
  armour->set_value(armour->get_value() - 10);
  get_armour_regen_system()->update(2);
  ASSERT_EQ(armour->get_value(), 40);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 2);
}

/// Test that the armour regen component is updated with a large delta time.
TEST_F(ArmourRegenSystemFixture, TestArmourRegenSystemUpdateLargeDeltaTime) {
  auto armour = registry.get_component<Armour>(0);
  armour->set_value(armour->get_value() - 10);
  get_armour_regen_system()->update(6);
  ASSERT_EQ(armour->get_value(), 41);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 0);
}

/// Test that the armour regen component is updated multiple times correctly.
TEST_F(ArmourRegenSystemFixture, TestArmourRegenSystemUpdateMultipleUpdates) {
  auto armour = registry.get_component<Armour>(0);
  armour->set_value(armour->get_value() - 10);
  get_armour_regen_system()->update(1);
  ASSERT_EQ(armour->get_value(), 40);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 1);
  get_armour_regen_system()->update(2);
  ASSERT_EQ(armour->get_value(), 40);
  ASSERT_EQ(registry.get_component<ArmourRegen>(0)->time_since_armour_regen, 3);
}
