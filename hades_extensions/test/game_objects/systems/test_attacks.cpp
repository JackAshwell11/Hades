// External includes
#include "gtest/gtest.h"

// Custom includes
#include "macros.hpp"
#include "game_objects/systems/attacks.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the game_objects/systems/attacks.hpp tests.
class AttacksFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// A list of targets for use in testing.
  std::vector<int> targets{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    // The lambda function for creating a target
    auto create_target = [&](Vec2d position) {
      std::vector<std::unique_ptr<ComponentBase>> components;
      components.push_back(std::make_unique<Health>(50, -1));
      components.push_back(std::make_unique<Armour>(0, -1));
      int target = registry.create_game_object(true, std::move(components));
      registry.get_kinematic_object(target)->position = position;
      return target;
    };
    targets = {
        create_target({-20, -100}),
        create_target({20, 60}),
        create_target({-200, 100}),
        create_target({100, -100}),
        create_target({-100, -99}),
        create_target({0, -200}),
        create_target({0, -192}),
        create_target({0, 0}),
    };
  }

  /// Create an attacks component for a game object.
  ///
  /// @tparam T - The type of the component or system.
  /// @param list - The initializer list to pass to the constructor.
  /// @return A unique pointer to the component or system.
  void create_attack_component(const std::vector<AttackAlgorithms> &enabled_attacks) {
    std::vector<std::unique_ptr<ComponentBase>> components;
    components.push_back(std::make_unique<Attacks>(enabled_attacks));
    int game_object_id = registry.create_game_object(true, std::move(components));
    registry.get_kinematic_object(game_object_id)->rotation = 180;
  }
};

// ----- TESTS ----------------------------------
/// Test that performing an area of effect attack works correctly.
TEST_F(AttacksFixture, TestAttacksDoAreaOfEffectAttack) {
  create_attack_component({AttackAlgorithms::AreaOfEffect});
  AttackSystem::do_attack(registry, 8, targets);
  ASSERT_EQ(registry.get_component<Health>(targets[0])->value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[1])->value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[2])->value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[3])->value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[4])->value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[5])->value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[6])->value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[7])->value(), 40);
}

/// Test that performing a melee attack works correctly.
TEST_F(AttacksFixture, TestAttacksDoMeleeAttack) {
  create_attack_component({AttackAlgorithms::Melee});
  AttackSystem::do_attack(registry, 8, targets);
  ASSERT_EQ(registry.get_component<Health>(targets[0])->value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[1])->value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[2])->value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[3])->value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[4])->value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[5])->value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[6])->value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[7])->value(), 40);
}

/// Test that performing a ranged attack works correctly.
TEST_F(AttacksFixture, TestAttacksDoRangedAttack) {
  // This is due to floating point precision
  create_attack_component({AttackAlgorithms::Ranged});
  std::tuple<Vec2d, double, double> attack_result = AttackSystem::do_attack(registry, 8, targets).ranged_attack.value();
  ASSERT_EQ(get<0>(attack_result), Vec2d(0, 0));
  ASSERT_EQ(get<1>(attack_result), -300);
  ASSERT_NEAR(get<2>(attack_result), 0, 1e-13);  // TODO: Improve this
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(AttacksFixture, TestAttacksDoAttackInvalidGameObjectId) {
  create_attack_component({AttackAlgorithms::Ranged});
  ASSERT_THROW_MESSAGE(AttackSystem::do_attack(registry, -1, targets),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}

/// Test that switching between attacks once works correctly.
TEST_F(AttacksFixture, TestAttacksPreviousNextAttackSingle) {
  create_attack_component({AttackAlgorithms::AreaOfEffect, AttackAlgorithms::Melee, AttackAlgorithms::Ranged});
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
  AttackSystem::next_attack(registry, 8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 1);
  AttackSystem::previous_attack(registry, 8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
}

/// Test that switching between attacks multiple times works correctly.
TEST_F(AttacksFixture, TestAttacksPreviousAttackMultiple) {
  create_attack_component({AttackAlgorithms::AreaOfEffect, AttackAlgorithms::Melee, AttackAlgorithms::Ranged});
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
  AttackSystem::next_attack(registry, 8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 1);
  AttackSystem::next_attack(registry, 8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 2);
  AttackSystem::next_attack(registry, 8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 2);
  AttackSystem::previous_attack(registry, 8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 1);
  AttackSystem::previous_attack(registry, 8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
  AttackSystem::previous_attack(registry, 8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
}

/// Test that changing the attack state works correctly when there are no attacks.
TEST_F(AttacksFixture, TestAttacksPreviousNextAttackEmptyAttacks) {
  create_attack_component({});
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
  AttackSystem::next_attack(registry, 8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
  AttackSystem::previous_attack(registry, 8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(AttacksFixture, TestAttacksPreviousNextAttackInvalidGameObjectId) {
  create_attack_component({});
  ASSERT_THROW_MESSAGE(AttackSystem::next_attack(registry, -1),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
  ASSERT_THROW_MESSAGE(AttackSystem::previous_attack(registry, -1),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}
