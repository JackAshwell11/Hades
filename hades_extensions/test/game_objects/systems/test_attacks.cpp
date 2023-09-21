// External includes
#include "gtest/gtest.h"

// Custom includes
#include "macros.hpp"
#include "game_objects/stats.hpp"
#include "game_objects/systems/attacks.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the AttackSystem tests.
class AttackSystemFixture : public testing::Test {
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

    // Create the targets and add the attacks system
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
    registry.add_system(std::make_shared<AttackSystem>(registry));
    registry.add_system(std::make_shared<DamageSystem>(registry));
  }

  /// Create an attacks component for a game object.
  ///
  /// @tparam T - The type of the component or system.
  /// @param list - The initializer list to pass to the constructor.
  void create_attack_component(const std::vector<AttackAlgorithms> &enabled_attacks) {
    std::vector<std::unique_ptr<ComponentBase>> components;
    components.push_back(std::make_unique<Attacks>(enabled_attacks));
    int game_object_id = registry.create_game_object(true, std::move(components));
    registry.get_kinematic_object(game_object_id)->rotation = 180;
  }

  /// Get the attacks system from the registry.
  ///
  /// @return The attacks system.
  std::shared_ptr<AttackSystem> get_attacks_system() {
    return registry.find_system<AttackSystem>();
  }
};

/// Implements the fixture for the DamageSystem tests.
class DamageSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.add_system(std::make_shared<DamageSystem>(registry));
  }

  /// Create health and armour attributes for use in testing.
  void create_health_and_armour_attributes() {
    std::vector<std::unique_ptr<ComponentBase>> components;
    components.push_back(std::make_unique<Health>(300, -1));
    components.push_back(std::make_unique<Armour>(100, -1));
    registry.create_game_object(false, std::move(components));
  }

  /// Get the damage system from the registry.
  ///
  /// @return The damage system.
  std::shared_ptr<DamageSystem> get_damage_system() {
    return registry.find_system<DamageSystem>();
  }
};

// ----- TESTS ----------------------------------
/// Test that performing an area of effect attack works correctly.
TEST_F(AttackSystemFixture, TestAttacksDoAreaOfEffectAttack) {
  create_attack_component({AttackAlgorithms::AreaOfEffect});
  get_attacks_system()->do_attack(8, targets);
  ASSERT_EQ(registry.get_component<Health>(targets[0])->get_value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[1])->get_value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[2])->get_value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[3])->get_value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[4])->get_value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[5])->get_value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[6])->get_value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[7])->get_value(), 40);
}

/// Test that performing a melee attack works correctly.
TEST_F(AttackSystemFixture, TestAttacksDoMeleeAttack) {
  create_attack_component({AttackAlgorithms::Melee});
  get_attacks_system()->do_attack(8, targets);
  ASSERT_EQ(registry.get_component<Health>(targets[0])->get_value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[1])->get_value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[2])->get_value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[3])->get_value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[4])->get_value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[5])->get_value(), 50);
  ASSERT_EQ(registry.get_component<Health>(targets[6])->get_value(), 40);
  ASSERT_EQ(registry.get_component<Health>(targets[7])->get_value(), 40);
}

/// Test that performing a ranged attack works correctly.
TEST_F(AttackSystemFixture, TestAttacksDoRangedAttack) {
  // This is due to floating point precision
  create_attack_component({AttackAlgorithms::Ranged});
  std::tuple<Vec2d, double, double> attack_result = get_attacks_system()->do_attack(8, targets).ranged_attack.value();
  ASSERT_EQ(get<0>(attack_result), Vec2d(0, 0));
  ASSERT_EQ(get<1>(attack_result), -300);
  ASSERT_NEAR(get<2>(attack_result), 0, 1e-13);
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(AttackSystemFixture, TestAttacksDoAttackInvalidGameObjectId) {
  create_attack_component({AttackAlgorithms::Ranged});
  ASSERT_THROW_MESSAGE(get_attacks_system()->do_attack(-1, targets),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}

/// Test that switching between attacks once works correctly.
TEST_F(AttackSystemFixture, TestAttacksPreviousNextAttackSingle) {
  create_attack_component({AttackAlgorithms::AreaOfEffect, AttackAlgorithms::Melee, AttackAlgorithms::Ranged});
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
  get_attacks_system()->next_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 1);
  get_attacks_system()->previous_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
}

/// Test that switching between attacks multiple times works correctly.
TEST_F(AttackSystemFixture, TestAttacksPreviousAttackMultiple) {
  create_attack_component({AttackAlgorithms::AreaOfEffect, AttackAlgorithms::Melee, AttackAlgorithms::Ranged});
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
  get_attacks_system()->next_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 1);
  get_attacks_system()->next_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 2);
  get_attacks_system()->next_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 2);
  get_attacks_system()->previous_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 1);
  get_attacks_system()->previous_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
  get_attacks_system()->previous_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
}

/// Test that changing the attack state works correctly when there are no attacks.
TEST_F(AttackSystemFixture, TestAttacksPreviousNextAttackEmptyAttacks) {
  create_attack_component({});
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
  get_attacks_system()->next_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
  get_attacks_system()->previous_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(AttackSystemFixture, TestAttacksPreviousNextAttackInvalidGameObjectId) {
  create_attack_component({});
  ASSERT_THROW_MESSAGE(get_attacks_system()->next_attack(-1),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
  ASSERT_THROW_MESSAGE(get_attacks_system()->previous_attack(-1),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}

/// Test that damage is dealt when health and armour are lower than damage.
TEST_F(DamageSystemFixture, TestGameObjectAttributeSystemDealDamageLowHealthArmour) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, 350);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 50);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 0);
}

/// Test that no damage is dealt when armour is larger than damage.
TEST_F(DamageSystemFixture, TestGameObjectAttributeSystemDealDamageLargeArmour) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, 50);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 50);
}

/// Test that no damage is dealt when damage is zero.
TEST_F(DamageSystemFixture, TestGameObjectAttributeSystemDealDamageZeroDamage) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, 0);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 100);
}

/// Test that damage is dealt when armour is zero.
TEST_F(DamageSystemFixture, TestGameObjectAttributeSystemDealDamageZeroArmour) {
  create_health_and_armour_attributes();
  auto armour = registry.get_component<Armour>(0);
  armour->set_value(0);
  get_damage_system()->deal_damage(0, 100);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 200);
  ASSERT_EQ(armour->get_value(), 0);
}

/// Test that damage is dealt when health is zero.
TEST_F(DamageSystemFixture, TestGameObjectAttributeSystemDealDamageZeroHealth) {
  create_health_and_armour_attributes();
  auto health = registry.get_component<Health>(0);
  health->set_value(0);
  get_damage_system()->deal_damage(0, 50);
  ASSERT_EQ(health->get_value(), 0);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 50);
}

/// Test that no damage is dealt when the attributes are not initialised.
TEST_F(DamageSystemFixture, TestGameObjectAttributeSystemDealDamageNonexistentAttributes) {
  registry.create_game_object(false, {});
  ASSERT_THROW_MESSAGE(get_damage_system()->deal_damage(0, 100),
                       RegistryException,
                       "The game object `0` is not registered with the registry.")
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(DamageSystemFixture, TestGameObjectAttributeSystemDealDamageInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(get_damage_system()->deal_damage(-1, 100),
                       RegistryException,
                       "The game object `-1` is not registered with the registry.")
}
