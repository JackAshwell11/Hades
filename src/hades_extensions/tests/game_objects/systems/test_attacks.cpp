// Local headers
#include "game_objects/stats.hpp"
#include "game_objects/systems/attacks.hpp"
#include "macros.hpp"

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
    auto create_target{[&](Vec2d position) {
      const int target{
          registry.create_game_object({std::make_shared<Health>(50, -1), std::make_shared<Armour>(0, -1)}, true)};
      registry.get_kinematic_object(target)->position = position;
      return target;
    }};

    // Create the targets and add the attacks system
    targets = {
        create_target({-20, -100}), create_target({20, 60}),  create_target({-200, 100}), create_target({100, -100}),
        create_target({-100, -99}), create_target({0, -200}), create_target({0, -192}),   create_target({0, 0}),
    };
    registry.add_system<AttackSystem>();
    registry.add_system<DamageSystem>();
  }

  /// Create an attacks component for a game object.
  ///
  /// @tparam T - The type of the component or system.
  /// @param list - The initializer list to pass to the constructor.
  void create_attack_component(const std::vector<AttackAlgorithm> &&enabled_attacks) {
    const int game_object_id{registry.create_game_object({std::make_shared<Attacks>(enabled_attacks)}, true)};
    registry.get_kinematic_object(game_object_id)->rotation = 180;
  }

  /// Get the attacks system from the registry.
  ///
  /// @return The attacks system.
  auto get_attacks_system() -> std::shared_ptr<AttackSystem> { return registry.get_system<AttackSystem>(); }
};

/// Implements the fixture for the DamageSystem tests.
class DamageSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};

  /// Set up the fixture for the tests.
  void SetUp() override { registry.add_system<DamageSystem>(); }

  /// Create health and armour attributes for use in testing.
  void create_health_and_armour_attributes() {
    registry.create_game_object({std::make_shared<Health>(300, -1), std::make_shared<Armour>(100, -1)});
  }

  /// Get the damage system from the registry.
  ///
  /// @return The damage system.
  auto get_damage_system() -> std::shared_ptr<DamageSystem> { return registry.get_system<DamageSystem>(); }
};

// ----- TESTS ----------------------------------
/// Test that performing an area of effect attack works correctly.
TEST_F(AttackSystemFixture, TestAttacksDoAreaOfEffectAttack) {
  create_attack_component({AttackAlgorithm::AreaOfEffect});
  ASSERT_FALSE(get_attacks_system()->do_attack(8, targets).has_value());
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
  create_attack_component({AttackAlgorithm::Melee});
  ASSERT_FALSE(get_attacks_system()->do_attack(8, targets).has_value());
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
  create_attack_component({AttackAlgorithm::Ranged});
  std::tuple<Vec2d, double, double> attack_result{get_attacks_system()->do_attack(8, targets).value()};
  ASSERT_EQ(get<0>(attack_result), Vec2d(0, 0));
  ASSERT_EQ(get<1>(attack_result), -300);
  ASSERT_NEAR(get<2>(attack_result), 0, 1e-13);
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(AttackSystemFixture, TestAttacksDoAttackInvalidGameObjectId) {
  create_attack_component({AttackAlgorithm::Ranged});
  ASSERT_THROW_MESSAGE(
      (get_attacks_system()->do_attack(-1, targets)), RegistryError,
      "The game object `-1` is not registered with the registry or does not have the required component.")
}

/// Test that switching between attacks once works correctly.
TEST_F(AttackSystemFixture, TestAttacksPreviousNextAttackSingle) {
  create_attack_component({AttackAlgorithm::AreaOfEffect, AttackAlgorithm::Melee, AttackAlgorithm::Ranged});
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
  get_attacks_system()->next_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 1);
  get_attacks_system()->previous_attack(8);
  ASSERT_EQ(registry.get_component<Attacks>(8)->attack_state, 0);
}

/// Test that switching between attacks multiple times works correctly.
TEST_F(AttackSystemFixture, TestAttacksPreviousAttackMultiple) {
  create_attack_component({AttackAlgorithm::AreaOfEffect, AttackAlgorithm::Melee, AttackAlgorithm::Ranged});
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
  ASSERT_THROW_MESSAGE(
      get_attacks_system()->next_attack(-1), RegistryError,
      "The game object `-1` is not registered with the registry or does not have the required component.")
  ASSERT_THROW_MESSAGE(
      get_attacks_system()->previous_attack(-1), RegistryError,
      "The game object `-1` is not registered with the registry or does not have the required component.")
}

/// Test that damage is dealt when health and armour are lower than damage.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageLowHealthArmour) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, 350);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 50);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 0);
}

/// Test that no damage is dealt when armour is larger than damage.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageLargeArmour) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, 50);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 50);
}

/// Test that no damage is dealt when damage is zero.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageZeroDamage) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, 0);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 100);
}

/// Test that damage is dealt when armour is zero.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageZeroArmour) {
  create_health_and_armour_attributes();
  auto armour{registry.get_component<Armour>(0)};
  armour->set_value(0);
  get_damage_system()->deal_damage(0, 100);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 200);
  ASSERT_EQ(armour->get_value(), 0);
}

/// Test that damage is dealt when health is zero.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageZeroHealth) {
  create_health_and_armour_attributes();
  auto health{registry.get_component<Health>(0)};
  health->set_value(0);
  get_damage_system()->deal_damage(0, 50);
  ASSERT_EQ(health->get_value(), 0);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 50);
}

/// Test that no damage is dealt when the attributes are not initialised.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageNonexistentAttributes) {
  registry.create_game_object({});
  ASSERT_THROW_MESSAGE(
      get_damage_system()->deal_damage(0, 100), RegistryError,
      "The game object `0` is not registered with the registry or does not have the required component.")
}

/// Test that an exception is raised if an invalid game object ID is provided.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(
      get_damage_system()->deal_damage(-1, 100), RegistryError,
      "The game object `-1` is not registered with the registry or does not have the required component.")
}
