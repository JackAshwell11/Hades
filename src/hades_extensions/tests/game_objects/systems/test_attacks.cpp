// External headers
#include <chipmunk/chipmunk_structs.h>

// Local headers
#include "game_objects/stats.hpp"
#include "game_objects/systems/attacks.hpp"
#include "game_objects/systems/physics.hpp"
#include "macros.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the AttackSystem tests.
class AttackSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// A list of targets for use in testing.
  std::vector<int> targets;

  /// Set up the fixture for the tests.
  void SetUp() override {
    auto create_target{[&](const cpVect position) {
      const int target{
          registry.create_game_object(cpvzero, {std::make_shared<Health>(50, -1), std::make_shared<Armour>(0, -1),
                                                std::make_shared<KinematicComponent>(std::vector<cpVect>{})})};
      registry.get_component<KinematicComponent>(target)->body->p = position;
      return target;
    }};

    // Create the targets and add the attacks system offseting the positions
    // need to be offset by (32, 32) since grid_pos_to_pixel converts the grid
    // position to the centre of the tile
    targets = {
        create_target({12, -68}),  create_target({52, 92}),   create_target({-168, 132}), create_target({132, -68}),
        create_target({-68, -67}), create_target({32, -168}), create_target({32, -160}),  create_target({32, 32}),
    };
    registry.add_system<AttackSystem>();
    registry.add_system<DamageSystem>();
  }

  /// Create an attacks component for a game object.
  ///
  /// @param enabled_attacks - The attacks to include in the component.
  void create_attack_component(const std::vector<AttackAlgorithm> &&enabled_attacks) {
    const int game_object_id{registry.create_game_object(
        cpvzero,
        {std::make_shared<Attacks>(enabled_attacks), std::make_shared<KinematicComponent>(std::vector<cpVect>{})})};
    registry.get_component<KinematicComponent>(game_object_id)->body->a = 180;
  }

  /// Get the attacks system from the registry.
  ///
  /// @return The attacks system.
  [[nodiscard]] auto get_attacks_system() const -> std::shared_ptr<AttackSystem> {
    return registry.get_system<AttackSystem>();
  }
};

/// Implements the fixture for the DamageSystem tests.
class DamageSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override { registry.add_system<DamageSystem>(); }

  /// Create health and armour attributes for use in testing.
  void create_health_and_armour_attributes() {
    registry.create_game_object(cpvzero, {std::make_shared<Health>(300, -1), std::make_shared<Armour>(100, -1)});
  }

  /// Get the damage system from the registry.
  ///
  /// @return The damage system.
  [[nodiscard]] auto get_damage_system() const -> std::shared_ptr<DamageSystem> {
    return registry.get_system<DamageSystem>();
  }
};

// TODO: Rename all tests docstrings for has_indicator_bar to has_indicator_bar()

// ----- TESTS ----------------------------------
/// Test that the required components return the correct value for has_indicator_bar.
TEST(Tests, TestAttackSystemComponentsHasIndicatorBar) { ASSERT_FALSE(Attacks{{}}.has_indicator_bar()); }

/// Test that the required components return the correct value for has_indicator_bar.
TEST(Tests, TestDamageSystemComponentsHasIndicatorBar) {
  ASSERT_TRUE(Health(-1, -1).has_indicator_bar());
  ASSERT_TRUE(Armour(-1, -1).has_indicator_bar());
}

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
  const std::tuple attack_result{get_attacks_system()->do_attack(8, targets).value()};
  ASSERT_EQ(get<0>(attack_result), cpv(32, 32));
  ASSERT_EQ(get<1>(attack_result), -300);
  ASSERT_NEAR(get<2>(attack_result), 0, 1e-13);
}

/// Test that an exception is thrown if an invalid game object ID is provided.
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

/// Test that an exception is thrown if an invalid game object ID is provided.
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
  const auto armour{registry.get_component<Armour>(0)};
  armour->set_value(0);
  get_damage_system()->deal_damage(0, 100);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 200);
  ASSERT_EQ(armour->get_value(), 0);
}

/// Test that damage is dealt when health is zero.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageZeroHealth) {
  create_health_and_armour_attributes();
  const auto health{registry.get_component<Health>(0)};
  health->set_value(0);
  get_damage_system()->deal_damage(0, 50);
  ASSERT_EQ(health->get_value(), 0);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 50);
}

/// Test that an exception is thrown if a game object does not have the required components.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageNonexistentComponents) {
  registry.create_game_object(cpvzero, {});
  ASSERT_THROW_MESSAGE(
      get_damage_system()->deal_damage(0, 100), RegistryError,
      "The game object `0` is not registered with the registry or does not have the required component.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(
      get_damage_system()->deal_damage(-1, 100), RegistryError,
      "The game object `-1` is not registered with the registry or does not have the required component.")
}
