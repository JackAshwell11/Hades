// Std headers
#include <numbers>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/movements.hpp"
#include "events.hpp"
#include "macros.hpp"

/// Implements the fixture for the AttackSystem tests.
class AttackSystemFixture : public testing::Test {
 protected:
  /// The configuration for creating an attacker.
  struct AttackerConfig {
    /// Whether the ranged attack is enabled or not.
    bool ranged = false;

    /// Whether the game object has a steering movement component or not.
    bool steering_movement = false;
  };

  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// A list of targets for use in testing.
  std::vector<int> targets;

  /// Set up the fixture for the tests.
  void SetUp() override {
    auto create_target{[&](const cpVect position) {
      const int target{registry.create_game_object(GameObjectType::Enemy)};
      registry.add_component<Armour>(target, 0);
      registry.add_component<Health>(target, 80);
      registry.add_component<KinematicComponent>(target, cpvzero);
      cpBodySetPosition(*registry.get_component<KinematicComponent>(target)->body, position);
      return target;
    }};

    // Create the targets and add the attack system
    targets = {
        create_target({-20, -100}),  create_target({20, 60}),  create_target({-200, 100}), create_target({100, -100}),
        create_target({-100, -100}), create_target({0, -200}), create_target({0, -192}),   create_target({0, 0}),
    };
    registry.add_system<AttackSystem>();
    registry.add_system<DamageSystem>();
    registry.add_system<PhysicsSystem>();
  }

  /// Tear down the fixture after the tests.
  void TearDown() override { clear_listeners(); }

  /// Create an attacker game object with the specified attack types.
  ///
  /// @param config - The component configuration for the attacker.
  void create_attacker(const AttackerConfig& config) {
    const int game_object_id{registry.create_game_object(GameObjectType::Player)};
    std::vector<std::unique_ptr<RangedAttack>> attacks;
    if (config.ranged) {
      attacks.push_back(std::make_unique<SingleBulletAttack>(2, 20, 3 * SPRITE_SIZE, 200));
      attacks.push_back(std::make_unique<MultiBulletAttack>(1, 10, 3 * SPRITE_SIZE, 200, 5));
    }
    registry.add_component<Attack>(game_object_id, std::move(attacks));
    registry.add_component<KinematicComponent>(game_object_id, cpvzero);
    registry.get_component<KinematicComponent>(game_object_id)->rotation = -std::numbers::pi / 2;
    if (config.steering_movement) {
      registry.add_component<SteeringMovement>(
          game_object_id, std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>>{});
    }
  }

  /// Get the attack system from the registry.
  ///
  /// @return The attack system.
  [[nodiscard]] auto get_attack_system() const -> std::shared_ptr<AttackSystem> {
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
    const auto game_object_id{registry.create_game_object(GameObjectType::Player)};
    registry.add_component<Armour>(game_object_id, 100);
    registry.add_component<Health>(game_object_id, 300);
  }

  /// Get the damage system from the registry.
  ///
  /// @return The damage system.
  [[nodiscard]] auto get_damage_system() const -> std::shared_ptr<DamageSystem> {
    return registry.get_system<DamageSystem>();
  }
};

/// Test that the attack component is updated correctly with a zero delta time.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateZeroDeltaTime) {
  create_attacker({.ranged = true});
  get_attack_system()->update(0);
  ASSERT_EQ(registry.get_component<Attack>(8)->get_selected_ranged_attack()->time_since_last_use, 0);
}

/// Test that the attack component is updated correctly with a non-zero delta time.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateNonZeroDeltaTime) {
  create_attacker({.ranged = true});
  get_attack_system()->update(5);
  ASSERT_EQ(registry.get_component<Attack>(8)->get_selected_ranged_attack()->time_since_last_use, 5);
}

/// Test that the attack system does not do an automated attack if the cooldown is not up.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateSteeringMovementZeroDeltaTime) {
  auto game_object_created{-1};
  auto game_object_creation_callback{
      [&](const GameObjectID game_object_id, const GameObjectType) { game_object_created = game_object_id; }};
  create_attacker({.ranged = true, .steering_movement = true});
  add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  get_attack_system()->update(0);
  ASSERT_EQ(game_object_created, -1);
}

/// Test that the attack system does not do an automated attack if the steering movement is not in the target state.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateSteeringMovementNotTarget) {
  auto game_object_created{-1};
  auto game_object_creation_callback{
      [&](const GameObjectID game_object_id, const GameObjectType) { game_object_created = game_object_id; }};
  create_attacker({.ranged = true, .steering_movement = true});
  add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  get_attack_system()->update(5);
  ASSERT_EQ(game_object_created, -1);
}

/// Test that the attack system does an automated attack correctly.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateSteeringMovement) {
  auto game_object_created{-1};
  auto game_object_creation_callback{
      [&](const GameObjectID game_object_id, const GameObjectType) { game_object_created = game_object_id; }};
  create_attacker({.ranged = true, .steering_movement = true});
  add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  registry.get_component<SteeringMovement>(8)->movement_state = SteeringMovementState::Target;
  get_attack_system()->update(5);
  ASSERT_EQ(game_object_created, 9);
}

/// Test that the attack system calls the correct callbacks during updating.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateCallbacks) {
  auto attack_cooldown_update_callback{[&](const GameObjectID game_object_id, const double ranged_cooldown) {
    ASSERT_EQ(game_object_id, 8);
    ASSERT_EQ(ranged_cooldown, 1);
  }};
  add_callback<EventType::AttackCooldownUpdate>(attack_cooldown_update_callback);
  create_attacker({.ranged = true});
  registry.get_component<Attack>(8)->get_selected_ranged_attack()->time_since_last_use = 1;
  get_attack_system()->update(0);
}

/// Test that performing a ranged attack with a single bullet works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackRangedSingle) {
  auto game_object_created{-1};
  auto game_object_creation_callback{
      [&](const GameObjectID game_object_id, const GameObjectType) { game_object_created = game_object_id; }};
  create_attacker({.ranged = true});
  add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  get_attack_system()->update(5);
  ASSERT_TRUE(get_attack_system()->do_attack(8));
  ASSERT_EQ(game_object_created, 9);
  const auto bullet{registry.get_component<KinematicComponent>(game_object_created)};
  const auto [pos_x, pos_y]{cpBodyGetPosition(*bullet->body)};
  const auto [vel_x, vel_y]{cpBodyGetVelocity(*bullet->body)};
  ASSERT_NEAR(pos_x, 0, 1e-13);
  ASSERT_NEAR(pos_y, -64, 1e-13);
  ASSERT_NEAR(vel_x, 0, 1e-13);
  ASSERT_NEAR(vel_y, -200, 1e-13);
}

/// Test that performing a ranged attack with multiple bullets works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackRangedMultipleBullets) {
  std::vector<GameObjectID> game_objects_created;
  auto game_object_creation_callback{
      [&](const GameObjectID game_object_id, const GameObjectType) { game_objects_created.push_back(game_object_id); }};
  create_attacker({.ranged = true});
  add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  get_attack_system()->update(5);
  registry.get_system<AttackSystem>()->next_ranged_attack(8);
  ASSERT_TRUE(get_attack_system()->do_attack(8));
  ASSERT_EQ(game_objects_created.size(), 5);

  // Check that each bullet has the expected velocity
  const std::vector<cpVect> expected_velocities{{{.x = -141.42135623730951, .y = -141.42135623730951},
                                                 {.x = -76.536686473017951, .y = -184.77590650225736},
                                                 {.x = 0.0, .y = -200},
                                                 {.x = 76.536686473017966, .y = -184.77590650225736},
                                                 {.x = 141.42135623730951, .y = -141.42135623730951}}};
  for (auto i{0}; std::cmp_less(i, game_objects_created.size()); i++) {
    const auto* bullet{*registry.get_component<KinematicComponent>(game_objects_created[i])->body};
    const auto [vel_x, vel_y]{cpBodyGetVelocity(bullet)};
    ASSERT_NEAR(vel_x, expected_velocities[i].x, 1e-13);
    ASSERT_NEAR(vel_y, expected_velocities[i].y, 1e-13);
  }
}

/// Test that performing an attack with no attack algorithms doesn't work.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackEmptyAttacks) {
  create_attacker({});
  get_attack_system()->update(5);
  ASSERT_FALSE(get_attack_system()->do_attack(8));
  ASSERT_EQ(registry.get_component<Health>(targets[0])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[1])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[2])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[3])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[4])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[5])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[6])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[7])->get_value(), 80);
}

/// Test that performing an attack before the cooldown is up doesn't work.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackCooldown) {
  create_attacker({.ranged = true});
  ASSERT_FALSE(get_attack_system()->do_attack(8));
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackInvalidGameObjectId) {
  create_attacker({.ranged = true});
  ASSERT_THROW_MESSAGE((get_attack_system()->do_attack(-1)), RegistryError,
                       "The component `Attack` for the game object ID `-1` is not registered with the registry.")
}

/// Test that switching between ranged attacks once works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemPreviousNextRangedAttackSingle) {
  create_attacker({.ranged = true});
  const auto attack{registry.get_component<Attack>(8)};
  ASSERT_EQ(attack->selected_ranged_attack, 0);
  registry.get_system<AttackSystem>()->next_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 1);
  registry.get_system<AttackSystem>()->previous_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 0);
}

/// Test that switching between ranged attacks multiple times works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemPreviousNextRangedAttackMultiple) {
  create_attacker({.ranged = true});
  const auto attack{registry.get_component<Attack>(8)};
  ASSERT_EQ(attack->selected_ranged_attack, 0);
  registry.get_system<AttackSystem>()->next_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 1);
  registry.get_system<AttackSystem>()->next_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 1);
  registry.get_system<AttackSystem>()->next_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 1);
  registry.get_system<AttackSystem>()->previous_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 0);
  registry.get_system<AttackSystem>()->previous_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 0);
  registry.get_system<AttackSystem>()->previous_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 0);
}

/// Test that switching between ranged attacks works correctly when there are no attacks.
TEST_F(AttackSystemFixture, TestAttackSystemPreviousNextRangedAttackEmptyAttacks) {
  create_attacker({});
  const auto attack{registry.get_component<Attack>(8)};
  ASSERT_EQ(attack->selected_ranged_attack, 0);
  registry.get_system<AttackSystem>()->next_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 0);
  registry.get_system<AttackSystem>()->previous_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 0);
}

/// Test that switching between ranged attacks calls the correct callback.
TEST_F(AttackSystemFixture, TestAttackSystemPreviousNextRangedAttackCallback) {
  std::vector<int> selected_attacks{};
  auto ranged_attack_callback{[&](const int selected_attack) { selected_attacks.push_back(selected_attack); }};
  add_callback<EventType::RangedAttackSwitch>(ranged_attack_callback);
  create_attacker({.ranged = true});
  const auto attack{registry.get_component<Attack>(8)};
  ASSERT_EQ(attack->selected_ranged_attack, 0);
  registry.get_system<AttackSystem>()->next_ranged_attack(8);
  registry.get_system<AttackSystem>()->previous_ranged_attack(8);
  const std::vector expected_attacks{1, 0};
  ASSERT_EQ(selected_attacks, expected_attacks);
}

/// Test that damage is dealt when health and armour are lower than damage.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageLowHealthArmour) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, 380);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 20);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 0);
}

/// Test that no damage is dealt when armour is larger than damage.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageLargeArmour) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, 80);
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 20);
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

/// Test that a game object is deleted if the damage drops the health to 0.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageDeleteGameObject) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, 500);
  registry.update(0);
  ASSERT_FALSE(registry.has_component<Health>(0));
  ASSERT_FALSE(registry.has_component<Armour>(0));
}

/// Test that a game object is deleted if the health is already 0.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageZeroHealth) {
  create_health_and_armour_attributes();
  registry.get_component<Health>(0)->set_value(0);
  get_damage_system()->deal_damage(0, 0);
  registry.update(0);
  ASSERT_FALSE(registry.has_component<Health>(0));
  ASSERT_FALSE(registry.has_component<Armour>(0));
}

/// Test that the damage system calls the correct callbacks during execution.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageCallbacks) {
  create_health_and_armour_attributes();
  std::vector<double> health_percentages;
  std::vector<double> armour_percentages;
  auto health_changed_callback{
      [&](const GameObjectID, const double percentage) { health_percentages.push_back(percentage); }};
  auto armour_changed_callback{
      [&](const GameObjectID, const double percentage) { armour_percentages.push_back(percentage); }};
  add_callback<EventType::HealthChanged>(health_changed_callback);
  add_callback<EventType::ArmourChanged>(armour_changed_callback);
  get_damage_system()->deal_damage(0, 50);
  get_damage_system()->deal_damage(0, 125);
  ASSERT_EQ(health_percentages, std::vector({0.75}));
  ASSERT_EQ(armour_percentages, std::vector({0.5, 0.0}));
}

/// Test that an exception is thrown if a game object does not have the required components.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageNonexistentComponents) {
  registry.create_game_object(GameObjectType::Player);
  ASSERT_THROW_MESSAGE(get_damage_system()->deal_damage(0, 100), RegistryError,
                       "The component `Health` for the game object ID `0` is not registered with the registry.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(get_damage_system()->deal_damage(-1, 100), RegistryError,
                       "The component `Health` for the game object ID `-1` is not registered with the registry.")
}
