// Std headers
#include <numbers>

// Local headers
#include "ecs/stats.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "macros.hpp"

/// Implements the fixture for the AttackSystem tests.
class AttackSystemFixture : public testing::Test {
 protected:
  /// The configuration for creating an attacker.
  struct AttackerConfig {
    /// Whether the ranged attack is enabled or not.
    bool ranged = false;

    /// Whether the melee attack is enabled or not.
    bool melee = false;

    /// Whether the area of effect attack is enabled or not.
    bool area_of_effect = false;

    /// Whether the game object has a steering movement component or not.
    bool steering_movement = false;
  };

  /// A random generator for use in testing.
  std::mt19937 random_generator;

  /// The registry that manages the game objects, components, and systems.
  Registry registry{random_generator};

  /// A list of targets for use in testing.
  std::vector<int> targets;

  /// Set up the fixture for the tests.
  void SetUp() override {
    auto create_target{[&](const cpVect position) {
      const int target{registry.create_game_object(
          GameObjectType::Enemy, cpvzero,
          {std::make_shared<Armour>(0, -1), std::make_shared<Health>(80, -1), std::make_shared<KinematicComponent>()})};
      cpBodySetPosition(*registry.get_component<KinematicComponent>(target)->body, position);
      return target;
    }};

    // Create the targets and add the attack system offsetting the positions by (32, 32) since grid_pos_to_pixel()
    // converts the target position to (32, 32)
    targets = {
        create_target({12, -68}),  create_target({52, 92}),   create_target({-168, 132}), create_target({132, -68}),
        create_target({-68, -68}), create_target({32, -168}), create_target({32, -160}),  create_target({32, 32}),
    };
    registry.add_system<AttackSystem>();
    registry.add_system<DamageSystem>();
    registry.add_system<PhysicsSystem>();
  }

  /// Create an attacker game object with the specified attack types.
  ///
  /// @param config - The component configuration for the attacker.
  void create_attacker(const AttackerConfig &config) {
    const auto attack{std::make_shared<Attack>()};
    if (config.ranged) {
      attack->add_ranged_attack(
          std::make_unique<SingleBulletAttack>(Stat(2, -1), Stat(20, -1), Stat(3 * SPRITE_SIZE, -1), Stat(200, -1)));
      attack->add_ranged_attack(std::make_unique<MultiBulletAttack>(Stat{1, 3}, Stat{10, 3}, Stat{3 * SPRITE_SIZE, 3},
                                                                    Stat{200, 1}, Stat{5, 3}));
    }
    if (config.melee) {
      attack->set_melee_attack({{2, -1}, {20, -1}, {3 * SPRITE_SIZE, -1}, {std::numbers::pi / 4, -1}});
    }
    if (config.area_of_effect) {
      attack->set_special_attack({{2, -1}, {20, -1}, {3 * SPRITE_SIZE, -1}});
    }
    std::vector<std::shared_ptr<ComponentBase>> components{attack, std::make_shared<KinematicComponent>()};
    if (config.steering_movement) {
      components.emplace_back(std::make_shared<SteeringMovement>(
          std::unordered_map<SteeringMovementState, std::vector<SteeringBehaviours>>{}));
    }
    const int game_object_id{registry.create_game_object(GameObjectType::Player, cpvzero, std::move(components))};
    registry.get_component<KinematicComponent>(game_object_id)->rotation = -std::numbers::pi / 2;
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
  /// A random generator for use in testing.
  std::mt19937 random_generator;

  /// The registry that manages the game objects, components, and systems.
  Registry registry{random_generator};

  /// Set up the fixture for the tests.
  void SetUp() override { registry.add_system<DamageSystem>(); }

  /// Create health and armour attributes for use in testing.
  void create_health_and_armour_attributes() {
    registry.create_game_object(GameObjectType::Player, cpvzero,
                                {std::make_shared<Armour>(100, -1), std::make_shared<Health>(300, -1)});
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
  create_attacker({.ranged{true}});
  get_attack_system()->update(0);
  ASSERT_EQ(registry.get_component<Attack>(8)->get_selected_ranged_attack()->time_since_last_use, 0);
}

/// Test that the attack component is updated correctly with a non-zero delta time.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateNonZeroDeltaTime) {
  create_attacker({.ranged{true}});
  get_attack_system()->update(5);
  ASSERT_EQ(registry.get_component<Attack>(8)->get_selected_ranged_attack()->time_since_last_use, 5);
}

/// Test that the attack system does not do an automated attack if the cooldown is not up.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateSteeringMovementZeroDeltaTime) {
  auto game_object_created{-1};
  auto game_object_creation_callback{[&](const GameObjectID game_object_id) { game_object_created = game_object_id; }};
  create_attacker({.ranged{true}, .steering_movement{true}});
  registry.add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  get_attack_system()->update(0);
  ASSERT_EQ(game_object_created, -1);
}

/// Test that the attack system does not do an automated attack if the steering movement is not in the target state.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateSteeringMovementNotTarget) {
  auto game_object_created{-1};
  auto game_object_creation_callback{[&](const GameObjectID game_object_id) { game_object_created = game_object_id; }};
  create_attacker({.ranged{true}, .steering_movement{true}});
  registry.add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  get_attack_system()->update(5);
  ASSERT_EQ(game_object_created, -1);
}

/// Test that the attack system does an automated attack correctly.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateSteeringMovement) {
  auto game_object_created{-1};
  auto game_object_creation_callback{[&](const GameObjectID game_object_id) { game_object_created = game_object_id; }};
  create_attacker({.ranged{true}, .steering_movement{true}});
  registry.add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  registry.get_component<SteeringMovement>(8)->movement_state = SteeringMovementState::Target;
  get_attack_system()->update(5);
  ASSERT_EQ(game_object_created, 9);
}

/// Test that performing a ranged attack with a single bullet works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackRangedSingle) {
  auto game_object_created{-1};
  auto game_object_creation_callback{[&](const GameObjectID game_object_id) {
    // The velocity for the bullet is set after the game object is created
    const auto *bullet{*registry.get_component<KinematicComponent>(game_object_id)->body};
    const auto [pos_x, pos_y]{cpBodyGetPosition(bullet)};
    const auto [vel_x, vel_y]{cpBodyGetVelocity(bullet)};
    ASSERT_NEAR(pos_x, 32, 1e-13);
    ASSERT_NEAR(pos_y, -32, 1e-13);
    ASSERT_NEAR(vel_x, 0, 1e-13);
    ASSERT_NEAR(vel_y, 0, 1e-13);
    game_object_created = game_object_id;
  }};
  create_attacker({.ranged{true}});
  registry.add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  get_attack_system()->update(5);
  ASSERT_TRUE(get_attack_system()->do_attack(8, AttackType::Ranged));
  ASSERT_EQ(game_object_created, 9);
}

/// Test that performing a ranged attack with a single bullet does not work if the attack type is wrong.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackRangedSingleWrongType) {
  create_attacker({.ranged{true}});
  get_attack_system()->update(5);
  ASSERT_FALSE(get_attack_system()->do_attack(8, AttackType::Melee));
}

/// Test that performing a ranged attack with multiple bullets works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackRangedMultipleBullets) {
  std::vector<GameObjectID> game_objects_created;
  auto game_object_creation_callback{
      [&](const GameObjectID game_object_id) { game_objects_created.push_back(game_object_id); }};
  create_attacker({.ranged{true}});
  registry.add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  get_attack_system()->update(5);
  registry.get_system<AttackSystem>()->next_ranged_attack(8);
  ASSERT_TRUE(get_attack_system()->do_attack(8, AttackType::Ranged));
  ASSERT_EQ(game_objects_created.size(), 5);

  // Check that each bullet has the expected velocity
  const std::vector<cpVect> expected_velocities{{{.x = -141.42135623730951, .y = -141.42135623730951},
                                                 {.x = -76.536686473017951, .y = -184.77590650225736},
                                                 {.x = 0.0, .y = -200},
                                                 {.x = 76.536686473017966, .y = -184.77590650225736},
                                                 {.x = 141.42135623730951, .y = -141.42135623730951}}};
  for (auto i{0}; std::cmp_less(i, game_objects_created.size()); i++) {
    const auto *bullet{*registry.get_component<KinematicComponent>(game_objects_created[i])->body};
    const auto [vel_x, vel_y]{cpBodyGetVelocity(bullet)};
    ASSERT_NEAR(vel_x, expected_velocities[i].x, 1e-13);
    ASSERT_NEAR(vel_y, expected_velocities[i].y, 1e-13);
  }
}

/// Test that performing a ranged attack with multiple bullets does not work if the attack type is wrong.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackRangedMultipleWrongType) {
  create_attacker({.ranged{true}});
  get_attack_system()->update(5);
  ASSERT_FALSE(get_attack_system()->do_attack(8, AttackType::Melee));
}

/// Test that performing a melee attack works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackMelee) {
  create_attacker({.melee{true}});
  get_attack_system()->update(5);
  ASSERT_TRUE(get_attack_system()->do_attack(8, AttackType::Melee));
  ASSERT_EQ(registry.get_component<Health>(targets[0])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[1])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[2])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[3])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[4])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[5])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[6])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[7])->get_value(), 60);
}

/// Test that performing a melee attack does not work if the attack type is wrong.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackMeleeWrongType) {
  create_attacker({.melee{true}});
  get_attack_system()->update(5);
  ASSERT_FALSE(get_attack_system()->do_attack(8, AttackType::Special));
  ASSERT_EQ(registry.get_component<Health>(targets[0])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[1])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[2])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[3])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[4])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[5])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[6])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[7])->get_value(), 80);
}

/// Test that performing an area of effect attack works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackAreaOfEffect) {
  create_attacker({.area_of_effect{true}});
  get_attack_system()->update(5);
  ASSERT_TRUE(get_attack_system()->do_attack(8, AttackType::Special));
  ASSERT_EQ(registry.get_component<Health>(targets[0])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[1])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[2])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[3])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[4])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[5])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[6])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[7])->get_value(), 60);
}

/// Test that performing an area of effect attack does not work if the attack type is wrong.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackAreaOfEffectWrongType) {
  create_attacker({.area_of_effect{true}});
  get_attack_system()->update(5);
  ASSERT_FALSE(get_attack_system()->do_attack(8, AttackType::Ranged));
  ASSERT_EQ(registry.get_component<Health>(targets[0])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[1])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[2])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[3])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[4])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[5])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[6])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[7])->get_value(), 80);
}

/// Test that performing an attack with no attack algorithms doesn't work.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackEmptyAttacks) {
  create_attacker({});
  get_attack_system()->update(5);
  ASSERT_FALSE(get_attack_system()->do_attack(8, AttackType::Ranged));
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
  create_attacker({.ranged{true}});
  ASSERT_FALSE(get_attack_system()->do_attack(8, AttackType::Ranged));
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackInvalidGameObjectId) {
  create_attacker({.ranged{true}});
  ASSERT_THROW_MESSAGE((get_attack_system()->do_attack(-1, AttackType::Ranged)), RegistryError,
                       "The component `Attack` for the game object ID `-1` is not registered with the registry.")
}

/// Test that switching between ranged attacks once works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemPreviousNextRangedAttackSingle) {
  create_attacker({.ranged{true}, .melee{true}, .area_of_effect{true}});
  const auto attack{registry.get_component<Attack>(8)};
  ASSERT_EQ(attack->selected_ranged_attack, 0);
  registry.get_system<AttackSystem>()->next_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 1);
  registry.get_system<AttackSystem>()->previous_ranged_attack(8);
  ASSERT_EQ(attack->selected_ranged_attack, 0);
}

/// Test that switching between ranged attacks multiple times works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemPreviousNextRangedAttackMultiple) {
  create_attacker({.ranged{true}, .melee{true}, .area_of_effect{true}});
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
  registry.add_callback<EventType::RangedAttackSwitch>(ranged_attack_callback);
  create_attacker({
      .ranged{true},
  });
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
  ASSERT_FALSE(registry.has_component(0, typeid(Health)));
  ASSERT_FALSE(registry.has_component(0, typeid(Armour)));
}

/// Test that a game object is deleted if the health is already 0.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageZeroHealth) {
  create_health_and_armour_attributes();
  registry.get_component<Health>(0)->set_value(0);
  get_damage_system()->deal_damage(0, 0);
  registry.update(0);
  ASSERT_FALSE(registry.has_component(0, typeid(Health)));
  ASSERT_FALSE(registry.has_component(0, typeid(Armour)));
}

/// Test that an exception is thrown if a game object does not have the required components.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageNonexistentComponents) {
  registry.create_game_object(GameObjectType::Player, cpvzero, {});
  ASSERT_THROW_MESSAGE(get_damage_system()->deal_damage(0, 100), RegistryError,
                       "The component `Health` for the game object ID `0` is not registered with the registry.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(get_damage_system()->deal_damage(-1, 100), RegistryError,
                       "The component `Health` for the game object ID `-1` is not registered with the registry.")
}
