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
          {std::make_shared<Armour>(0, -1), std::make_shared<AttackRange>(-1, -1), std::make_shared<Health>(80, -1),
           std::make_shared<KinematicComponent>(std::vector<cpVect>{})})};
      cpBodySetPosition(*registry.get_component<KinematicComponent>(target)->body, position);
      return target;
    }};

    // Create the targets and add the attack system offseting the positions by (32, 32) since grid_pos_to_pixel()
    // converts the target position to (32, 32)
    targets = {
        create_target({12, -68}),  create_target({52, 92}),   create_target({-168, 132}), create_target({132, -68}),
        create_target({-68, -68}), create_target({32, -168}), create_target({32, -160}),  create_target({32, 32}),
    };
    registry.add_system<AttackSystem>();
    registry.add_system<DamageSystem>();
    registry.add_system<PhysicsSystem>();
  }

  /// Create an attack component for a game object.
  ///
  /// @param enabled_attacks - The attacks to include in the component.
  /// @param steering_movement - Whether the game object has a steering movement component or not.
  void create_attack_component(const std::vector<AttackAlgorithm> &&enabled_attacks,
                               const bool steering_movement = false) {
    std::vector<std::shared_ptr<ComponentBase>> components{std::make_shared<Attack>(enabled_attacks),
                                                           std::make_shared<AttackCooldown>(2, -1),
                                                           std::make_shared<AttackRange>(3 * SPRITE_SIZE, -1),
                                                           std::make_shared<Damage>(20, -1),
                                                           std::make_shared<KinematicComponent>(std::vector<cpVect>{}),
                                                           std::make_shared<MeleeAttackSize>(std::numbers::pi / 4, -1)};
    if (steering_movement) {
      components.push_back(std::make_shared<SteeringMovement>(
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

  /// Create an attacker game object with a given damage.
  ///
  /// @param damage - The damage to give the attacker.
  /// @return The game object ID of the attacker.
  [[nodiscard]] auto create_attacker(const int damage) -> int {
    return registry.create_game_object(GameObjectType::Player, cpvzero, {std::make_shared<Damage>(damage, -1)});
  }

  /// Get the damage system from the registry.
  ///
  /// @return The damage system.
  [[nodiscard]] auto get_damage_system() const -> std::shared_ptr<DamageSystem> {
    return registry.get_system<DamageSystem>();
  }
};

/// Test that the attack component is updated correctly with a zero deltatime.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateZeroDeltaTime) {
  create_attack_component({AttackAlgorithm::Ranged});
  get_attack_system()->update(0);
  ASSERT_EQ(registry.get_component<Attack>(8)->time_since_last_attack, 0);
}

/// Test that the attack component is updated correctly with a non-zero deltatime.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateNonZeroDeltaTime) {
  create_attack_component({AttackAlgorithm::Ranged});
  get_attack_system()->update(5);
  ASSERT_EQ(registry.get_component<Attack>(8)->time_since_last_attack, 5);
}

/// Test that the attack system does not do an automated attack if the cooldown is not up.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateSteeringMovementZeroDeltaTime) {
  auto game_object_created{-1};
  auto game_object_creation_callback{[&](const GameObjectID game_object_id) { game_object_created = game_object_id; }};
  create_attack_component({AttackAlgorithm::Ranged}, true);
  registry.add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  get_attack_system()->update(0);
  ASSERT_EQ(game_object_created, -1);
}

/// Test that the attack system does not do an automated attack if the steering movement is not in the target state.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateSteeringMovementNotTarget) {
  auto game_object_created{-1};
  auto game_object_creation_callback{[&](const GameObjectID game_object_id) { game_object_created = game_object_id; }};
  create_attack_component({AttackAlgorithm::Ranged}, true);
  registry.add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  get_attack_system()->update(5);
  ASSERT_EQ(game_object_created, -1);
}

/// Test that the attack system does an automated attack correctly.
TEST_F(AttackSystemFixture, TestAttackSystemUpdateSteeringMovement) {
  auto game_object_created{-1};
  auto game_object_creation_callback{[&](const GameObjectID game_object_id) { game_object_created = game_object_id; }};
  create_attack_component({AttackAlgorithm::Ranged}, true);
  registry.add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  registry.get_component<SteeringMovement>(8)->movement_state = SteeringMovementState::Target;
  get_attack_system()->update(5);
  ASSERT_EQ(game_object_created, 9);
}

/// Test that performing an area of effect attack works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackAreaOfEffect) {
  create_attack_component({AttackAlgorithm::AreaOfEffect});
  get_attack_system()->update(5);
  get_attack_system()->do_attack(8, targets);
  ASSERT_EQ(registry.get_component<Health>(targets[0])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[1])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[2])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[3])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[4])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[5])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[6])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[7])->get_value(), 60);
}

/// Test that performing a melee attack works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackMelee) {
  create_attack_component({AttackAlgorithm::Melee});
  get_attack_system()->update(5);
  get_attack_system()->do_attack(8, targets);
  ASSERT_EQ(registry.get_component<Health>(targets[0])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[1])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[2])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[3])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[4])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[5])->get_value(), 80);
  ASSERT_EQ(registry.get_component<Health>(targets[6])->get_value(), 60);
  ASSERT_EQ(registry.get_component<Health>(targets[7])->get_value(), 60);
}

/// Test that performing a ranged attack works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackRanged) {
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
  create_attack_component({AttackAlgorithm::Ranged});
  registry.add_callback<EventType::GameObjectCreation>(game_object_creation_callback);
  get_attack_system()->update(5);
  get_attack_system()->do_attack(8, targets);
  ASSERT_EQ(game_object_created, 9);
}

/// Test that performing an attack with no attack algorithms doesn't work.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackEmptyAttacks) {
  create_attack_component({});
  get_attack_system()->update(5);
  get_attack_system()->do_attack(8, targets);
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
  create_attack_component({AttackAlgorithm::Ranged});
  get_attack_system()->do_attack(8, targets);
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(AttackSystemFixture, TestAttackSystemDoAttackInvalidGameObjectId) {
  create_attack_component({AttackAlgorithm::Ranged});
  ASSERT_THROW_MESSAGE((get_attack_system()->do_attack(-1, targets)), RegistryError,
                       "The component `Attack` for the game object ID `-1` is not registered with the registry.")
}

/// Test that switching between attacks once works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemPreviousNextAttackSingle) {
  create_attack_component({AttackAlgorithm::AreaOfEffect, AttackAlgorithm::Melee, AttackAlgorithm::Ranged});
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 0);
  get_attack_system()->next_attack(8);
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 1);
  get_attack_system()->previous_attack(8);
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 0);
}

/// Test that switching between attacks multiple times works correctly.
TEST_F(AttackSystemFixture, TestAttackSystemPreviousAttackMultiple) {
  create_attack_component({AttackAlgorithm::AreaOfEffect, AttackAlgorithm::Melee, AttackAlgorithm::Ranged});
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 0);
  get_attack_system()->next_attack(8);
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 1);
  get_attack_system()->next_attack(8);
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 2);
  get_attack_system()->next_attack(8);
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 2);
  get_attack_system()->previous_attack(8);
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 1);
  get_attack_system()->previous_attack(8);
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 0);
  get_attack_system()->previous_attack(8);
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 0);
}

/// Test that changing the attack state works correctly when there are no attacks.
TEST_F(AttackSystemFixture, TestAttackSystemPreviousNextAttackEmptyAttacks) {
  create_attack_component({});
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 0);
  get_attack_system()->next_attack(8);
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 0);
  get_attack_system()->previous_attack(8);
  ASSERT_EQ(registry.get_component<Attack>(8)->attack_state, 0);
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(AttackSystemFixture, TestAttackSystemPreviousNextAttackInvalidGameObjectId) {
  create_attack_component({});
  ASSERT_THROW_MESSAGE(get_attack_system()->next_attack(-1), RegistryError,
                       "The component `Attack` for the game object ID `-1` is not registered with the registry.")
  ASSERT_THROW_MESSAGE(get_attack_system()->previous_attack(-1), RegistryError,
                       "The component `Attack` for the game object ID `-1` is not registered with the registry.")
}

/// Test that damage is dealt when health and armour are lower than damage.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageLowHealthArmour) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, create_attacker(380));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 20);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 0);
}

/// Test that no damage is dealt when armour is larger than damage.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageLargeArmour) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, create_attacker(80));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 20);
}

/// Test that no damage is dealt when damage is zero.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageZeroDamage) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, create_attacker(0));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<Armour>(0)->get_value(), 100);
}

/// Test that damage is dealt when armour is zero.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageZeroArmour) {
  create_health_and_armour_attributes();
  const auto armour{registry.get_component<Armour>(0)};
  armour->set_value(0);
  get_damage_system()->deal_damage(0, create_attacker(100));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 200);
  ASSERT_EQ(armour->get_value(), 0);
}

/// Test that a game object is deleted if the damage drops the health to 0.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageDeleteGameObject) {
  create_health_and_armour_attributes();
  get_damage_system()->deal_damage(0, create_attacker(500));
  ASSERT_FALSE(registry.has_component(0, typeid(Health)));
  ASSERT_FALSE(registry.has_component(0, typeid(Armour)));
}

/// Test that a game object is deleted if the health is already 0.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageZeroHealth) {
  create_health_and_armour_attributes();
  registry.get_component<Health>(0)->set_value(0);
  get_damage_system()->deal_damage(0, create_attacker(0));
  ASSERT_FALSE(registry.has_component(0, typeid(Health)));
  ASSERT_FALSE(registry.has_component(0, typeid(Armour)));
}

/// Test that an exception is thrown if a game object does not have the required components.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageNonexistentComponents) {
  registry.create_game_object(GameObjectType::Player, cpvzero, {});
  ASSERT_THROW_MESSAGE(get_damage_system()->deal_damage(0, create_attacker(100)), RegistryError,
                       "The component `Health` for the game object ID `0` is not registered with the registry.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(DamageSystemFixture, TestDamageSystemDealDamageInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(get_damage_system()->deal_damage(-1, create_attacker(100)), RegistryError,
                       "The component `Health` for the game object ID `-1` is not registered with the registry.")
}
