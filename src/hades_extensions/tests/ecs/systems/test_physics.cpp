// Local headers
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "macros.hpp"

/// Implements the fixture for the PhysicsSystem tests.
class PhysicsSystemFixture : public testing::Test {
 protected:
  /// A random generator for use in testing.
  std::mt19937 random_generator;

  /// The registry that manages the game objects, components, and systems.
  Registry registry{random_generator};

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object(
        GameObjectType::Player, cpvzero,
        {std::make_shared<KinematicComponent>(std::vector<cpVect>{}), std::make_shared<MovementForce>(100, -1)});
    registry.add_system<PhysicsSystem>();
  }

  /// Create an item at the specified position.
  ///
  /// @param position The position to create the item.
  void create_item(const cpVect &position) {
    const auto item_id{
        registry.create_game_object(GameObjectType::HealthPotion, cpvzero, {std::make_shared<KinematicComponent>()})};
    cpBodySetPosition(*registry.get_component<KinematicComponent>(item_id)->body, position);
  }

  /// Get the physics system from the registry.
  ///
  /// @return The physics system.
  [[nodiscard]] auto get_physics_system() const -> std::shared_ptr<PhysicsSystem> {
    return registry.get_system<PhysicsSystem>();
  }
};

/// Test updating the physics system with a game object that has no velocity and no force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateNoVelocityNoForce) {
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpv(32, 32));
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test updating the physics system with a game object that has no velocity and a force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateNoVelocityValidForce) {
  cpBodySetForce(*registry.get_component<KinematicComponent>(0)->body, {.x = 10, .y = 0});
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpv(32, 32));
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpv(10, 0));
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test updating the physics system with a game object that has velocity and no force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateValidVelocityNoForce) {
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = 100, .y = 0});
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpv(132, 32));
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpv(0.01, 0));
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test updating the physics system with a game object that has velocity and a force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateValidVelocityValidForce) {
  cpBodySetVelocity(*registry.get_component<KinematicComponent>(0)->body, {.x = 100, .y = 0});
  cpBodySetForce(*registry.get_component<KinematicComponent>(0)->body, {.x = 10, .y = 0});
  get_physics_system()->update(1.0);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(0)->body), cpv(132, 32));
  ASSERT_EQ(cpBodyGetVelocity(*registry.get_component<KinematicComponent>(0)->body), cpv(10.01, 0));
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test that adding a zero force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceZeroForceNoForce) {
  get_physics_system()->add_force(0, cpvzero);
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpvzero);
}

/// Test that adding a positive force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForcePositiveForceNoForce) {
  get_physics_system()->add_force(0, {10, 0});
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(100, 0));
}

/// Test that adding a negative force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceNegativeForceNoForce) {
  get_physics_system()->add_force(0, {-10, 0});
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(-100, 0));
}

/// Test that adding a positive infinite force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForcePositiveInfiniteForceNoForce) {
  get_physics_system()->add_force(0,
                                  {std::numeric_limits<cpFloat>::infinity(), std::numeric_limits<cpFloat>::infinity()});
  ASSERT_TRUE(std::isnan(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body).x));
  ASSERT_TRUE(std::isnan(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body).y));
}

/// Test that adding a positive force to a game object with a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForcePositiveForceValidForce) {
  cpBodySetForce(*registry.get_component<KinematicComponent>(0)->body, {.x = 10, .y = 0});
  get_physics_system()->add_force(0, {10, 0});
  ASSERT_EQ(cpBodyGetForce(*registry.get_component<KinematicComponent>(0)->body), cpv(110, 0));
}

/// Test that adding a force to a static wall body doesn't change its position.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceStaticWall) {
  // Walls should never have a MovementForce component, but if they do, their position should not change
  const auto wall_id{registry.create_game_object(
      GameObjectType::Wall, cpvzero,
      {std::make_shared<KinematicComponent>(true), std::make_shared<MovementForce>(100, -1)})};
  get_physics_system()->add_force(wall_id, {10, 0});
  get_physics_system()->update(1);
  ASSERT_EQ(cpBodyGetPosition(*registry.get_component<KinematicComponent>(wall_id)->body), cpv(32, 32));
}

/// Test that an exception is thrown if a game object does not have a kinematic component.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceNonexistentKinematicComponent) {
  registry.create_game_object(GameObjectType::Player, cpvzero, {std::make_shared<MovementForce>(100, -1)});
  ASSERT_THROW_MESSAGE(
      get_physics_system()->add_force(1, {0, 0}), RegistryError,
      "The component `KinematicComponent` for the game object ID `1` is not registered with the registry.");
}

/// Test that an exception is thrown if a game object does not have a movement force component.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceNonexistentMovementForceComponent) {
  registry.create_game_object(GameObjectType::Player, cpvzero,
                              {std::make_shared<KinematicComponent>(std::vector<cpVect>{})});
  ASSERT_THROW_MESSAGE(get_physics_system()->add_force(1, {0, 0}), RegistryError,
                       "The component `MovementForce` for the game object ID `1` is not registered with the registry.");
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceInvalidGameObjectId) {
  ASSERT_THROW_MESSAGE(
      get_physics_system()->add_force(-1, {0, 0}), RegistryError,
      "The component `MovementForce` for the game object ID `-1` is not registered with the registry.");
}

/// Test that adding a bullet with a zero position and velocity works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddBulletZero) {
  get_physics_system()->add_bullet({cpvzero, cpvzero}, 50, GameObjectType::Player);
  const auto *body{*registry.get_component<KinematicComponent>(1)->body};
  ASSERT_EQ(cpBodyGetPosition(body), cpvzero);
  ASSERT_EQ(cpBodyGetForce(body), cpvzero);
}

/// Test that adding a bullet with a non-zero position works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddBulletNonZeroPosition) {
  get_physics_system()->add_bullet({{.x = 100, .y = 0}, cpvzero}, 100, GameObjectType::Player);
  const auto *body{*registry.get_component<KinematicComponent>(1)->body};
  ASSERT_EQ(cpBodyGetPosition(body), cpv(100, 0));
  ASSERT_EQ(cpBodyGetForce(body), cpvzero);
}

/// Test that adding a bullet with a non-zero velocity works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddBulletNonZeroVelocity) {
  get_physics_system()->add_bullet({cpvzero, {.x = 200, .y = 150}}, 10, GameObjectType::Player);
  const auto *body{*registry.get_component<KinematicComponent>(1)->body};
  ASSERT_EQ(cpBodyGetPosition(body), cpvzero);
  ASSERT_EQ(cpBodyGetVelocity(body), cpv(200, 150));
}

/// Test that getting the nearest item doesn't work if the game object is far away.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemFarAway) {
  create_item({100, 100});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), -1);
}

/// Test that getting the nearest item doesn't work if the game object is next to the item.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemNextTo) {
  create_item({65, 32});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), -1);
}

/// Test that getting the nearest item works if the game object is touching the item.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemTouching) {
  create_item({64, 32});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), 1);
}

/// Test that getting the nearest item works if the game object is on top of the item.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemOnTopOf) {
  create_item({32, 32});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), 1);
}

/// Test that getting the nearest item works if the game object is touching the item and there are multiple items.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemGetNearestItemMultipleItems) {
  create_item({48, 32});
  create_item({32, 32});
  ASSERT_EQ(get_physics_system()->get_nearest_item(0), 2);
}
