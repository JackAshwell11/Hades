// External headers
#include <chipmunk/chipmunk_structs.h>

// Local headers
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "macros.hpp"

/// Implements the fixture for the PhysicsSystem tests.
class PhysicsSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object(
        GameObjectType::Player, cpvzero,
        {std::make_shared<KinematicComponent>(std::vector<cpVect>{}), std::make_shared<MovementForce>(100, -1)});
    registry.add_system<PhysicsSystem>();
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
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->p, cpv(32, 32));
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->v, cpvzero);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpvzero);
}

/// Test updating the physics system with a game object that has no velocity and a force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateNoVelocityValidForce) {
  registry.get_component<KinematicComponent>(0)->body->f = {.x = 10, .y = 0};
  get_physics_system()->update(1.0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->p, cpv(32, 32));
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->v, cpv(10, 0));
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpvzero);
}

/// Test updating the physics system with a game object that has velocity and no force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateValidVelocityNoForce) {
  registry.get_component<KinematicComponent>(0)->body->v = {.x = 100, .y = 0};
  get_physics_system()->update(1.0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->p, cpv(132, 32));
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->v, cpv(0.01, 0));
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpvzero);
}

/// Test updating the physics system with a game object that has velocity and a force.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemUpdateValidVelocityValidForce) {
  registry.get_component<KinematicComponent>(0)->body->v = {.x = 100, .y = 0};
  registry.get_component<KinematicComponent>(0)->body->f = {.x = 10, .y = 0};
  get_physics_system()->update(1.0);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->p, cpv(132, 32));
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->v, cpv(10.01, 0));
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpvzero);
}

/// Test that adding a zero force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceZeroForceNoForce) {
  get_physics_system()->add_force(0, cpvzero);
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpvzero);
}

/// Test that adding a positive force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForcePositiveForceNoForce) {
  get_physics_system()->add_force(0, {10, 0});
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpv(100, 0));
}

/// Test that adding a negative force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceNegativeForceNoForce) {
  get_physics_system()->add_force(0, {-10, 0});
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpv(-100, 0));
}

/// Test that adding a positive infinite force to a game object without a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForcePositiveInfiniteForceNoForce) {
  get_physics_system()->add_force(0,
                                  {std::numeric_limits<cpFloat>::infinity(), std::numeric_limits<cpFloat>::infinity()});
  ASSERT_TRUE(std::isnan(registry.get_component<KinematicComponent>(0)->body->f.x));
  ASSERT_TRUE(std::isnan(registry.get_component<KinematicComponent>(0)->body->f.y));
}

/// Test that adding a positive force to a game object with a force works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForcePositiveForceValidForce) {
  registry.get_component<KinematicComponent>(0)->body->f = {.x = 10, .y = 0};
  get_physics_system()->add_force(0, {10, 0});
  ASSERT_EQ(registry.get_component<KinematicComponent>(0)->body->f, cpv(110, 0));
}

/// Test that adding a force to a static wall body doesn't change its position.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddForceStaticWall) {
  // Walls should never have a MovementForce component, but if they do, their position should not change
  const auto wall_id{registry.create_game_object(
      GameObjectType::Wall, cpvzero,
      {std::make_shared<KinematicComponent>(true), std::make_shared<MovementForce>(100, -1)})};
  get_physics_system()->add_force(wall_id, {10, 0});
  registry.get_system<PhysicsSystem>()->update(1);
  ASSERT_EQ(registry.get_component<KinematicComponent>(wall_id)->body->p, cpv(32, 32));
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
  const auto bullet_id{get_physics_system()->add_bullet({cpvzero, cpvzero}, 50, GameObjectType::Player)};
  ASSERT_EQ(bullet_id, 1);
  const auto *body{*registry.get_component<KinematicComponent>(bullet_id)->body};
  ASSERT_EQ(body->p, cpvzero);
  ASSERT_EQ(body->f, cpvzero);
}

/// Test that adding a bullet with a non-zero position works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddBulletNonZeroPosition) {
  const auto bullet_id{get_physics_system()->add_bullet({{.x = 100, .y = 0}, cpvzero}, 100, GameObjectType::Player)};
  ASSERT_EQ(bullet_id, 1);
  const auto *body{*registry.get_component<KinematicComponent>(bullet_id)->body};
  ASSERT_EQ(body->p, cpv(100, 0));
  ASSERT_EQ(body->f, cpvzero);
}

/// Test that adding a bullet with a non-zero velocity works correctly.
TEST_F(PhysicsSystemFixture, TestPhysicsSystemAddBulletNonZeroVelocity) {
  const auto bullet_id{get_physics_system()->add_bullet({cpvzero, {.x = 200, .y = 150}}, 10, GameObjectType::Player)};
  ASSERT_EQ(bullet_id, 1);
  const auto *body{*registry.get_component<KinematicComponent>(bullet_id)->body};
  ASSERT_EQ(body->p, cpvzero);
  ASSERT_EQ(body->v, cpv(200, 150));
}
