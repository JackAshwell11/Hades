// Std headers
#include <numbers>

// Local headers
#include "ecs/steering.hpp"
#include "game_object.hpp"
#include "macros.hpp"

/// Implements the fixture for the obstacle_avoidance() tests.
class ObstacleAvoidanceFixture : public testing::Test {
 protected:
  /// The Chipmunk2D bodies and shapes for the game objects.
  std::vector<std::pair<ChipmunkHandle<cpBody, cpBodyFree>, ChipmunkHandle<cpShape, cpShapeFree>>> game_objects;

  /// The Chipmunk2D space.
  ChipmunkHandle<cpSpace, cpSpaceFree> space{cpSpaceNew()};

  /// Add the game object to the space.
  ///
  /// @param positions - The list of game object positions.
  /// @param entity - The type of game object to add.
  void add_game_objects(const std::vector<cpVect> &&positions, const GameObjectType entity = GameObjectType::Wall) {
    for (const auto &position : positions) {
      auto body{ChipmunkHandle<cpBody, cpBodyFree>(cpBodyNewStatic())};
      auto shape{ChipmunkHandle<cpShape, cpShapeFree>(cpBoxShapeNew(*body, SPRITE_SIZE, SPRITE_SIZE, 0))};
      cpBodySetPosition(*body, position);
      cpShapeSetCollisionType(*shape, static_cast<cpCollisionType>(entity));
      cpShapeSetFilter(*shape, {CP_NO_GROUP, static_cast<cpBitmask>(entity), CP_ALL_CATEGORIES});
      cpShapeSetBody(*shape, *body);
      cpSpaceAddBody(*space, *body);
      cpSpaceAddShape(*space, *shape);
      game_objects.emplace_back(std::move(body), std::move(shape));
    }
  }
};

/// Test if a position outside the radius produces the correct arrive force.
TEST(Tests, TestArriveOutsideSlowingRange) {
  ASSERT_EQ(arrive({500, 500}, cpvzero), cpv(-0.7071067811865475, -0.7071067811865475));
}

/// Test if a position on the radius produces the correct arrive force.
TEST(Tests, TestArriveOnSlowingRange) {
  ASSERT_EQ(arrive({135, 135}, cpvzero), cpv(-0.70710678118654757, -0.70710678118654757));
}

/// Test if a position inside the radius produces the correct arrive force.
TEST(Tests, TestArriveInsideSlowingRange) {
  ASSERT_EQ(arrive({100, 100}, cpvzero), cpv(-0.70710678118654746, -0.70710678118654746));
}

/// Test if a position near the target produces the correct arrive force.
TEST(Tests, TestArriveNearTarget) {
  ASSERT_EQ(arrive({50, 50}, cpvzero), cpv(-0.70710678118654746, -0.70710678118654746));
}

/// Test if a position on the target produces the correct arrive force.
TEST(Tests, TestArriveOnTarget) { ASSERT_EQ(arrive(cpvzero, cpvzero), cpvzero); }

/// Test if a non-moving target produces the correct evade force.
TEST(Tests, TestEvadeNonMovingTarget) {
  ASSERT_EQ(evade(cpvzero, {100, 100}, cpvzero), cpv(-0.70710678118654757, -0.70710678118654757));
}

/// Test if a moving target produces the correct evade force.
TEST(Tests, TestEvadeMovingTarget) {
  ASSERT_EQ(evade(cpvzero, {100, 100}, {-50, 0}), cpv(-0.44721359549995798, -0.89442719099991597));
}

/// Test if having the same position produces the correct evade force.
TEST(Tests, TestEvadeSamePositions) {
  ASSERT_EQ(evade(cpvzero, cpvzero, cpvzero), cpvzero);
  ASSERT_EQ(evade(cpvzero, cpvzero, {-50, 0}), cpv(1, 0));
}

/// Test if a higher current position produces the correct flee force.
TEST(Tests, TestFleeHigherCurrent) {
  ASSERT_EQ(flee({100, 100}, {50, 50}), cpv(0.70710678118654757, 0.70710678118654757));
}

/// Test if a higher target position produces the correct flee force.
TEST(Tests, TestFleeHigherTarget) {
  ASSERT_EQ(flee({50, 50}, {100, 100}), cpv(-0.70710678118654757, -0.70710678118654757));
}

/// Test if two equal positions produce the correct flee force.
TEST(Tests, TestFleeEqual) { ASSERT_EQ(flee({100, 100}, {100, 100}), cpvzero); }

/// Test if a negative current position produces the correct flee force.
TEST(Tests, TestFleeNegativeCurrent) {
  ASSERT_EQ(flee({-50, -50}, {100, 100}), cpv(-0.7071067811865475, -0.7071067811865475));
}

/// Test if a negative target position produces the correct flee force.
TEST(Tests, TestFleeNegativeTarget) {
  ASSERT_EQ(flee({100, 100}, {-50, -50}), cpv(0.7071067811865475, 0.7071067811865475));
}

/// Test if two negative positions produce the correct flee force.
TEST(Tests, TestFleeNegativePositions) { ASSERT_EQ(flee({-50, -50}, {-50, -50}), cpvzero); }

/// Test if a multiple position list produces the correct follow path force.
TEST(Tests, TestFollowPathSinglePosition) {
  std::vector<cpVect> path_list{{.x = 250, .y = 250}};
  ASSERT_EQ(follow_path({100, 100}, path_list), cpv(0.7071067811865475, 0.7071067811865475));
}

/// Test if reaching a position in a single position list produces the correct follow path force.
TEST(Tests, TestFollowPathSinglePositionReached) {
  std::vector<cpVect> path_list{{.x = 100, .y = 100}};
  ASSERT_EQ(follow_path({100, 100}, path_list), cpvzero);
  ASSERT_EQ(path_list, std::vector{cpv(100, 100)});
}

/// Test if a multiple position list produces the correct follow path force.
TEST(Tests, TestFollowPathMultiplePositions) {
  std::vector<cpVect> path_list{{.x = 350, .y = 350}, {.x = 500, .y = 500}};
  ASSERT_EQ(follow_path({200, 200}, path_list), cpv(0.7071067811865475, 0.7071067811865475));
}

/// Test if reaching a position in a multiple position list produces the correct follow path force.
TEST(Tests, TestFollowPathMultiplePositionsReached) {
  std::vector<cpVect> path_list{{.x = 100, .y = 100}, {.x = 250, .y = 250}};
  ASSERT_EQ(follow_path({100, 100}, path_list), cpv(0.7071067811865475, 0.7071067811865475));
  ASSERT_EQ(path_list, std::vector({cpv(250, 250), cpv(100, 100)}));
}

/// Test if an empty list raises the correct exception.
TEST(Tests, TestFollowPathEmptyList) {
  std::vector<cpVect> path_list;
  ASSERT_THROW_MESSAGE(follow_path({100, 100}, path_list), std::length_error, "The path list is empty.")
}

/// Test if no obstacles produce the correct obstacle avoidance force.
TEST_F(ObstacleAvoidanceFixture, TestObstacleAvoidanceNoObstacles) {
  ASSERT_EQ(obstacle_avoidance(*space, {0, 0}, {0, 100}), cpvzero);
}

/// Test if a non-moving game object produces the correct obstacle avoidance force.
TEST_F(ObstacleAvoidanceFixture, TestObstacleAvoidanceNonMoving) {
  add_game_objects({{.x = 0, .y = 75}});
  ASSERT_EQ(obstacle_avoidance(*space, {0, 0}, {0, 0}), cpvzero);
}

/// Test if a single forward obstacle produces the correct obstacle avoidance force.
TEST_F(ObstacleAvoidanceFixture, TestObstacleAvoidanceSingleForward) {
  add_game_objects({{.x = 0, .y = 75}});
  ASSERT_EQ(obstacle_avoidance(*space, {0, 0}, {0, 100}), cpv(0, -1));
}

/// Test if a single left obstacle produces the correct obstacle avoidance force.
TEST_F(ObstacleAvoidanceFixture, TestObstacleAvoidanceSingleLeft) {
  add_game_objects({{.x = -64, .y = 75}});
  ASSERT_EQ(obstacle_avoidance(*space, {0, 0}, {0, 100}), cpv(0.73612033100878249, -0.67685069127210062));
}

/// Test if a single right obstacle produces the correct obstacle avoidance force.
TEST_F(ObstacleAvoidanceFixture, TestObstacleAvoidanceSingleRight) {
  add_game_objects({{.x = 64, .y = 75}});
  ASSERT_EQ(obstacle_avoidance(*space, {0, 0}, {0, 100}), cpv(-0.73612033100878249, -0.67685069127210062));
}

/// Test if a left and forward obstacle produces the correct obstacle avoidance force.
TEST_F(ObstacleAvoidanceFixture, TestObstacleAvoidanceLeftForward) {
  add_game_objects({{.x = -64, .y = 75}, {.x = 0, .y = 75}});
  ASSERT_EQ(obstacle_avoidance(*space, {0, 0}, {0, 100}), cpv(0.73612033100878249, -1.6768506912721006));
}

/// Test if a right and forward obstacle produces the correct obstacle avoidance force.
TEST_F(ObstacleAvoidanceFixture, TestObstacleAvoidanceRightForward) {
  add_game_objects({{.x = 64, .y = 75}, {.x = 0, .y = 75}});
  ASSERT_EQ(obstacle_avoidance(*space, {0, 0}, {0, 100}), cpv(-0.73612033100878249, -1.6768506912721006));
}

/// Test if all three obstacles produce the correct obstacle avoidance force.
TEST_F(ObstacleAvoidanceFixture, TestObstacleAvoidanceLeftRightForward) {
  add_game_objects({{.x = -64, .y = 75}, {.x = 0, .y = 75}, {.x = 64, .y = 75}});
  ASSERT_EQ(obstacle_avoidance(*space, {0, 0}, {0, 100}), cpv(0, -2.3537013825442012));
}

/// Test if an angled velocity produces the correct obstacle avoidance force.
TEST_F(ObstacleAvoidanceFixture, TestObstacleAvoidanceAngledVelocity) {
  add_game_objects({{.x = 64, .y = 75}});
  ASSERT_EQ(obstacle_avoidance(*space, {0, 0}, {100, 100}), cpv(-0.59701076932020425, -0.80223322127402441));
}

/// Test if an out of range obstacle produces the correct obstacle avoidance force.
TEST_F(ObstacleAvoidanceFixture, TestObstacleAvoidanceObstacleOutOfRange) {
  add_game_objects({{.x = 0, .y = 300}});
  ASSERT_EQ(obstacle_avoidance(*space, {0, 0}, {0, 100}), cpvzero);
}

/// Test if having a player entity produces the correct obstacle avoidance force.
TEST_F(ObstacleAvoidanceFixture, TestObstacleAvoidancePlayerEntity) {
  add_game_objects({{.x = 0, .y = 75}});
  add_game_objects({{.x = 0, .y = 0}}, GameObjectType::Player);
  ASSERT_EQ(obstacle_avoidance(*space, {0, 0}, {0, 100}), cpv(0, -1));
}

/// Test if a non-moving target produces the correct pursue force.
TEST(Tests, TestPursueNonMovingTarget) {
  ASSERT_EQ(pursue(cpvzero, {100, 100}, cpvzero), cpv(0.70710678118654757, 0.70710678118654757));
}

/// Test if a moving target produces the correct pursue force.
TEST(Tests, TestPursueMovingTarget) {
  ASSERT_EQ(pursue(cpvzero, {100, 100}, {-50, 0}), cpv(0.44721359549995798, 0.89442719099991597));
}

/// Test if having the same position produces the correct pursue force.
TEST(Tests, TestPursueSamePositions) {
  ASSERT_EQ(pursue(cpvzero, cpvzero, cpvzero), cpvzero);
  ASSERT_EQ(pursue(cpvzero, cpvzero, {-50, 0}), cpv(-1, 0));
}

/// Test if a higher current position produces the correct seek force.
TEST(Tests, TestSeekHigherCurrent) {
  ASSERT_EQ(seek({100, 100}, {50, 50}), cpv(-0.70710678118654757, -0.70710678118654757));
}

/// Test if a higher target position produces the correct seek force.
TEST(Tests, TestSeekHigherTarget) {
  ASSERT_EQ(seek({50, 50}, {100, 100}), cpv(0.70710678118654757, 0.70710678118654757));
}

/// Test if two equal positions produce the correct seek force.
TEST(Tests, TestSeekEqual) { ASSERT_EQ(seek({100, 100}, {100, 100}), cpvzero); }

/// Test if a negative current position produces the correct seek force.
TEST(Tests, TestSeekNegativeCurrent) {
  ASSERT_EQ(seek({-50, -50}, {100, 100}), cpv(0.7071067811865475, 0.7071067811865475));
}

/// Test if a negative target position produces the correct seek force.
TEST(Tests, TestSeekNegativeTarget) {
  ASSERT_EQ(seek({100, 100}, {-50, -50}), cpv(-0.7071067811865475, -0.7071067811865475));
}

/// Test if two negative positions produce the correct seek force.
TEST(Tests, TestSeekNegativePositions) { ASSERT_EQ(seek({-50, -50}, {-50, -50}), cpvzero); }

/// Test if a non-moving game object produces the correct wander force.
TEST(Tests, TestWanderNonMoving) {
  // This is due to floating point precision
  const auto [non_moving_result_x, non_moving_result_y] = wander(cpvzero, std::numbers::pi / 3);
  ASSERT_DOUBLE_EQ(non_moving_result_x, 0.8660254037844385);
  ASSERT_DOUBLE_EQ(non_moving_result_y, -0.5);
}

/// Test if a moving game object produces the correct wander force.
TEST(Tests, TestWanderMoving) {
  ASSERT_EQ(wander({100, -100}, std::numbers::pi / 3), cpv(0.76590121355591045, -0.64295826542131307));
}

/// Test if a zero angle produces the correct wander force.
TEST(Tests, TestWanderZeroAngle) { ASSERT_EQ(wander(cpvzero, 0), cpv(0, -1)); }
