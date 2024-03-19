// Local headers
#include "game_objects/steering.hpp"
#include "macros.hpp"

// ----- TESTS ------------------------------
/// Test if a position outside the radius produces the correct arrive force.
TEST(Tests, TestArriveOutsideSlowingRange) {
  ASSERT_EQ(arrive({500, 500}, {0, 0}), cpVect(-0.7071067811865475, -0.7071067811865475));
}

/// Test if a position on the radius produces the correct arrive force.
TEST(Tests, TestArriveOnSlowingRange) {
  ASSERT_EQ(arrive({135, 135}, {0, 0}), cpVect(-0.70710678118654757, -0.70710678118654757));
}

/// Test if a position inside the radius produces the correct arrive force.
TEST(Tests, TestArriveInsideSlowingRange) {
  ASSERT_EQ(arrive({100, 100}, {0, 0}), cpVect(-0.70710678118654746, -0.70710678118654746));
}

/// Test if a position near the target produces the correct arrive force.
TEST(Tests, TestArriveNearTarget) {
  ASSERT_EQ(arrive({50, 50}, {0, 0}), cpVect(-0.70710678118654746, -0.70710678118654746));
}

/// Test if a position on the target produces the correct arrive force.
TEST(Tests, TestArriveOnTarget) { ASSERT_EQ(arrive({0, 0}, {0, 0}), cpVect(0, 0)); }

/// Test if a non-moving target produces the correct evade force.
TEST(Tests, TestEvadeNonMovingTarget) {
  ASSERT_EQ(evade({0, 0}, {100, 100}, {0, 0}), cpVect(-0.70710678118654757, -0.70710678118654757));
}

/// Test if a moving target produces the correct evade force.
TEST(Tests, TestEvadeMovingTarget) {
  ASSERT_EQ(evade({0, 0}, {100, 100}, {-50, 0}), cpVect(-0.54288882138918848, -0.8398045770360254));
}

/// Test if having the same position produces the correct evade force.
TEST(Tests, TestEvadeSamePositions) {
  ASSERT_EQ(evade({0, 0}, {0, 0}, {0, 0}), cpVect(0, 0));
  ASSERT_EQ(evade({0, 0}, {0, 0}, {-50, 0}), cpVect(0, 0));
}

/// Test if a higher current position produces the correct flee force.
TEST(Tests, TestFleeHigherCurrent) {
  ASSERT_EQ(flee({100, 100}, {50, 50}), cpVect(0.70710678118654757, 0.70710678118654757));
}

/// Test if a higher target position produces the correct flee force.
TEST(Tests, TestFleeHigherTarget) {
  ASSERT_EQ(flee({50, 50}, {100, 100}), cpVect(-0.70710678118654757, -0.70710678118654757));
}

/// Test if two equal positions produce the correct flee force.
TEST(Tests, TestFleeEqual) { ASSERT_EQ(flee({100, 100}, {100, 100}), cpVect(0, 0)); }

/// Test if a negative current position produces the correct flee force.
TEST(Tests, TestFleeNegativeCurrent) {
  ASSERT_EQ(flee({-50, -50}, {100, 100}), cpVect(-0.7071067811865475, -0.7071067811865475));
}

/// Test if a negative target position produces the correct flee force.
TEST(Tests, TestFleeNegativeTarget) {
  ASSERT_EQ(flee({100, 100}, {-50, -50}), cpVect(0.7071067811865475, 0.7071067811865475));
}

/// Test if two negative positions produce the correct flee force.
TEST(Tests, TestFleeNegativePositions) { ASSERT_EQ(flee({-50, -50}, {-50, -50}), cpVect(0, 0)); }

/// Test if a multiple position list produces the correct follow path force.
TEST(Tests, TestFollowPathSinglePosition) {
  std::vector<cpVect> path_list{{250, 250}};
  ASSERT_EQ(follow_path({100, 100}, path_list), cpVect(0.7071067811865475, 0.7071067811865475));
}

/// Test if reaching a position in a single position list produces the correct follow path force.
TEST(Tests, TestFollowPathSinglePositionReached) {
  std::vector<cpVect> path_list{{100, 100}};
  ASSERT_EQ(follow_path({100, 100}, path_list), cpVect(0, 0));
  ASSERT_EQ(path_list, std::vector{cpVect(100, 100)});
}

/// Test if a multiple position list produces the correct follow path force.
TEST(Tests, TestFollowPathMultiplePositions) {
  std::vector<cpVect> path_list{{350, 350}, {500, 500}};
  ASSERT_EQ(follow_path({200, 200}, path_list), cpVect(0.7071067811865475, 0.7071067811865475));
}

/// Test if reaching a position in a multiple position list produces the correct follow path force.
TEST(Tests, TestFollowPathMultiplePositionsReached) {
  std::vector<cpVect> path_list{{100, 100}, {250, 250}};
  ASSERT_EQ(follow_path({100, 100}, path_list), cpVect(0.7071067811865475, 0.7071067811865475));
  ASSERT_EQ(path_list, std::vector({cpVect(250, 250), cpVect(100, 100)}));
}

/// Test if an empty list raises the correct exception.
TEST(Tests, TestFollowPathEmptyList) {
  std::vector<cpVect> path_list;
  ASSERT_THROW_MESSAGE(follow_path({100, 100}, path_list), std::length_error, "The path list is empty.")
}

// /// Test if no obstacles produce the correct obstacle avoidance force.
// TEST(Tests, TestObstacleAvoidanceNoObstacles) {
//   ASSERT_EQ(obstacle_avoidance({100, 100}, {0, 100}, {}), cpVect(0, 0));
// }
//
// /// Test if an out of range obstacle produces the correct obstacle avoidance force.
// TEST(Tests, TestObstacleAvoidanceObstacleOutOfRange) {
//   ASSERT_EQ(obstacle_avoidance({100, 100}, {0, 100}, {{10, 10}}), cpVect(0, 0));
// }
//
// /// Test if an angled velocity produces the correct obstacle avoidance force.
// TEST(Tests, TestObstacleAvoidanceAngledVelocity) {
//   ASSERT_EQ(obstacle_avoidance({100, 100}, {100, 100}, {{1, 2}}), cpVect(0.2588190451025206, -0.9659258262890683));
// }
//
// /// Test if a non-moving game object produces the correct obstacle avoidance force.
// TEST(Tests, TestObstacleAvoidanceNonMoving) {
//   ASSERT_EQ(obstacle_avoidance({100, 100}, {0, 100}, {{1, 2}}), cpVect(0, 0));
// }
//
// /// Test if a single forward obstacle produces the correct obstacle avoidance force.
// TEST(Tests, TestObstacleAvoidanceSingleForward) {
//   ASSERT_EQ(obstacle_avoidance({100, 100}, {0, 100}, {{1, 2}}), cpVect(0, 0));
// }
//
// /// Test if a single left obstacle produces the correct obstacle avoidance force.
// TEST(Tests, TestObstacleAvoidanceSingleLeft) {
//   // This is due to floating point precision
//   const auto [single_left_result_x, single_left_result_y] = obstacle_avoidance({100, 100}, {0, 100}, {{0, 2}});
//   ASSERT_DOUBLE_EQ(single_left_result_x, 0.8660254037844387);
//   ASSERT_DOUBLE_EQ(single_left_result_y, -0.5);
// }
//
// /// Test if a single right obstacle produces the correct obstacle avoidance force.
// TEST(Tests, TestObstacleAvoidanceSingleRight) {
//   // This is due to floating point precision
//   const auto [single_right_result_x, single_right_result_y] = obstacle_avoidance({100, 100}, {0, 100}, {{2, 2}});
//   ASSERT_DOUBLE_EQ(single_right_result_x, -0.8660254037844386);
//   ASSERT_DOUBLE_EQ(single_right_result_y, -0.5);
// }
//
// /// Test if a left and forward obstacle produces the correct obstacle avoidance force.
// TEST(Tests, TestObstacleAvoidanceLeftForward) {
//   // This is due to floating point precision
//   const auto [left_forward_result_x, left_forward_result_y] =
//       obstacle_avoidance({100, 100}, {0, 100}, {{0, 2}, {1, 2}});
//   ASSERT_DOUBLE_EQ(left_forward_result_x, 0.8660254037844387);
//   ASSERT_DOUBLE_EQ(left_forward_result_y, -0.5);
// }
//
// /// Test if a right and forward obstacle produces the correct obstacle avoidance force.
// TEST(Tests, TestObstacleAvoidanceRightForward) {
//   // This is due to floating point precision
//   const auto [right_forward_result_x, right_forward_result_y] =
//       obstacle_avoidance({100, 100}, {0, 100}, {{1, 2}, {2, 2}});
//   ASSERT_DOUBLE_EQ(right_forward_result_x, -0.8660254037844386);
//   ASSERT_DOUBLE_EQ(right_forward_result_y, -0.5);
// }
//
// /// Test if all three obstacles produce the correct obstacle avoidance force.
// TEST(Tests, TestObstacleAvoidanceLeftRightForward) {
//   ASSERT_EQ(obstacle_avoidance({100, 100}, {0, 100}, {{0, 2}, {1, 2}, {2, 2}}), cpVect(0, -1));
// }

/// Test if a non-moving target produces the correct pursue force.
TEST(Tests, TestPursueNonMovingTarget) {
  ASSERT_EQ(pursue({0, 0}, {100, 100}, {0, 0}), cpVect(0.70710678118654757, 0.70710678118654757));
}

/// Test if a moving target produces the correct pursue force.
TEST(Tests, TestPursueMovingTarget) {
  ASSERT_EQ(pursue({0, 0}, {100, 100}, {-50, 0}), cpVect(0.54288882138918848, 0.8398045770360254));
}

/// Test if having the same position produces the correct pursue force.
TEST(Tests, TestPursueSamePositions) {
  ASSERT_EQ(pursue({0, 0}, {0, 0}, {0, 0}), cpVect(0, 0));
  ASSERT_EQ(pursue({0, 0}, {0, 0}, {-50, 0}), cpVect(0, 0));
}

/// Test if a higher current position produces the correct seek force.
TEST(Tests, TestSeekHigherCurrent) {
  ASSERT_EQ(seek({100, 100}, {50, 50}), cpVect(-0.70710678118654757, -0.70710678118654757));
}

/// Test if a higher target position produces the correct seek force.
TEST(Tests, TestSeekHigherTarget) {
  ASSERT_EQ(seek({50, 50}, {100, 100}), cpVect(0.70710678118654757, 0.70710678118654757));
}

/// Test if two equal positions produce the correct seek force.
TEST(Tests, TestSeekEqual) { ASSERT_EQ(seek({100, 100}, {100, 100}), cpVect(0, 0)); }

/// Test if a negative current position produces the correct seek force.
TEST(Tests, TestSeekNegativeCurrent) {
  ASSERT_EQ(seek({-50, -50}, {100, 100}), cpVect(0.7071067811865475, 0.7071067811865475));
}

/// Test if a negative target position produces the correct seek force.
TEST(Tests, TestSeekNegativeTarget) {
  ASSERT_EQ(seek({100, 100}, {-50, -50}), cpVect(-0.7071067811865475, -0.7071067811865475));
}

/// Test if two negative positions produce the correct seek force.
TEST(Tests, TestSeekNegativePositions) { ASSERT_EQ(seek({-50, -50}, {-50, -50}), cpVect(0, 0)); }

/// Test if a non-moving game object produces the correct wander force.
TEST(Tests, TestWanderNonMoving) {
  // This is due to floating point precision
  const auto [non_moving_result_x, non_moving_result_y] = wander({0, 0}, 60);
  ASSERT_DOUBLE_EQ(non_moving_result_x, 0.8660254037844385);
  ASSERT_DOUBLE_EQ(non_moving_result_y, -0.5);
}

/// Test if a moving game object produces the correct wander force.
TEST(Tests, TestWanderMoving) { ASSERT_EQ(wander({100, -100}, 60), cpVect(0.76590121355591045, -0.64295826542131307)); }

/// Test if a zero angle produces the correct wander force.
TEST(Tests, TestWanderZeroAngle) { ASSERT_EQ(wander({0, 0}, 0), cpVect(0, -1)); }
