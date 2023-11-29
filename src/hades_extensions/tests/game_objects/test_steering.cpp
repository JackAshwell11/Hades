// Local headers
#include "game_objects/steering.hpp"
#include "macros.hpp"

// ----- TESTS ------------------------------
/// Test that adding two vectors produce the correct result.
TEST(Tests, TestVec2dAddition) {
  ASSERT_EQ(Vec2d(0, 0) + Vec2d(1, 1), Vec2d(1, 1));
  ASSERT_EQ(Vec2d(-3, -2) + Vec2d(-1, -1), Vec2d(-4, -3));
  ASSERT_EQ(Vec2d(6, 3) + Vec2d(5, 5), Vec2d(11, 8));
  ASSERT_EQ(Vec2d(1, 1) + Vec2d(1, 1), Vec2d(2, 2));
  ASSERT_EQ(Vec2d(-5, 4) + Vec2d(7, -1), Vec2d(2, 3));
}

/// Test that compound adding two vectors produce the correct result.
TEST(Tests, TestVec2dCompoundAddition) {
  ASSERT_EQ(Vec2d(0, 0) += Vec2d(1, 1), Vec2d(1, 1));
  ASSERT_EQ(Vec2d(-3, -2) += Vec2d(-1, -1), Vec2d(-4, -3));
  ASSERT_EQ(Vec2d(6, 3) += Vec2d(5, 5), Vec2d(11, 8));
  ASSERT_EQ(Vec2d(1, 1) += Vec2d(1, 1), Vec2d(2, 2));
  ASSERT_EQ(Vec2d(-5, 4) += Vec2d(7, -1), Vec2d(2, 3));
}

/// Test that subtracting two vectors produce the correct result.
TEST(Tests, TestVec2dSubtraction) {
  ASSERT_EQ(Vec2d(0, 0) - Vec2d(1, 1), Vec2d(-1, -1));
  ASSERT_EQ(Vec2d(-3, -2) - Vec2d(-1, -1), Vec2d(-2, -1));
  ASSERT_EQ(Vec2d(6, 3) - Vec2d(5, 5), Vec2d(1, -2));
  ASSERT_EQ(Vec2d(1, 1) - Vec2d(1, 1), Vec2d(0, 0));
  ASSERT_EQ(Vec2d(-5, 4) - Vec2d(7, -1), Vec2d(-12, 5));
}

/// Test that multiplying a vector by a scalar produces the correct result.
TEST(Tests, TestVec2dMultiplication) {
  ASSERT_EQ(Vec2d(0, 0) * 1, Vec2d(0, 0));
  ASSERT_EQ(Vec2d(-3, -2) * 2, Vec2d(-6, -4));
  ASSERT_EQ(Vec2d(6, 3) * 3, Vec2d(18, 9));
  ASSERT_EQ(Vec2d(1, 1) * 4, Vec2d(4, 4));
  ASSERT_EQ(Vec2d(-5, 4) * 5, Vec2d(-25, 20));
}

/// Test that dividing a vector by a scalar produces the correct result.
TEST(Tests, TestVec2dDivision) {
  ASSERT_EQ(Vec2d(0, 0) / 1, Vec2d(0, 0));
  ASSERT_EQ(Vec2d(-3, -2) / 2, Vec2d(-2, -1));
  ASSERT_EQ(Vec2d(6, 3) / 3, Vec2d(2, 1));
  ASSERT_EQ(Vec2d(1, 1) / 4, Vec2d(0, 0));
  ASSERT_EQ(Vec2d(-5, 4) / 5, Vec2d(-1, 0));
}

/// Test that getting the magnitude of a vector produces the correct result.
TEST(Tests, TestVec2dMagnitude) {
  ASSERT_EQ(Vec2d(0, 0).magnitude(), 0);
  ASSERT_EQ(Vec2d(-3, -2).magnitude(), 3.605551275463989);
  ASSERT_EQ(Vec2d(6, 3).magnitude(), 6.708203932499369);
  ASSERT_EQ(Vec2d(1, 1).magnitude(), 1.4142135623730951);
  ASSERT_EQ(Vec2d(-5, 4).magnitude(), 6.4031242374328485);
}

/// Test that getting the normalised vector produces the correct result.
TEST(Tests, TestVec2dNormalised) {
  ASSERT_EQ(Vec2d(0, 0).normalised(), Vec2d(0, 0));
  ASSERT_EQ(Vec2d(-3, -2).normalised(), Vec2d(-0.8320502943378437, -0.5547001962252291));
  ASSERT_EQ(Vec2d(6, 3).normalised(), Vec2d(0.8944271909999159, 0.4472135954999579));
  ASSERT_EQ(Vec2d(1, 1).normalised(), Vec2d(0.7071067811865475, 0.7071067811865475));
  ASSERT_EQ(Vec2d(-5, 4).normalised(), Vec2d(-0.7808688094430304, 0.6246950475544243));
}

/// Test that rotating a vector produces the correct result.
TEST(Tests, TestVec2dRotated) {
  ASSERT_EQ(Vec2d(0, 0).rotated(360 * PI_RADIANS), Vec2d(0, 0));
  const Vec2d rotated_result_one{Vec2d{-3, -2}.rotated(270 * PI_RADIANS)};
  ASSERT_DOUBLE_EQ(rotated_result_one.x, -2);
  ASSERT_DOUBLE_EQ(rotated_result_one.y, 3);
  const Vec2d rotated_result_two{Vec2d{6, 3}.rotated(180 * PI_RADIANS)};
  ASSERT_DOUBLE_EQ(rotated_result_two.x, -6);
  ASSERT_DOUBLE_EQ(rotated_result_two.y, -3);
  const Vec2d rotated_result_three{Vec2d{1, 1}.rotated(90 * PI_RADIANS)};
  ASSERT_DOUBLE_EQ(rotated_result_three.x, -1);
  ASSERT_DOUBLE_EQ(rotated_result_three.y, 1);
  const Vec2d rotated_result_four{Vec2d{-5, 4}.rotated(0 * PI_RADIANS)};
  ASSERT_DOUBLE_EQ(rotated_result_four.x, -5);
  ASSERT_DOUBLE_EQ(rotated_result_four.y, 4);
}

/// Test that getting the angle of a vector produces the correct result.
TEST(Tests, TestVec2dAngleBetween) {
  ASSERT_EQ(Vec2d(0, 0).angle_between({1, 1}), 0);
  ASSERT_EQ(Vec2d(-3, -2).angle_between({-1, -1}), 0.19739555984988044);
  ASSERT_EQ(Vec2d(6, 3).angle_between({5, 5}), 0.32175055439664213);
  ASSERT_EQ(Vec2d(1, 1).angle_between({1, 1}), 0);
  ASSERT_EQ(Vec2d(-5, 4).angle_between({7, -1}), 3.674436541209182);
}

/// Test that getting the distance of two vectors produces the correct result.
TEST(Tests, TestVec2dDistanceTo) {
  ASSERT_EQ(Vec2d(0, 0).distance_to({1, 1}), 1.4142135623730951);
  ASSERT_EQ(Vec2d(-3, -2).distance_to({-1, -1}), 2.23606797749979);
  ASSERT_EQ(Vec2d(6, 3).distance_to({5, 5}), 2.23606797749979);
  ASSERT_EQ(Vec2d(1, 1).distance_to({1, 1}), 0);
  ASSERT_EQ(Vec2d(-5, 4).distance_to({7, -1}), 13);
}

/// Test if a position outside the radius produces the correct arrive force.
TEST(Tests, TestArriveOutsideSlowingRange) {
  ASSERT_EQ(arrive({500, 500}, {0, 0}), Vec2d(-0.7071067811865475, -0.7071067811865475));
}

/// Test if a position on the radius produces the correct arrive force.
TEST(Tests, TestArriveOnSlowingRange) {
  ASSERT_EQ(arrive({135, 135}, {0, 0}), Vec2d(-0.7071067811865475, -0.7071067811865475));
}

/// Test if a position inside the radius produces the correct arrive force.
TEST(Tests, TestArriveInsideSlowingRange) {
  ASSERT_EQ(arrive({100, 100}, {0, 0}), Vec2d(-0.7071067811865476, -0.7071067811865476));
}

/// Test if a position near the target produces the correct arrive force.
TEST(Tests, TestArriveNearTarget) {
  ASSERT_EQ(arrive({50, 50}, {0, 0}), Vec2d(-0.7071067811865476, -0.7071067811865476));
}

/// Test if a position on the target produces the correct arrive force.
TEST(Tests, TestArriveOnTarget) { ASSERT_EQ(arrive({0, 0}, {0, 0}), Vec2d(0, 0)); }

/// Test if a non-moving target produces the correct evade force.
TEST(Tests, TestEvadeNonMovingTarget) {
  ASSERT_EQ(evade({0, 0}, {100, 100}, {0, 0}), Vec2d(-0.7071067811865475, -0.7071067811865475));
}

/// Test if a moving target produces the correct evade force.
TEST(Tests, TestEvadeMovingTarget) {
  ASSERT_EQ(evade({0, 0}, {100, 100}, {-50, 0}), Vec2d(-0.5428888213891885, -0.8398045770360255));
}

/// Test if having the same position produces the correct evade force.
TEST(Tests, TestEvadeSamePositions) {
  ASSERT_EQ(evade({0, 0}, {0, 0}, {0, 0}), Vec2d(0, 0));
  ASSERT_EQ(evade({0, 0}, {0, 0}, {-50, 0}), Vec2d(0, 0));
}

/// Test if a higher current position produces the correct flee force.
TEST(Tests, TestFleeHigherCurrent) {
  ASSERT_EQ(flee({100, 100}, {50, 50}), Vec2d(0.7071067811865475, 0.7071067811865475));
}

/// Test if a higher target position produces the correct flee force.
TEST(Tests, TestFleeHigherTarget) {
  ASSERT_EQ(flee({50, 50}, {100, 100}), Vec2d(-0.7071067811865475, -0.7071067811865475));
}

/// Test if two equal positions produce the correct flee force.
TEST(Tests, TestFleeEqual) { ASSERT_EQ(flee({100, 100}, {100, 100}), Vec2d(0, 0)); }

/// Test if a negative current position produces the correct flee force.
TEST(Tests, TestFleeNegativeCurrent) {
  ASSERT_EQ(flee({-50, -50}, {100, 100}), Vec2d(-0.7071067811865475, -0.7071067811865475));
}

/// Test if a negative target position produces the correct flee force.
TEST(Tests, TestFleeNegativeTarget) {
  ASSERT_EQ(flee({100, 100}, {-50, -50}), Vec2d(0.7071067811865475, 0.7071067811865475));
}

/// Test if two negative positions produce the correct flee force.
TEST(Tests, TestFleeNegativePositions) { ASSERT_EQ(flee({-50, -50}, {-50, -50}), Vec2d(0, 0)); }

/// Test if a multiple position list produces the correct follow path force.
TEST(Tests, TestFollowPathSinglePosition) {
  std::vector<Vec2d> path_list{{250, 250}};
  ASSERT_EQ(follow_path({100, 100}, path_list), Vec2d(0.7071067811865475, 0.7071067811865475));
}

/// Test if reaching a position in a single position list produces the correct follow path force.
TEST(Tests, TestFollowPathSinglePositionReached) {
  std::vector<Vec2d> path_list{{100, 100}};
  ASSERT_EQ(follow_path({100, 100}, path_list), Vec2d(0, 0));
  ASSERT_EQ(path_list, std::vector<Vec2d>{Vec2d(100, 100)});
}

/// Test if a multiple position list produces the correct follow path force.
TEST(Tests, TestFollowPathMultiplePositions) {
  std::vector<Vec2d> path_list{{350, 350}, {500, 500}};
  ASSERT_EQ(follow_path({200, 200}, path_list), Vec2d(0.7071067811865475, 0.7071067811865475));
}

/// Test if reaching a position in a multiple position list produces the correct follow path force.
TEST(Tests, TestFollowPathMultiplePositionsReached) {
  std::vector<Vec2d> path_list{{100, 100}, {250, 250}};
  ASSERT_EQ(follow_path({100, 100}, path_list), Vec2d(0.7071067811865475, 0.7071067811865475));
  ASSERT_EQ(path_list, std::vector<Vec2d>({Vec2d(250, 250), Vec2d(100, 100)}));
}

/// Test if an empty list raises the correct exception.
TEST(Tests, TestFollowPathEmptyList) {
  std::vector<Vec2d> path_list;
  ASSERT_THROW_MESSAGE(follow_path({100, 100}, path_list), std::length_error, "The path list is empty")
}

/// Test if no obstacles produce the correct obstacle avoidance force.
TEST(Tests, TestObstacleAvoidanceNoObstacles) { ASSERT_EQ(obstacle_avoidance({100, 100}, {0, 100}, {}), Vec2d(0, 0)); }

/// Test if an out of range obstacle produces the correct obstacle avoidance force.
TEST(Tests, TestObstacleAvoidanceObstacleOutOfRange) {
  ASSERT_EQ(obstacle_avoidance({100, 100}, {0, 100}, {{10, 10}}), Vec2d(0, 0));
}

/// Test if an angled velocity produces the correct obstacle avoidance force.
TEST(Tests, TestObstacleAvoidanceAngledVelocity) {
  ASSERT_EQ(obstacle_avoidance({100, 100}, {100, 100}, {{1, 2}}), Vec2d(0.2588190451025206, -0.9659258262890683));
}

/// Test if a non-moving game object produces the correct obstacle avoidance force.
TEST(Tests, TestObstacleAvoidanceNonMoving) {
  ASSERT_EQ(obstacle_avoidance({100, 100}, {0, 100}, {{1, 2}}), Vec2d(0, 0));
}

/// Test if a single forward obstacle produces the correct obstacle avoidance force.
TEST(Tests, TestObstacleAvoidanceSingleForward) {
  ASSERT_EQ(obstacle_avoidance({100, 100}, {0, 100}, {{1, 2}}), Vec2d(0, 0));
}

/// Test if a single left obstacle produces the correct obstacle avoidance force.
TEST(Tests, TestObstacleAvoidanceSingleLeft) {
  // This is due to floating point precision
  const Vec2d single_left_result{obstacle_avoidance({100, 100}, {0, 100}, {{0, 2}})};
  ASSERT_DOUBLE_EQ(single_left_result.x, 0.8660254037844387);
  ASSERT_DOUBLE_EQ(single_left_result.y, -0.5);
}

/// Test if a single right obstacle produces the correct obstacle avoidance force.
TEST(Tests, TestObstacleAvoidanceSingleRight) {
  // This is due to floating point precision
  const Vec2d single_right_result{obstacle_avoidance({100, 100}, {0, 100}, {{2, 2}})};
  ASSERT_DOUBLE_EQ(single_right_result.x, -0.8660254037844386);
  ASSERT_DOUBLE_EQ(single_right_result.y, -0.5);
}

/// Test if a left and forward obstacle produces the correct obstacle avoidance force.
TEST(Tests, TestObstacleAvoidanceLeftForward) {
  // This is due to floating point precision
  const Vec2d left_forward_result{obstacle_avoidance({100, 100}, {0, 100}, {{0, 2}, {1, 2}})};
  ASSERT_DOUBLE_EQ(left_forward_result.x, 0.8660254037844387);
  ASSERT_DOUBLE_EQ(left_forward_result.y, -0.5);
}

/// Test if a right and forward obstacle produces the correct obstacle avoidance force.
TEST(Tests, TestObstacleAvoidanceRightForward) {
  // This is due to floating point precision
  const Vec2d right_forward_result{obstacle_avoidance({100, 100}, {0, 100}, {{1, 2}, {2, 2}})};
  ASSERT_DOUBLE_EQ(right_forward_result.x, -0.8660254037844386);
  ASSERT_DOUBLE_EQ(right_forward_result.y, -0.5);
}

/// Test if all three obstacles produce the correct obstacle avoidance force.
TEST(Tests, TestObstacleAvoidanceLeftRightForward) {
  ASSERT_EQ(obstacle_avoidance({100, 100}, {0, 100}, {{0, 2}, {1, 2}, {2, 2}}), Vec2d(0, -1));
}

/// Test if a non-moving target produces the correct pursue force.
TEST(Tests, TestPursueNonMovingTarget) {
  ASSERT_EQ(pursue({0, 0}, {100, 100}, {0, 0}), Vec2d(0.7071067811865475, 0.7071067811865475));
}

/// Test if a moving target produces the correct pursue force.
TEST(Tests, TestPursueMovingTarget) {
  ASSERT_EQ(pursue({0, 0}, {100, 100}, {-50, 0}), Vec2d(0.5428888213891885, 0.8398045770360255));
}

/// Test if having the same position produces the correct pursue force.
TEST(Tests, TestPursueSamePositions) {
  ASSERT_EQ(pursue({0, 0}, {0, 0}, {0, 0}), Vec2d(0, 0));
  ASSERT_EQ(pursue({0, 0}, {0, 0}, {-50, 0}), Vec2d(0, 0));
}

/// Test if a higher current position produces the correct seek force.
TEST(Tests, TestSeekHigherCurrent) {
  ASSERT_EQ(seek({100, 100}, {50, 50}), Vec2d(-0.7071067811865475, -0.7071067811865475));
}

/// Test if a higher target position produces the correct seek force.
TEST(Tests, TestSeekHigherTarget) {
  ASSERT_EQ(seek({50, 50}, {100, 100}), Vec2d(0.7071067811865475, 0.7071067811865475));
}

/// Test if two equal positions produce the correct seek force.
TEST(Tests, TestSeekEqual) { ASSERT_EQ(seek({100, 100}, {100, 100}), Vec2d(0, 0)); }

/// Test if a negative current position produces the correct seek force.
TEST(Tests, TestSeekNegativeCurrent) {
  ASSERT_EQ(seek({-50, -50}, {100, 100}), Vec2d(0.7071067811865475, 0.7071067811865475));
}

/// Test if a negative target position produces the correct seek force.
TEST(Tests, TestSeekNegativeTarget) {
  ASSERT_EQ(seek({100, 100}, {-50, -50}), Vec2d(-0.7071067811865475, -0.7071067811865475));
}

/// Test if two negative positions produce the correct seek force.
TEST(Tests, TestSeekNegativePositions) { ASSERT_EQ(seek({-50, -50}, {-50, -50}), Vec2d(0, 0)); }

/// Test if a non-moving game object produces the correct wander force.
TEST(Tests, TestWanderNonMoving) {
  // This is due to floating point precision
  const Vec2d non_moving_result{wander({0, 0}, 60)};
  ASSERT_DOUBLE_EQ(non_moving_result.x, 0.8660254037844385);
  ASSERT_DOUBLE_EQ(non_moving_result.y, -0.5);
}

/// Test if a moving game object produces the correct wander force.
TEST(Tests, TestWanderMoving) { ASSERT_EQ(wander({100, -100}, 60), Vec2d(0.7659012135559103, -0.6429582654213131)); }

/// Test if a zero angle produces the correct wander force.
TEST(Tests, TestWanderZeroAngle) { ASSERT_EQ(wander({0, 0}, 0), Vec2d(0, -1)); }
