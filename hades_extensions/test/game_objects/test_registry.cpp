// External includes
#include "gtest/gtest.h"

// Custom includes
#include "fixtures.hpp"
#include "game_objects/registry.hpp"

// ----- TESTS ------------------------------
TEST_F(GameObjectsFixtures, TestRegistryEmptyGameObject) {
  ASSERT_EQ(registry.create_game_object(false, {}), 0);
  ASSERT_EQ(registry.get_component<TestGameObjectComponentOne>(0), nullptr);
  ASSERT_EQ(registry.get_component<TestGameObjectComponentTwo>(0), nullptr);
  ASSERT_EQ(registry.get_components<TestGameObjectComponentOne>().size(), {});
  ASSERT_EQ(registry.get_components<TestGameObjectComponentTwo>().size(), {});
  ASSERT_EQ(registry.get_walls().size(), 0);
  ASSERT_EQ(registry.get_kinematic_object(0), nullptr);
}

TEST_F(GameObjectsFixtures, TestRegistryGameObjectComponents) {
  // Construct the components list
  std::vector<std::unique_ptr<ComponentBase>> components;
  components.emplace_back(create_object<TestGameObjectComponentOne>());
  components.emplace_back(create_object<TestGameObjectComponentTwo>({10}));

  // Test that creating the game object works correctly
  registry.create_game_object(false, std::move(components));
  ASSERT_TRUE(registry.get_component<TestGameObjectComponentOne>(0) != nullptr);
  ASSERT_TRUE(registry.get_component<TestGameObjectComponentTwo>(0) != nullptr);
  ASSERT_EQ(registry.get_components<TestGameObjectComponentOne>().size(), 1);
  ASSERT_EQ(registry.get_components<TestGameObjectComponentTwo>().size(), 1);
  auto multiple_result_one = registry.get_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size();
  ASSERT_EQ(multiple_result_one, 1);

  // Test that deleting the game object works correctly
  registry.delete_game_object(0);
  ASSERT_EQ(registry.get_component<TestGameObjectComponentOne>(0), nullptr);
  ASSERT_EQ(registry.get_component<TestGameObjectComponentTwo>(0), nullptr);
  ASSERT_EQ(registry.get_components<TestGameObjectComponentOne>().size(), 0);
  ASSERT_EQ(registry.get_components<TestGameObjectComponentTwo>().size(), 0);
  auto multiple_result_two = registry.get_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size();
  ASSERT_EQ(multiple_result_two, 0);
}

TEST_F(GameObjectsFixtures, TestRegistryGameObjectKinematic) {
  // Test that creating the kinematic game object works correctly
  registry.create_game_object(true, {});
  std::unique_ptr<KinematicObject> kinematic_object = registry.get_kinematic_object(0);
  ASSERT_EQ(kinematic_object->position, Vec2d(0, 0));
  ASSERT_EQ(kinematic_object->velocity, Vec2d(0, 0));
  ASSERT_EQ(kinematic_object->rotation, 0);

  // Test that deleting the kinematic game object works correctly
  registry.delete_game_object(0);
  ASSERT_EQ(registry.get_kinematic_object(0), nullptr);
}

TEST_F(GameObjectsFixtures, TestRegistryMultipleGameObjects) {
  // Construct the components list for both game objects
  std::vector<std::unique_ptr<ComponentBase>> components_one, components_two;
  components_one.emplace_back(create_object<TestGameObjectComponentOne>());
  components_two.emplace_back(create_object<TestGameObjectComponentOne>());
  components_two.emplace_back(create_object<TestGameObjectComponentTwo>({10}));

  // Test that creating two game objects works correctly
  ASSERT_EQ(registry.create_game_object(false, std::move(components_one)), 0);
  ASSERT_EQ(registry.create_game_object(false, std::move(components_two)), 1);
  ASSERT_EQ(registry.get_components<TestGameObjectComponentOne>().size(), 2);
  ASSERT_EQ(registry.get_components<TestGameObjectComponentTwo>().size(), 1);
  auto multiple_result_one = registry.get_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size();
  ASSERT_EQ(multiple_result_one, 1);

  // Test that deleting the first game object works correctly
  registry.delete_game_object(0);
  ASSERT_EQ(registry.get_components<TestGameObjectComponentOne>().size(), 1);
  ASSERT_EQ(registry.get_components<TestGameObjectComponentTwo>().size(), 1);
  auto multiple_result_two = registry.get_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size();
  ASSERT_EQ(multiple_result_two, 1);
}

TEST_F(GameObjectsFixtures, TestRegistryGameObjectDuplicateComponents) {
  // Construct the components list
  std::vector<std::unique_ptr<ComponentBase>> components;
  components.emplace_back(create_object<TestGameObjectComponentTwo>({10}));
  components.emplace_back(create_object<TestGameObjectComponentTwo>({20}));

  // Test that creating a game object with two of the same components only adds
  // the first one
  registry.create_game_object(false, std::move(components));
  ASSERT_EQ(registry.get_component<TestGameObjectComponentTwo>(0)->test_list[0], 10);
}

TEST_F(GameObjectsFixtures, TestRegistryZeroSystems) {
  ASSERT_EQ(registry.get_system<TestSystem>(), nullptr);
}

TEST_F(GameObjectsFixtures, TestRegistrySystemUpdate) {
  registry.add_system<TestSystem>(create_object<TestSystem>());
  auto system_result = registry.get_system<TestSystem>();
  ASSERT_TRUE(system_result != nullptr);
  registry.update(0);
  ASSERT_TRUE(system_result->called);
}

TEST_F(GameObjectsFixtures, TestRegistryWalls) {
  ASSERT_EQ(registry.get_walls(), std::unordered_set<Vec2d>());
  registry.add_wall({0, 0});
  registry.add_wall({1, 1});
  registry.add_wall({0, 0});
  ASSERT_EQ(registry.get_walls(), std::unordered_set<Vec2d>({{0, 0}, {1, 1}}));
}
