// Local headers
#include "game_objects/registry.hpp"
#include "macros.hpp"

// ----- COMPONENTS ------------------------------
/// Represents a game object component useful for testing.
struct TestGameObjectComponentOne : public ComponentBase {};

/// Represents a game object component with data useful for testing.
struct TestGameObjectComponentTwo : public ComponentBase {
  /// A test list of integers.
  std::vector<int> test_list;

  /// Initialise the object.
  ///
  /// @param test_lst - The list to be used for testing.
  explicit TestGameObjectComponentTwo(const std::vector<int> &test_lst) : test_list(test_lst) {}
};

// ----- SYSTEMS --------------------------------
/// Represents a test system useful for testing.
struct TestSystem : public SystemBase {
  /// Whether the system has been called or not.
  mutable bool called{false};

  /// Initialise the system.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit TestSystem(Registry *registry) : SystemBase(registry) {}

  /// Update the system.
  void update(const double /*delta_time*/) const final { called = true; }
};

// ----- FIXTURES ------------------------------
/// Implements the fixture for the game_objects/registry.hpp tests.
class RegistryFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry{};
};

// ----- TESTS ------------------------------
/// Test that an exception is thrown if a component is not registered.
TEST_F(RegistryFixture, TestRegistryEmptyGameObject) {
  ASSERT_EQ(registry.create_game_object({}), 0);
  ASSERT_THROW_MESSAGE(
      registry.get_component<TestGameObjectComponentOne>(0), RegistryError,
      "The game object `0` is not registered with the registry or does not have the required component.")
  ASSERT_THROW_MESSAGE(
      (registry.get_component(0, typeid(TestGameObjectComponentTwo))), RegistryError,
      "The game object `0` is not registered with the registry or does not have the required component.")
  ASSERT_EQ(registry.find_components<TestGameObjectComponentOne>().size(), {});
  ASSERT_EQ(registry.find_components<TestGameObjectComponentTwo>().size(), {});
  ASSERT_EQ(registry.get_walls().size(), 0);
  ASSERT_THROW_MESSAGE((registry.get_kinematic_object(0)), RegistryError,
                       "The game object `0` is not registered with the registry or is not kinematic.")
  registry.delete_game_object(0);
  ASSERT_THROW_MESSAGE(registry.delete_game_object(0), RegistryError,
                       "The game object `0` is not registered with the registry.")
}

/// Test that multiple components are added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectComponents) {
  // Test that creating the game object works correctly
  const std::vector<int> test_list{10};
  registry.create_game_object(
      {std::make_shared<TestGameObjectComponentOne>(), std::make_shared<TestGameObjectComponentTwo>(test_list)});
  ASSERT_NE(registry.get_component<TestGameObjectComponentOne>(0), nullptr);
  ASSERT_NE(registry.get_component(0, typeid(TestGameObjectComponentTwo)), nullptr);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentOne>().size(), 1);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentTwo>().size(), 1);
  auto multiple_result_one{registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size()};
  ASSERT_EQ(multiple_result_one, 1);

  // Test that deleting the game object works correctly
  registry.delete_game_object(0);
  ASSERT_THROW_MESSAGE(
      registry.get_component<TestGameObjectComponentOne>(0), RegistryError,
      "The game object `0` is not registered with the registry or does not have the required component.")
  ASSERT_THROW_MESSAGE(
      (registry.get_component(0, typeid(TestGameObjectComponentTwo))), RegistryError,
      "The game object `0` is not registered with the registry or does not have the required component.")
  ASSERT_EQ(registry.find_components<TestGameObjectComponentOne>().size(), 0);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentTwo>().size(), 0);
  auto multiple_result_two{registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size()};
  ASSERT_EQ(multiple_result_two, 0);
}

/// Test that a kinematic game object is added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectKinematic) {
  // Test that creating the kinematic game object works correctly
  registry.create_game_object({}, true);
  const std::shared_ptr<KinematicObject> kinematic_object{registry.get_kinematic_object(0)};
  ASSERT_EQ(kinematic_object->position, Vec2d(0, 0));
  ASSERT_EQ(kinematic_object->velocity, Vec2d(0, 0));
  ASSERT_EQ(kinematic_object->rotation, 0);

  // Test that deleting the kinematic game object works correctly
  registry.delete_game_object(0);
  ASSERT_THROW_MESSAGE((registry.get_kinematic_object(0)), RegistryError,
                       "The game object `0` is not registered with the registry or is not kinematic.")
}

/// Test that multiple game objects are added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryMultipleGameObjects) {
  // Test that creating two game objects works correctly
  const std::vector<int> test_list{10};
  ASSERT_EQ(registry.create_game_object({std::make_shared<TestGameObjectComponentOne>()}), 0);
  ASSERT_EQ(registry.create_game_object({std::make_shared<TestGameObjectComponentOne>(),
                                         std::make_shared<TestGameObjectComponentTwo>(test_list)}),
            1);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentOne>().size(), 2);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentTwo>().size(), 1);
  auto multiple_result_one{registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size()};
  ASSERT_EQ(multiple_result_one, 1);

  // Test that deleting the first game object works correctly
  registry.delete_game_object(0);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentOne>().size(), 1);
  ASSERT_EQ(registry.find_components<TestGameObjectComponentTwo>().size(), 1);
  auto multiple_result_two{registry.find_components<TestGameObjectComponentOne, TestGameObjectComponentTwo>().size()};
  ASSERT_EQ(multiple_result_two, 1);
}

/// Test that a game object with duplicate components is added to the registry correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectDuplicateComponents) {
  // Test that creating a game object with two of the same components only adds the first one
  const std::vector<int> test_list_one{10};
  const std::vector<int> test_list_two{20};
  registry.create_game_object({std::make_shared<TestGameObjectComponentTwo>(test_list_one),
                               std::make_shared<TestGameObjectComponentTwo>(test_list_two)});
  ASSERT_EQ(registry.get_component<TestGameObjectComponentTwo>(0)->test_list[0], 10);
}

/// Test that passing the same component to multiple game objects works correctly.
TEST_F(RegistryFixture, TestRegistryGameObjectSameComponent) {
  const std::vector<int> test_list{10};
  const std::shared_ptr<TestGameObjectComponentTwo> component_one{
      std::make_shared<TestGameObjectComponentTwo>(test_list)};
  registry.create_game_object({component_one});
  registry.create_game_object({component_one});
  registry.get_component<TestGameObjectComponentTwo>(0)->test_list[0] = 20;
  ASSERT_EQ(registry.get_component<TestGameObjectComponentTwo>(0)->test_list[0], 20);
  ASSERT_EQ(registry.get_component<TestGameObjectComponentTwo>(1)->test_list[0], 20);
}

/// Test that an exception is thrown if a system is not registered.
TEST_F(RegistryFixture,
       TestRegistryZeroSystems){ASSERT_THROW_MESSAGE(registry.get_system<TestSystem>(), RegistryError,
                                                     "The templated type is not registered with the registry.")}

/// Test that a system is updated correctly.
TEST_F(RegistryFixture, TestRegistrySystemUpdate) {
  // Test that the system is added correctly
  const std::vector<int> test_list{10};
  registry.create_game_object({std::make_shared<TestGameObjectComponentTwo>(test_list)});
  registry.add_system<TestSystem>();
  ASSERT_THROW_MESSAGE(registry.add_system<TestSystem>(), RegistryError,
                       "The templated type is already registered with the registry.")
  auto system_result{registry.get_system<TestSystem>()};
  ASSERT_NE(system_result, nullptr);

  // Test that the system is updated correctly
  registry.update(0);
  ASSERT_TRUE(system_result->called);
}

/// Test that multiple walls are added to the system correctly.
TEST_F(RegistryFixture, TestRegistryWalls) {
  ASSERT_EQ(registry.get_walls(), std::unordered_set<Vec2d>());
  registry.add_wall({0, 0});
  registry.add_wall({1, 1});
  registry.add_wall({0, 0});
  ASSERT_EQ(registry.get_walls(), std::unordered_set<Vec2d>({{0, 0}, {1, 1}}));
}
