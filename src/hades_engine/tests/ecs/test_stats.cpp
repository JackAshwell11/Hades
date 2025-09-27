// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "macros.hpp"

/// Represents a test stat useful for testing the stat serialisation and deserialisation.
struct TestStat final : Stat {
  /// Initialise the object.
  TestStat() : Stat(200, -1) {}

  /// Serialise the component to a JSON object.
  ///
  /// @param json - The JSON object to serialise to.
  void to_file(nlohmann::json& json) const override { to_file_base(json["test_stat"]); }

  /// Deserialise the component from a JSON object.
  ///
  /// @param json - The JSON object to deserialise from.
  void from_file(const nlohmann::json& json) override { from_file_base(json.at("test_stat")); }
};

/// Implements the fixture for the ecs/stats.hpp tests.
class StatsFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {}
};

/// Test that the stat component is reset correctly.
TEST_F(StatsFixture, TestStatReset) {
  const auto test_stat{std::make_shared<TestStat>()};
  test_stat->set_value(100);
  test_stat->reset();
  ASSERT_EQ(test_stat->get_value(), 200);
  ASSERT_EQ(test_stat->get_max_value(), 200);
  ASSERT_EQ(test_stat->get_max_level(), -1);
  ASSERT_EQ(test_stat->get_current_level(), 0);
}

/// Test that serialising to a JSON object which already contains the key for the stat works correctly.
TEST_F(StatsFixture, TestStatToFileExistingKey) {
  nlohmann::json json(nlohmann::json::parse(R"({"test_stat":{}})"));
  const auto test_stat{std::make_shared<TestStat>()};
  test_stat->to_file(json);
  ASSERT_EQ(json,
            nlohmann::json::parse(R"({"test_stat":{"value":200,"max_level":-1,"max_value":200,"current_level":0}})"));
}

/// Test that deserialising from a JSON object which is missing the serialised data throws an exception.
TEST_F(StatsFixture, TestStatFromFileMissingKey) {
  const nlohmann::json json(
      nlohmann::json::parse(R"({"test":{"value":100,"max_level":-1,"max_value":100,"current_level":0}})"));
  const auto test_stat{std::make_shared<TestStat>()};
  ASSERT_THROW_MESSAGE(test_stat->from_file(json), nlohmann::json::exception,
                       "[json.exception.out_of_range.403] key 'test_stat' not found");
}

/// Test that the armour component is serialised to a file correctly.
TEST_F(StatsFixture, TestArmourToFile) {
  nlohmann::json json;
  const auto armour{std::make_shared<Armour>(50, -1)};
  armour->to_file(json);
  ASSERT_EQ(json, nlohmann::json::parse(R"({"armour":{"value":50,"max_level":-1,"max_value":50,"current_level":0}})"));
}

/// Test that the armour component is deserialised from a file correctly.
TEST_F(StatsFixture, TestArmourFromFile) {
  const nlohmann::json json(
      nlohmann::json::parse(R"({"armour":{"value":75,"max_level":-1,"max_value":75,"current_level":0}})"));
  const auto armour{std::make_shared<Armour>(50, -1)};
  armour->from_file(json);
  ASSERT_EQ(armour->get_value(), 75);
  ASSERT_EQ(armour->get_max_value(), 75);
  ASSERT_EQ(armour->get_max_level(), -1);
  ASSERT_EQ(armour->get_current_level(), 0);
}

/// Test that the health component is serialised to a file correctly.
TEST_F(StatsFixture, TestHealthToFile) {
  nlohmann::json json;
  const auto health{std::make_shared<Health>(100, -1)};
  health->to_file(json);
  ASSERT_EQ(json,
            nlohmann::json::parse(R"({"health":{"value":100,"max_level":-1,"max_value":100,"current_level":0}})"));
}

/// Test that the health component is deserialised from a file correctly.
TEST_F(StatsFixture, TestHealthFromFile) {
  const nlohmann::json json(
      nlohmann::json::parse(R"({"health":{"value":150,"max_level":-1,"max_value":150,"current_level":0}})"));
  const auto health{std::make_shared<Health>(100, -1)};
  health->from_file(json);
  ASSERT_EQ(health->get_value(), 150);
  ASSERT_EQ(health->get_max_value(), 150);
  ASSERT_EQ(health->get_max_level(), -1);
  ASSERT_EQ(health->get_current_level(), 0);
}
