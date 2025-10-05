// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/level.hpp"
#include "ecs/systems/shop.hpp"
#include "events.hpp"
#include "macros.hpp"

/// Implements the fixture for the ShopSystem tests.
class ShopSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.add_system<ShopSystem>();
    const auto game_object_id{registry.create_game_object(GameObjectType::Player)};
    registry.add_component<Health>(game_object_id, 200);
    registry.add_component<Money>(game_object_id);
    registry.add_component<PlayerLevel>(game_object_id);
  }

  /// Tear down the fixture after the tests.
  void TearDown() override { clear_listeners(); }

  /// Add an offering to the shop system.
  void add_offering() const {
    std::istringstream json_stream(R"([
    {
      "type": "component",
      "name": "Test Component Unlock",
      "description": "A test component unlock offering",
      "icon_type": "TestComponent",
      "base_cost": 100,
      "cost_multiplier": 1.5
    }])");
    get_shop_system()->add_offerings(json_stream, 0);
  }

  /// Get the shop system from the registry.
  ///
  /// @return The shop system.
  [[nodiscard]] auto get_shop_system() const -> std::shared_ptr<ShopSystem> {
    return registry.get_system<ShopSystem>();
  }
};

/// Test that adding a component unlock offering to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsComponentSingle) {
  add_offering();
  ASSERT_EQ(get_shop_system()->get_offering(0).name, "Test Component Unlock");
  ASSERT_THROW_MESSAGE(get_shop_system()->get_offering(1), std::out_of_range, "Offering index out of range");
}

/// Test that adding multiple component unlock offerings to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsComponentMultiple) {
  std::istringstream json_stream(R"([
  {
    "type": "component",
    "name": "Test Component Unlock 2",
    "description": "A test component unlock offering 2",
    "icon_type": "TestComponent",
    "base_cost": 150,
    "cost_multiplier": 1.8
  }])");
  add_offering();
  get_shop_system()->add_offerings(json_stream, 0);
  ASSERT_EQ(get_shop_system()->get_offering(0).name, "Test Component Unlock");
  ASSERT_EQ(get_shop_system()->get_offering(1).name, "Test Component Unlock 2");
  ASSERT_THROW_MESSAGE(get_shop_system()->get_offering(2), std::out_of_range, "Offering index out of range");
}

/// Test that the shop item loaded callback is called correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsItemLoadedCallback) {
  std::tuple<int, std::tuple<std::string, std::string, std::string>, int> callback_args;
  add_callback<EventType::ShopItemLoaded>(
      [&callback_args](const int offering_index, const std::tuple<std::string, std::string, std::string>& data,
                       const int cost) { callback_args = std::make_tuple(offering_index, data, cost); });
  add_offering();
  ASSERT_EQ(std::get<0>(callback_args), 0);
  ASSERT_EQ(std::get<1>(callback_args),
            std::make_tuple("Test Component Unlock", "A test component unlock offering", "TestComponent"));
  ASSERT_EQ(std::get<2>(callback_args), 101);
}

/// Test that adding an offering with an unknown type throws an exception.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsUnknownType) {
  std::istringstream json_stream(R"([
  {
    "type": "unknown",
    "name": "Mystery Offering",
    "description": "An offering of unknown type",
    "icon_type": "Mystery",
    "base_cost": 100,
    "cost_multiplier": 1.5
  }])");
  ASSERT_THROW_MESSAGE(get_shop_system()->add_offerings(json_stream, 0), std::runtime_error,
                       "Unknown offering type: unknown");
}

/// Test that adding an offering with an empty stream throws an exception.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsEmptyStream) {
  std::istringstream json_stream("");
  ASSERT_THROW_MESSAGE(get_shop_system()->add_offerings(json_stream, 0), nlohmann::json::exception,
                       "[json.exception.parse_error.101] parse error at line 1, column 1: attempting to parse an empty "
                       "input; check that your input string or stream contains the expected JSON");
}

/// Test that getting the cost of a component unlock offering works correctly.
TEST_F(ShopSystemFixture, TestShopSystemGetComponentUnlockOfferingCost) {
  add_offering();
  ASSERT_EQ(get_shop_system()->get_offering(0).get_cost(&registry, 0), 101);
}

/// Test that purchasing a component unlock offering works correctly if the player has enough money.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseComponentUnlockSufficientMoney) {
  add_offering();
  registry.get_component<Money>(0)->money = 500;
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_EQ(registry.get_component<Money>(0)->money, 399);
}

/// Test that purchasing a component unlock offering does not work if the player does not have enough money.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseComponentUnlockInsufficientMoney) {
  add_offering();
  ASSERT_FALSE(get_shop_system()->purchase(0, 0));
}

// /// Test that purchasing a component unlock offering does not work if the player already has the component.
// TEST_F(ShopSystemFixture, TestShopSystemPurchaseComponentUnlockAlreadyUnlocked) {
//   add_offering();
//   registry.get_component<Money>(0)->money = 500;
//   ASSERT_FALSE(get_shop_system()->purchase(0, 0));
// }

/// Test that the shop item purchased callback is called correctly.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseItemShopItemPurchasedCallback) {
  add_offering();
  registry.get_component<Money>(0)->money = 500;
  std::tuple<int, int> callback_args;
  add_callback<EventType::ShopItemPurchased>(
      [&](const int offering_index, const int cost) { callback_args = std::make_tuple(offering_index, cost); });
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_EQ(std::get<0>(callback_args), 0);
  ASSERT_EQ(std::get<1>(callback_args), 101);
}

/// Test that nothing happens if the offering index is invalid.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseInvalidOfferingIndex) {
  registry.get_component<Money>(0)->money = 500;
  ASSERT_FALSE(get_shop_system()->purchase(0, 0));
}

/// Test that an exception is thrown if the game object does not have a money component.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseNoMoneyComponent) {
  const auto game_object_id{registry.create_game_object(GameObjectType::Player)};
  registry.add_component<Health>(game_object_id, 0);
  add_offering();
  ASSERT_THROW_MESSAGE(get_shop_system()->purchase(1, 0), RegistryError,
                       "The component `Money` for the game object ID `1` is not registered with the registry.");
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseInvalidGameObjectId) {
  add_offering();
  ASSERT_THROW_MESSAGE(get_shop_system()->purchase(-1, 0), RegistryError,
                       "The component `Money` for the game object ID `-1` is not registered with the registry.");
}
