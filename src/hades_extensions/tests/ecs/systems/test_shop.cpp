// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
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
    registry.create_game_object(GameObjectType::Player, cpvzero,
                                {std::make_shared<Money>(), std::make_shared<Health>(200, 5)});
  }

  /// Tear down the fixture after the tests.
  void TearDown() override { clear_listeners(); }

  /// Add a stat upgrade offering to the shop system.
  void add_stat_upgrade() const {
    std::istringstream json_stream(R"([
    {
      "type": "stat",
      "name": "Test Stat Upgrade",
      "description": "A test stat upgrade offering",
      "icon_type": "StatTest",
      "base_cost": 100,
      "cost_multiplier": 1.5,
      "stat_type": "Health",
      "base_value": 200,
      "value_multiplier": 1.2
    }])");
    get_shop_system()->add_offerings(json_stream, 0);
  }

  /// Add a component unlock offering to the shop system.
  void add_component_unlock() const {
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

  /// Add an item offering to the shop system.
  void add_item() const {
    std::istringstream json_stream(R"([
    {
      "type": "item",
      "name": "Test Item Offering",
      "description": "A test item offering",
      "icon_type": "TestItem",
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

/// Test that adding a stat upgrade offering to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsStatSingle) {
  add_stat_upgrade();
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Stat Upgrade");
  ASSERT_EQ(get_shop_system()->get_offering(1), nullptr);
}

/// Test that adding multiple stat upgrade offerings to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsStatMultiple) {
  std::istringstream json_stream(R"([
  {
    "type": "stat",
    "name": "Test Stat Upgrade 2",
    "description": "A test stat upgrade offering 2",
    "icon_type": "StatTest2",
    "base_cost": 150,
    "cost_multiplier": 1.8,
    "stat_type": "Health",
    "base_value": 250,
    "value_multiplier": 1.5
  }])");
  add_stat_upgrade();
  get_shop_system()->add_offerings(json_stream, 0);
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Stat Upgrade");
  ASSERT_EQ(get_shop_system()->get_offering(1)->name, "Test Stat Upgrade 2");
  ASSERT_EQ(get_shop_system()->get_offering(2), nullptr);
}

/// Test that adding a component unlock offering to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsComponentSingle) {
  add_component_unlock();
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Component Unlock");
  ASSERT_EQ(get_shop_system()->get_offering(1), nullptr);
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
  add_component_unlock();
  get_shop_system()->add_offerings(json_stream, 0);
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Component Unlock");
  ASSERT_EQ(get_shop_system()->get_offering(1)->name, "Test Component Unlock 2");
  ASSERT_EQ(get_shop_system()->get_offering(2), nullptr);
}

/// Test that adding an item offering to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsItemSingle) {
  add_item();
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Item Offering");
  ASSERT_EQ(get_shop_system()->get_offering(1), nullptr);
}

/// Test that adding multiple item offerings to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsItemMultiple) {
  std::istringstream json_stream(R"([
  {
    "type": "item",
    "name": "Test Item Offering 2",
    "description": "A test item offering 2",
    "icon_type": "TestItem",
    "base_cost": 150,
    "cost_multiplier": 1.8
  }])");
  add_item();
  get_shop_system()->add_offerings(json_stream, 0);
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Item Offering");
  ASSERT_EQ(get_shop_system()->get_offering(1)->name, "Test Item Offering 2");
  ASSERT_EQ(get_shop_system()->get_offering(2), nullptr);
}

/// Test that adding multiple different offerings to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsMultipleTypes) {
  add_stat_upgrade();
  add_component_unlock();
  add_item();

  // Check the stat upgrade offering
  const auto stat_offering{get_shop_system()->get_offering(0)};
  ASSERT_EQ(stat_offering->name, "Test Stat Upgrade");
  ASSERT_EQ(stat_offering->get_cost(&registry, 0), 100);

  // Check the component unlock offering
  const auto component_offering{get_shop_system()->get_offering(1)};
  ASSERT_EQ(component_offering->name, "Test Component Unlock");
  ASSERT_EQ(component_offering->get_cost(&registry, 0), 100);

  // Check the item offering
  const auto item_offering{get_shop_system()->get_offering(2)};
  ASSERT_EQ(item_offering->name, "Test Item Offering");
  ASSERT_EQ(item_offering->get_cost(&registry, 0), 100);

  // Ensure there are no more offerings
  ASSERT_EQ(get_shop_system()->get_offering(3), nullptr);
}

/// Test that the shop item loaded callback is called correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsItemLoadedCallback) {
  std::tuple<int, std::tuple<std::string, std::string, std::string>, int> callback_args;
  add_callback<EventType::ShopItemLoaded>(
      [&callback_args](const int offering_index, const std::tuple<std::string, std::string, std::string> &data,
                       const int cost) { callback_args = std::make_tuple(offering_index, data, cost); });
  add_stat_upgrade();
  ASSERT_EQ(std::get<0>(callback_args), 0);
  ASSERT_EQ(std::get<1>(callback_args),
            std::make_tuple("Test Stat Upgrade", "A test stat upgrade offering", "StatTest"));
  ASSERT_EQ(std::get<2>(callback_args), 100);
}

/// Test that adding a stat upgrade offering with an invalid stat type throws an exception.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsStatInvalidType) {
  std::istringstream json_stream(R"([
  {
    "type": "stat",
    "name": "Invalid Stat Upgrade",
    "description": "An invalid stat upgrade offering",
    "icon_type": "StatInvalid",
    "base_cost": 100,
    "cost_multiplier": 1.5,
    "stat_type": "InvalidStat",
    "base_value": 200,
    "value_multiplier": 1.2
  }])");
  ASSERT_THROW_MESSAGE(get_shop_system()->add_offerings(json_stream, 0), std::runtime_error,
                       "Unknown component type: InvalidStat");
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

/// Test that adding an offering with an invalid JSON format throws an exception.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsInvalidJSON) {
  std::istringstream json_stream(R"([
  {
    "type": "stat",
    "name": "Invalid Offering",
    "description": "This offering has an invalid JSON format",
    "icon_type": "Invalid",
    "base_cost": 100,
    "cost_multiplier": 1.5,
  }])");
  ASSERT_THROW_MESSAGE(get_shop_system()->add_offerings(json_stream, 0), nlohmann::json::exception,
                       "[json.exception.parse_error.101] parse error at line 9, column 3: syntax error while parsing "
                       "object key - unexpected '}'; expected string literal");
}

/// Test that adding an offering with an empty stream throws an exception.
TEST_F(ShopSystemFixture, TestShopSystemAddOfferingsEmptyStream) {
  std::istringstream json_stream("");
  ASSERT_THROW_MESSAGE(get_shop_system()->add_offerings(json_stream, 0), nlohmann::json::exception,
                       "[json.exception.parse_error.101] parse error at line 1, column 1: attempting to parse an empty "
                       "input; check that your input string or stream contains the expected JSON");
}

/// Test that getting the cost returns the correct value if no offerings are available.
TEST_F(ShopSystemFixture, TestShopSystemGetOfferingCostNoOfferings) {
  ASSERT_EQ(get_shop_system()->get_offering_cost(0, 0), -1);
}

/// Test that getting the cost of a stat upgrade offering works correctly.
TEST_F(ShopSystemFixture, TestShopSystemGetStatUpgradeOfferingCost) {
  add_stat_upgrade();
  ASSERT_EQ(get_shop_system()->get_offering_cost(0, 0), 100);
}

/// Test that getting the cost of a levelled stat upgrade offering works correctly.
TEST_F(ShopSystemFixture, TestShopSystemGetLevelledStatUpgradeOfferingCost) {
  add_stat_upgrade();
  registry.get_component<Health>(0)->increment_current_level();
  ASSERT_EQ(get_shop_system()->get_offering_cost(0, 0), 101);
}

/// Test that getting the cost of a component unlock offering works correctly.
TEST_F(ShopSystemFixture, TestShopSystemGetComponentUnlockOfferingCost) {
  add_component_unlock();
  ASSERT_EQ(get_shop_system()->get_offering_cost(0, 0), 100);
}

/// Test that getting the cost of an item offering works correctly.
TEST_F(ShopSystemFixture, TestShopSystemGetItemOfferingCost) {
  add_item();
  ASSERT_EQ(get_shop_system()->get_offering_cost(0, 0), 100);
}

/// Test that purchasing a stat upgrade offering works correctly if the player has enough money.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseStatUpgradeSufficientMoney) {
  add_stat_upgrade();
  registry.get_component<Money>(0)->money = 500;
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 400);
  ASSERT_EQ(registry.get_component<Health>(0)->get_max_value(), 400);
  ASSERT_EQ(registry.get_component<Health>(0)->get_current_level(), 1);
  ASSERT_EQ(registry.get_component<Money>(0)->money, 400);
}

/// Test that purchasing a stat upgrade offering works correctly if the stat is not at the maximum value.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseStatUpgradeNotMaxValue) {
  add_stat_upgrade();
  registry.get_component<Health>(0)->set_value(100);
  registry.get_component<Money>(0)->money = 500;
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<Health>(0)->get_max_value(), 400);
  ASSERT_EQ(registry.get_component<Health>(0)->get_current_level(), 1);
  ASSERT_EQ(registry.get_component<Money>(0)->money, 400);
}

/// Test that purchasing a stat upgrade offering works correctly multiple times.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseStatUpgradeMultipleTimes) {
  add_stat_upgrade();
  registry.get_component<Money>(0)->money = 500;
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 601);
  ASSERT_EQ(registry.get_component<Health>(0)->get_max_value(), 601);
  ASSERT_EQ(registry.get_component<Health>(0)->get_current_level(), 2);
  ASSERT_EQ(registry.get_component<Money>(0)->money, 299);
}

/// Test that purchasing a stat upgrade offering does not work if the player does not have enough money.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseStatUpgradeInsufficientMoney) {
  add_stat_upgrade();
  ASSERT_FALSE(get_shop_system()->purchase(0, 0));
}

/// Test that purchasing a stat upgrade offering does not work if the stat is at the maximum level.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseStatUpgradeMaxLevel) {
  add_stat_upgrade();
  registry.get_component<Money>(0)->money = 500;
  for (int i{0}; i < 5; i++) {
    registry.get_component<Health>(0)->increment_current_level();
  }
  ASSERT_FALSE(get_shop_system()->purchase(0, 0));
}

/// Test that an exception is thrown when purchasing a stat upgrade offering if the game object does not have the
/// required component.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseStatUpgradeNoComponent) {
  registry.create_game_object(GameObjectType::Player, cpvzero, {std::make_shared<Money>()});
  add_stat_upgrade();
  ASSERT_THROW_MESSAGE(get_shop_system()->purchase(1, 0), RegistryError,
                       "The component `Health` for the game object ID `1` is not registered with the registry.");
}

/// Test that purchasing a component unlock offering works correctly if the player has enough money.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseComponentUnlockSufficientMoney) {
  add_component_unlock();
  registry.get_component<Money>(0)->money = 500;
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_EQ(registry.get_component<Money>(0)->money, 400);
}

/// Test that purchasing a component unlock offering does not work if the player does not have enough money.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseComponentUnlockInsufficientMoney) {
  add_component_unlock();
  ASSERT_FALSE(get_shop_system()->purchase(0, 0));
}

// /// Test that purchasing a component unlock offering does not work if the player already has the component.
// TEST_F(ShopSystemFixture, TestShopSystemPurchaseComponentUnlockAlreadyUnlocked) {
//   add_component_unlock();
//   registry.get_component<Money>(0)->money = 500;
//   ASSERT_FALSE(get_shop_system()->purchase(0, 0));
// }

/// Test that purchasing an item offering works correctly if the player has enough money.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseItemOfferingSufficientMoney) {
  add_item();
  registry.get_component<Money>(0)->money = 500;
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_EQ(registry.get_component<Money>(0)->money, 400);
}

/// Test that purchasing an item offering does not work if the player does not have enough money.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseItemOfferingInsufficientMoney) {
  add_item();
  ASSERT_FALSE(get_shop_system()->purchase(0, 0));
}

/// Test that purchasing multiple item offerings works correctly if the player has enough money.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseMultipleItemOfferingsSufficientMoney) {
  add_item();
  registry.get_component<Money>(0)->money = 500;
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_EQ(registry.get_component<Money>(0)->money, 300);
}

/// Test that the shop item purchased callback is called correctly.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseItemShopItemPurchasedCallback) {
  add_stat_upgrade();
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
  registry.create_game_object(GameObjectType::Player, cpvzero, {std::make_shared<Health>(0, -1)});
  add_stat_upgrade();
  ASSERT_THROW_MESSAGE(get_shop_system()->purchase(1, 0), RegistryError,
                       "The component `Money` for the game object ID `1` is not registered with the registry.");
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseInvalidGameObjectId) {
  add_stat_upgrade();
  ASSERT_THROW_MESSAGE(get_shop_system()->purchase(-1, 0), RegistryError,
                       "The component `Money` for the game object ID `-1` is not registered with the registry.");
}
