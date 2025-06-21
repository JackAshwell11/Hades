// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/shop.hpp"
#include "macros.hpp"

/// Represents a test stat useful for testing the shop system.
struct TestShopStat final : Stat {
  /// Initialise the object.
  TestShopStat() : Stat(200, 5) {}
};

/// Implements the fixture for the ShopSystem tests.
class ShopSystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.add_system<ShopSystem>();
    registry.create_game_object(GameObjectType::Player, cpvzero,
                                {std::make_shared<Money>(), std::make_shared<TestShopStat>()});
  }

  /// Add a stat upgrade offering to the shop system.
  void add_stat_upgrade() const {
    get_shop_system()->add_stat_upgrade("Test Stat Upgrade", "A test stat upgrade offering", typeid(TestShopStat), 100,
                                        1.5, 200, 1.2);
  }

  /// Add a component unlock offering to the shop system.
  void add_component_unlock() const {
    get_shop_system()->add_component_unlock("Test Component Unlock", "A test component unlock offering", 100, 1.5);
  }

  /// Add an item offering to the shop system.
  void add_item() const { get_shop_system()->add_item("Test Item Offering", "A test item offering", 100, 1.5); }

  /// Get the shop system from the registry.
  ///
  /// @return The shop system.
  [[nodiscard]] auto get_shop_system() const -> std::shared_ptr<ShopSystem> {
    return registry.get_system<ShopSystem>();
  }
};

/// Test that adding a stat upgrade offering to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddStatUpgradeSingle) {
  add_stat_upgrade();
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Stat Upgrade");
  ASSERT_EQ(get_shop_system()->get_offering(1), nullptr);
}

/// Test that adding multiple stat upgrade offerings to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddStatUpgradeMultiple) {
  add_stat_upgrade();
  get_shop_system()->add_stat_upgrade("Test Stat Upgrade 2", "A test stat upgrade offering 2", typeid(TestShopStat),
                                      150, 1.8, 250, 1.5);
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Stat Upgrade");
  ASSERT_EQ(get_shop_system()->get_offering(1)->name, "Test Stat Upgrade 2");
  ASSERT_EQ(get_shop_system()->get_offering(2), nullptr);
}

/// Test that adding a component unlock offering to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddComponentUnlockSingle) {
  add_component_unlock();
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Component Unlock");
  ASSERT_EQ(get_shop_system()->get_offering(1), nullptr);
}

/// Test that adding multiple component unlock offerings to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddComponentUnlockMultiple) {
  add_component_unlock();
  get_shop_system()->add_component_unlock("Test Component Unlock 2", "A test component unlock offering 2", 150, 1.8);
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Component Unlock");
  ASSERT_EQ(get_shop_system()->get_offering(1)->name, "Test Component Unlock 2");
  ASSERT_EQ(get_shop_system()->get_offering(2), nullptr);
}

/// Test that adding an item offering to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddItemOfferingSingle) {
  add_item();
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Item Offering");
  ASSERT_EQ(get_shop_system()->get_offering(1), nullptr);
}

/// Test that adding multiple item offerings to the shop works correctly.
TEST_F(ShopSystemFixture, TestShopSystemAddItemOfferingMultiple) {
  add_item();
  get_shop_system()->add_item("Test Item Offering 2", "A test item offering 2", 150, 1.8);
  ASSERT_EQ(get_shop_system()->get_offering(0)->name, "Test Item Offering");
  ASSERT_EQ(get_shop_system()->get_offering(1)->name, "Test Item Offering 2");
  ASSERT_EQ(get_shop_system()->get_offering(2), nullptr);
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
  registry.get_component<TestShopStat>(0)->increment_current_level();
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
  ASSERT_EQ(registry.get_component<TestShopStat>(0)->get_value(), 400);
  ASSERT_EQ(registry.get_component<TestShopStat>(0)->get_max_value(), 400);
  ASSERT_EQ(registry.get_component<TestShopStat>(0)->get_current_level(), 1);
  ASSERT_EQ(registry.get_component<Money>(0)->money, 400);
}

/// Test that purchasing a stat upgrade offering works correctly if the stat is not at the maximum value.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseStatUpgradeNotMaxValue) {
  add_stat_upgrade();
  registry.get_component<TestShopStat>(0)->set_value(100);
  registry.get_component<Money>(0)->money = 500;
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_EQ(registry.get_component<TestShopStat>(0)->get_value(), 300);
  ASSERT_EQ(registry.get_component<TestShopStat>(0)->get_max_value(), 400);
  ASSERT_EQ(registry.get_component<TestShopStat>(0)->get_current_level(), 1);
  ASSERT_EQ(registry.get_component<Money>(0)->money, 400);
}

/// Test that purchasing a stat upgrade offering works correctly multiple times.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseStatUpgradeMultipleTimes) {
  add_stat_upgrade();
  registry.get_component<Money>(0)->money = 500;
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_TRUE(get_shop_system()->purchase(0, 0));
  ASSERT_EQ(registry.get_component<TestShopStat>(0)->get_value(), 601);
  ASSERT_EQ(registry.get_component<TestShopStat>(0)->get_max_value(), 601);
  ASSERT_EQ(registry.get_component<TestShopStat>(0)->get_current_level(), 2);
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
    registry.get_component<TestShopStat>(0)->increment_current_level();
  }
  ASSERT_FALSE(get_shop_system()->purchase(0, 0));
}

/// Test that an exception is thrown when purchasing a stat upgrade offering if the game object does not have the
/// required component.
TEST_F(ShopSystemFixture, TestShopSystemPurchaseStatUpgradeNoComponent) {
  registry.create_game_object(GameObjectType::Player, cpvzero, {std::make_shared<Money>()});
  add_stat_upgrade();
  ASSERT_THROW_MESSAGE(get_shop_system()->purchase(1, 0), RegistryError,
                       "The component `TestShopStat` for the game object ID `1` is not registered with the registry.");
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
  registry.add_callback<EventType::ShopItemPurchased>(
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
  registry.create_game_object(GameObjectType::Player, cpvzero, {std::make_shared<TestShopStat>()});
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
