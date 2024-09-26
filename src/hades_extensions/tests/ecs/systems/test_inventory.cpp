// Local headers
#include "ecs/registry.hpp"
#include "ecs/stats.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/physics.hpp"
#include "macros.hpp"

/// Implements the fixture for the InventorySystem fixture.
class InventorySystemFixture : public testing::Test {
 protected:
  /// The registry that manages the game objects, components, and systems.
  Registry registry;

  /// Set up the fixture for the tests.
  void SetUp() override {
    registry.create_game_object(GameObjectType::Player, cpvzero,
                                {std::make_shared<Health>(200, -1), std::make_shared<Inventory>(),
                                 std::make_shared<InventorySize>(8, -1), std::make_shared<StatusEffect>()});
    registry.add_system<InventorySystem>();
    registry.add_system<EffectSystem>();
  }

  /// Create an item with the specified type.
  ///
  /// @param type - The type of the game object.
  /// @param kinematic - Whether the item is kinematic or not.
  /// @return The game object ID of the item.
  [[nodiscard]] auto create_item(const GameObjectType type, const bool kinematic = false) -> GameObjectID {
    std::vector<std::shared_ptr<ComponentBase>> components;
    if (kinematic) {
      components.push_back(std::make_shared<KinematicComponent>());
    }
    return registry.create_game_object(type, cpvzero, std::move(components));
  }

  /// Get the inventory system from the registry.
  ///
  /// @return The inventory system.
  [[nodiscard]] auto get_inventory_system() const -> std::shared_ptr<InventorySystem> {
    return registry.get_system<InventorySystem>();
  }

  /// Create a status effect item.
  ///
  /// @return The game object ID of the status effect item.
  [[nodiscard]] auto create_status_effect_item() -> GameObjectID {
    return registry.create_game_object(GameObjectType::HealthPotion, cpvzero,
                                       {std::make_shared<EffectApplier>(
                                            std::unordered_map<std::type_index, ActionFunction>{
                                                {typeid(Health), [](const int level) { return (level * 3) + 5; }}},
                                            std::unordered_map<std::type_index, StatusEffectData>{}),
                                        std::make_shared<EffectLevel>(1, -1)});
  }
};

/// Test that a valid item is added to the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryValid) {
  // Add the callbacks to the registry
  bool inventory_update{false};
  auto inventory_update_callback{[&](const GameObjectID /*game_object_id*/) { inventory_update = true; }};
  registry.add_callback(EventType::InventoryUpdate, inventory_update_callback);
  bool sprite_removal{false};
  auto sprite_removal_callback{[&](const GameObjectID /*game_object_id*/) { sprite_removal = true; }};
  registry.add_callback(EventType::SpriteRemoval, sprite_removal_callback);

  // Add the item to the inventory and check the results
  const auto game_object_id{create_item(GameObjectType::HealthPotion)};
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, game_object_id));
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, std::vector{game_object_id});
  ASSERT_TRUE(inventory_update);
  ASSERT_TRUE(sprite_removal);
}

/// Test that an item with a kinematic component is added to the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryKinematic) {
  const auto game_object_id{create_item(GameObjectType::HealthPotion, true)};
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, game_object_id));
  ASSERT_TRUE(registry.get_component<KinematicComponent>(game_object_id)->collected);
}

/// Test that an item is not added to the inventory if it is not a collectible item.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryInvalidType) {
  const auto game_object_id{create_item(GameObjectType::Player)};
  ASSERT_FALSE(get_inventory_system()->add_item_to_inventory(0, game_object_id));
}

/// Test that an item is not added to the inventory if it is not a valid game object.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryInvalidItem) {
  ASSERT_FALSE(get_inventory_system()->add_item_to_inventory(0, -1));
}

/// Test that a valid item is not added to a zero size inventory.
TEST_F(InventorySystemFixture, TestInventorySystemAddItemToInventoryZeroSize) {
  const auto game_object_id{create_item(GameObjectType::HealthPotion)};
  registry.get_component<InventorySize>(0)->set_value(0);
  ASSERT_THROW_MESSAGE(get_inventory_system()->add_item_to_inventory(0, game_object_id), std::runtime_error,
                       "The inventory is full.")
}

/// Test that a valid item is removed from the inventory correctly.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryValid) {
  // Add the callbacks to the registry
  auto inventory_update{false};
  auto inventory_update_callback{[&](const GameObjectID /*game_object_id*/) { inventory_update = true; }};
  registry.add_callback(EventType::InventoryUpdate, inventory_update_callback);
  auto sprite_removal{false};
  auto sprite_removal_callback{[&](const GameObjectID /*game_object_id*/) { sprite_removal = true; }};
  registry.add_callback(EventType::SpriteRemoval, sprite_removal_callback);

  // Add two items and remove one of them from the inventory and check the
  // results
  const auto item_id_one{create_item(GameObjectType::HealthPotion)};
  const auto item_id_two{create_item(GameObjectType::HealthPotion)};
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, item_id_one));
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, item_id_two));
  ASSERT_TRUE(get_inventory_system()->remove_item_from_inventory(0, item_id_one));
  const std::vector result{item_id_two};
  ASSERT_EQ(registry.get_component<Inventory>(0)->items, result);
  ASSERT_TRUE(inventory_update);
  ASSERT_TRUE(sprite_removal);
}

/// Test that an item is not removed from the inventory if it is not added.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryNotAdded) {
  const auto item_id_one{create_item(GameObjectType::HealthPotion)};
  const auto item_id_two{create_item(GameObjectType::HealthPotion)};
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, item_id_one));
  ASSERT_FALSE(get_inventory_system()->remove_item_from_inventory(0, item_id_two));
}

/// Test that an item is not removed from the inventory if it is not a valid game object.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryInvalidItem) {
  ASSERT_FALSE(get_inventory_system()->remove_item_from_inventory(0, -1));
}

/// Test that an exception is thrown if the game object does not have an inventory component.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryNonexistentComponent) {
  const auto game_object_id{create_item(GameObjectType::Player)};
  ASSERT_THROW_MESSAGE((get_inventory_system()->remove_item_from_inventory(game_object_id, 0)), RegistryError,
                       "The component `Inventory` for the game object ID `1` is not registered with the registry.")
}

/// Test that an exception is thrown if an invalid game object ID is provided.
TEST_F(InventorySystemFixture, TestInventorySystemRemoveItemFromInventoryInvalidGameObjectID){
    ASSERT_THROW_MESSAGE((get_inventory_system()->remove_item_from_inventory(-1, 0)), RegistryError,
                         "The component `Inventory` for the game object ID `-1` is not registered with the registry.")}

/// Test that a status effect item is used correctly if the target can be affected.
TEST_F(InventorySystemFixture, TestInventorySystemUseItemStatusEffectValid) {
  const auto item_id{create_status_effect_item()};
  registry.get_component<Health>(0)->set_value(50);
  ASSERT_TRUE(get_inventory_system()->add_item_to_inventory(0, item_id));
  ASSERT_TRUE(get_inventory_system()->use_item(0, item_id));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 58);
  ASSERT_EQ(registry.get_component<Inventory>(0)->items.size(), 0);
}

/// Test that a status effect item is not used if the target cannot be affected.
TEST_F(InventorySystemFixture, TestInventorySystemUseItemStatusEffectInvalid) {
  const auto item_id{create_status_effect_item()};
  ASSERT_FALSE(get_inventory_system()->use_item(0, item_id));
  ASSERT_EQ(registry.get_component<Health>(0)->get_value(), 200);
}

/// Test that an item is not used if it doesn't match any of the strategies.
TEST_F(InventorySystemFixture, TestInventorySystemUseItemNoEffect) {
  const auto item_id{registry.create_game_object(GameObjectType::HealthPotion, cpvzero, {})};
  ASSERT_FALSE(get_inventory_system()->use_item(0, item_id));
}

/// Test that nothing happens if the item game object does not exist.
TEST_F(InventorySystemFixture, TestInventorySystemUseItemStatusEffectInvalidItemID) {
  ASSERT_FALSE(get_inventory_system()->use_item(0, -1));
}

/// Test that an exception is thrown if the game object does not have the required components.
TEST_F(InventorySystemFixture, TestInventorySystemUseItemStatusEffectInvalidGameObjectID) {
  const auto item_id{create_status_effect_item()};
  ASSERT_THROW_MESSAGE(get_inventory_system()->use_item(-1, item_id), RegistryError,
                       "The component `StatusEffect` for the game object ID `-1` is not registered with the registry.")
}
