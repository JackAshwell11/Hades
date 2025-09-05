// Local headers
#include "events.hpp"
#include "macros.hpp"

/// Implements the fixture for the events.hpp tests.
class EventsFixture : public testing::Test {
 protected:
  /// Tear down the fixture after the tests.
  void TearDown() override { clear_listeners(); }
};

/// Test that a callback is not called if none are added.
TEST_F(EventsFixture, TestNotifyCallbacksNoCallbacksAdded) {
  constexpr bool called{false};
  notify<EventType::GameObjectDeath>(1);
  ASSERT_FALSE(called);
}

/// Test that a callback is not called if it is not listening for the event.
TEST_F(EventsFixture, TestNotifyCallbacksNoCallbacksListening) {
  auto called{-1};
  add_callback<EventType::GameObjectCreation>(
      [&called](const GameObjectID event, const GameObjectType) { called = event; });
  notify<EventType::GameObjectDeath>(1);
  ASSERT_EQ(called, -1);
}

/// Test that a callback is called if it is listening for the event.
TEST_F(EventsFixture, TestRegistryNotifyCallbacksListeningCallback) {
  auto called{-1};
  add_callback<EventType::GameObjectDeath>([&called](const GameObjectID event) { called = event; });
  notify<EventType::GameObjectDeath>(1);
  ASSERT_EQ(called, 1);
}

/// Test that multiple callbacks are called for the same event.
TEST_F(EventsFixture, TestNotifyCallbacksMultipleCallbacks) {
  auto called_one{-1};
  auto called_two{-1};
  add_callback<EventType::GameObjectDeath>([&called_one](const GameObjectID event) { called_one = event; });
  add_callback<EventType::GameObjectDeath>([&called_two](const GameObjectID event) { called_two = event * 2; });
  notify<EventType::GameObjectDeath>(1);
  ASSERT_EQ(called_one, 1);
  ASSERT_EQ(called_two, 2);
}

/// Test that a callback is called with no arguments.
TEST_F(EventsFixture, TestNotifyCallbacksNoArgs) {
  bool called{false};
  add_callback<EventType::InventoryOpen>([&called] { called = true; });
  notify<EventType::InventoryOpen>();
  ASSERT_TRUE(called);
}
