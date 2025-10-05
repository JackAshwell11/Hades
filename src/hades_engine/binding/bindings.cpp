// Std headers
#include <fstream>

// Local headers
#include "common.hpp"
#include "ecs/systems/shop.hpp"
#include "events.hpp"
#include "factories.hpp"
#include "game_engine.hpp"

PYBIND11_MODULE(hades_engine, module) {  // NOLINT
  module.doc() = "Manages the various C++ extension modules for the game.";
  module.def("load_hitbox", &load_hitbox, pybind11::arg("game_object_type"), pybind11::arg("hitbox"),
             "Load a hitbox for a game object type.\n\n"
             "Args:\n"
             "    game_object_type: The type of game object to load the hitbox for.\n"
             "    hitbox: The hitbox to load for the game object type.\n\n"
             "Returns:\n"
             "    Whether the hitbox was loaded or not.");
  module.def(
      "add_callback",
      [](const EventType event_type, const pybind11::function& callback) {
        switch (event_type) {
          case EventType::GameObjectCreation:
            add_callback<EventType::GameObjectCreation>(callback);
            break;
          case EventType::GameObjectDeath:
            add_callback<EventType::GameObjectDeath>(callback);
            break;
          case EventType::PositionChanged:
            add_callback<EventType::PositionChanged>(callback);
            break;
          case EventType::InventoryUpdate:
            add_callback<EventType::InventoryUpdate>(callback);
            break;
          case EventType::SpriteRemoval:
            add_callback<EventType::SpriteRemoval>(callback);
            break;
          case EventType::StatusEffectUpdate:
            add_callback<EventType::StatusEffectUpdate>(callback);
            break;
          case EventType::MoneyUpdate:
            add_callback<EventType::MoneyUpdate>(callback);
            break;
          case EventType::AttackCooldownUpdate:
            add_callback<EventType::AttackCooldownUpdate>(callback);
            break;
          case EventType::RangedAttackSwitch:
            add_callback<EventType::RangedAttackSwitch>(callback);
            break;
          case EventType::ShopItemLoaded:
            add_callback<EventType::ShopItemLoaded>(callback);
            break;
          case EventType::ShopItemPurchased:
            add_callback<EventType::ShopItemPurchased>(callback);
            break;
          case EventType::ShopOpen:
            add_callback<EventType::ShopOpen>(callback);
            break;
          case EventType::InventoryOpen:
            add_callback<EventType::InventoryOpen>(callback);
            break;
          case EventType::GameOpen:
            add_callback<EventType::GameOpen>(callback);
            break;
          case EventType::HealthChanged:
            add_callback<EventType::HealthChanged>(callback);
            break;
          case EventType::ArmourChanged:
            add_callback<EventType::ArmourChanged>(callback);
            break;
          default:
            throw std::runtime_error("Unsupported event type.");
        }
      },
      pybind11::arg("event_type"), pybind11::arg("callback"),
      "Add a callback to the registry to listen for events.\n\n"
      "Args:\n"
      "    event_type: The type of event to listen for.\n"
      "    callback: The callback to add.");
  module.add_object("_listener_cleanup", pybind11::capsule([] { clear_listeners(); }));

  pybind11::enum_<EventType>(module, "EventType", "Stores the different types of events that can occur.")
      .value("GameObjectCreation", EventType::GameObjectCreation)
      .value("GameObjectDeath", EventType::GameObjectDeath)
      .value("PositionChanged", EventType::PositionChanged)
      .value("InventoryUpdate", EventType::InventoryUpdate)
      .value("SpriteRemoval", EventType::SpriteRemoval)
      .value("StatusEffectUpdate", EventType::StatusEffectUpdate)
      .value("MoneyUpdate", EventType::MoneyUpdate)
      .value("AttackCooldownUpdate", EventType::AttackCooldownUpdate)
      .value("RangedAttackSwitch", EventType::RangedAttackSwitch)
      .value("ShopItemLoaded", EventType::ShopItemLoaded)
      .value("ShopItemPurchased", EventType::ShopItemPurchased)
      .value("ShopOpen", EventType::ShopOpen)
      .value("InventoryOpen", EventType::InventoryOpen)
      .value("GameOpen", EventType::GameOpen)
      .value("HealthChanged", EventType::HealthChanged)
      .value("ArmourChanged", EventType::ArmourChanged);

  pybind11::enum_<DifficultyLevel>(module, "DifficultyLevel", "Stores the different types of difficulty levels.")
      .value("Easy", DifficultyLevel::Easy)
      .value("Normal", DifficultyLevel::Normal)
      .value("Hard", DifficultyLevel::Hard);

  pybind11::class_<GameState, std::shared_ptr<GameState>>(module, "GameState", "Stores the state of the game.")
      .def_property_readonly("player_id", &GameState::get_player_id)
      .def_property("difficulty_level", &GameState::get_difficulty_level, &GameState::set_difficulty_level)
      .def("set_seed", &GameState::set_seed, pybind11::arg("seed"),
           "Set the seed for the random generator.\n\n"
           "Args:\n"
           "    seed: The seed to set for the random generator.")
      .def("set_window_size", &GameState::set_window_size, pybind11::arg("width"), pybind11::arg("height"),
           "Set the window size.\n\n"
           "Args:\n"
           "    width: The width of the window.\n"
           "    height: The height of the window.");

  pybind11::class_<InputHandler, std::shared_ptr<InputHandler>>(
      module, "InputHandler", "Handles input events such as key presses, releases, and mouse clicks.")
      .def("on_key_press", &InputHandler::on_key_press, pybind11::arg("symbol"), pybind11::arg("modifiers"),
           "Process key press functionality.\n\n"
           "Args:\n"
           "    symbol: The key that was hit.\n"
           "    modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.")
      .def("on_key_release", &InputHandler::on_key_release, pybind11::arg("symbol"), pybind11::arg("modifiers"),
           "Process key release functionality.\n\n"
           "Args:\n"
           "    symbol: The key that was released.\n"
           "    modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.")
      .def("on_mouse_press", &InputHandler::on_mouse_press, pybind11::arg("x"), pybind11::arg("y"),
           pybind11::arg("button"), pybind11::arg("modifiers"),
           "Process mouse press functionality.\n\n"
           "Args:\n"
           "    x: The x position of the mouse.\n"
           "    y: The y position of the mouse.\n"
           "    button: The button that was pressed.\n"
           "    modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.")
      .def("on_mouse_motion", &InputHandler::on_mouse_motion, pybind11::arg("x"), pybind11::arg("y"),
           pybind11::arg("delta_x"), pybind11::arg("delta_y"),
           "Process mouse motion functionality.\n\n"
           "Args:\n"
           "    x: The x position of the mouse.\n"
           "    y: The y position of the mouse.\n"
           "    delta_x: The change in x position of the mouse.\n"
           "    delta_y: The change in y position of the mouse.");

  pybind11::class_<GameEngine, std::shared_ptr<GameEngine>>(module, "GameEngine",
                                                            "Manages the game objects and systems.")
      .def(pybind11::init<>(), "Initialise the object.")
      .def_property_readonly("registry", &GameEngine::get_registry)
      .def_property_readonly("game_state", &GameEngine::get_game_state)
      .def_property_readonly("input_handler", &GameEngine::get_input_handler)
      .def(
          "setup",
          [](GameEngine& engine, const std::string& shop_file_path) {
            std::ifstream shop_stream{shop_file_path};
            engine.get_registry()->get_system<ShopSystem>()->add_offerings(shop_stream,
                                                                           engine.get_game_state()->get_player_id());
          },
          pybind11::arg("shop_file_path") = "",
          "Set up the game engine.\n\n"
          "Args:\n"
          "    shop_file_path: The path to the shop file.")
      .def("on_update", &GameEngine::on_update, pybind11::arg("delta_time"),
           "Process update logic for the game engine.\n\n"
           "Args:\n"
           "    delta_time: The time interval since the last time the function was called.")
      .def("on_fixed_update", &GameEngine::on_fixed_update, pybind11::arg("delta_time"),
           "Process fixed update logic for the game engine.\n\n"
           "Args:\n"
           "    delta_time: The time interval since the last time the function was called.")
      .def("enter_dungeon", &GameEngine::enter_dungeon, "Enter the dungeon from the lobby.");

  // Create the submodules for the ECS
  auto ecs{module.def_submodule(
      "ecs", "Contains the registry and the various components and systems that can be used with it.")};
  const auto systems{ecs.def_submodule("systems", "Contains the systems which manage the game objects.")};
  bind_ecs(ecs);
  bind_systems(systems);
}
