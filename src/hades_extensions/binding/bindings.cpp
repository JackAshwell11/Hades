// Local headers
#include "common.hpp"
#include "factories.hpp"
#include "game_engine.hpp"

PYBIND11_MODULE(hades_extensions, module) {  // NOLINT
  module.doc() = "Manages the various C++ extension modules for the game.";
  module.def("load_hitbox", &load_hitbox, pybind11::arg("game_object_type"), pybind11::arg("hitbox"),
             "Load a hitbox for a game object type.\n\n"
             "Args:\n"
             "    game_object_type: The type of game object to load the hitbox for.\n"
             "    hitbox: The hitbox to load for the game object type.\n\n"
             "Returns:\n"
             "    Whether the hitbox was loaded or not.");
  pybind11::class_<GameEngine>(module, "GameEngine", "Manages the game objects and systems.")
      .def(pybind11::init<int, std::optional<unsigned int>>(), pybind11::arg("level"),
           pybind11::arg("seed") = pybind11::none(),
           "Initialise the object.\n\n"
           "Args:\n"
           "    level: The level to generate the game engine for.\n"
           "    seed: The seed to use for the random number generator.")
      .def_property_readonly("player_id", &GameEngine::get_player_id)
      .def_property_readonly("level_constants", &GameEngine::get_level_constants)
      .def_property_readonly("nearest_item", &GameEngine::get_nearest_item)
      .def("get_registry", &GameEngine::get_registry,
           "Get the registry.\n\n"
           "Returns:\n"
           "    The registry.")
      .def("create_game_objects", &GameEngine::create_game_objects,
           "Create the game objects.\n\n"
           "If this is called twice, the game objects will be duplicated.\n\n"
           "Returns:\n"
           "    The game objects.")
      .def("generate_enemy", &GameEngine::generate_enemy, pybind11::arg("delta_time"),
           "Generate an enemy.\n\n"
           "Args:\n"
           "    delta_time: The time interval since the last time the function was called.")
      .def("on_update", &GameEngine::on_update, pybind11::arg("delta_time"),
           "Process update logic for the game engine.\n\n"
           "Args:\n"
           "    delta_time: The time interval since the last time the function was called.")
      .def("on_fixed_update", &GameEngine::on_fixed_update, pybind11::arg("delta_time"),
           "Process fixed update logic for the game engine.\n\n"
           "Args:\n"
           "    delta_time: The time interval since the last time the function was called.")
      .def("on_key_press", &GameEngine::on_key_press, pybind11::arg("symbol"), pybind11::arg("modifiers"),
           "Process key press functionality.\n\n"
           "Args:\n"
           "    symbol: The key that was hit.\n"
           "    modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.")
      .def("on_key_release", &GameEngine::on_key_release, pybind11::arg("symbol"), pybind11::arg("modifiers"),
           "Process key release functionality.\n\n"
           "Args:\n"
           "    symbol: The key that was released.\n"
           "    modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.")
      .def("on_mouse_press", &GameEngine::on_mouse_press, pybind11::arg("x"), pybind11::arg("y"),
           pybind11::arg("button"), pybind11::arg("modifiers"),
           "Process mouse press functionality.\n\n"
           "Args:\n"
           "    x: The x position of the mouse.\n"
           "    y: The y position of the mouse.\n"
           "    button: The button that was pressed.\n"
           "    modifiers: Bitwise AND of all modifiers (shift, ctrl, num lock) pressed during this event.");

  // Create the submodules for the ECS
  auto ecs{module.def_submodule(
      "ecs", "Contains the registry and the various components and systems that can be used with it.")};
  const auto systems{ecs.def_submodule("systems", "Contains the systems which manage the game objects.")};
  const auto components{ecs.def_submodule("components", "Contains the components which can be added to game objects.")};
  bind_ecs(ecs);
  bind_components(components);
  bind_systems(systems);
}
