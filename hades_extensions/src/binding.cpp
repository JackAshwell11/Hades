// Std includes
#include <optional>

// External includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Custom includes
#include "generation/map.hpp"

// ----- PYTHON MODULE CREATION ------------------------------
PYBIND11_MODULE(hades_extensions, m) {
  // Add the module docstring
  m.doc() = "Manages the various C++ extension modules for the game.";

  // Create the generation module
  pybind11::module generation = m.def_submodule("generation", "Generates the dungeon and places game objects in it.");
  generation.def(
      "create_map",
      &create_map,
      pybind11::arg("level"),
      pybind11::arg("seed") = pybind11::none(),
      (
          "Generate the game map for a given game level.\n\n"
          "Args:\n"
          "    level: The game level to generate a map for.\n"
          "    seed: The seed to initialise the random generator.\n\n"
          "Returns:\n"
          "    A tuple containing the generated map and the level constants.\n\n"
      )
  );
  pybind11::enum_<TileType>(generation, "TileType")
      .value("Empty", TileType::Empty)
      .value("Floor", TileType::Floor)
      .value("Wall", TileType::Wall)
      .value("Obstacle", TileType::Obstacle)
      .value("Player", TileType::Player)
      .value("Potion", TileType::Potion);
}
