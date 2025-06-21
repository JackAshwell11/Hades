// Ensure this file is only included once
#pragma once

// Std headers
#include <random>

// Local headers
#include "generation/dijkstra.hpp"

/// Manages the generation of the map.
class MapGenerator {
 public:
  /// Initialise the object.
  MapGenerator();

  /// Initialise the object.
  ///
  /// @param level - The game level to generate a map for.
  /// @param random_generator - The random generator to use for the map generation.
  explicit MapGenerator(int level, const std::mt19937 &random_generator);

  /// Generate the rooms in the dungeon.
  ///
  /// @return A reference to the MapGenerator object.
  auto generate_rooms() -> MapGenerator &;

  /// Create connections between the rooms in the dungeon.
  ///
  /// @return A reference to the MapGenerator object.
  auto create_connections() -> MapGenerator &;

  /// Generate the hallways in the dungeon.
  ///
  /// @return A reference to the MapGenerator object.
  auto generate_hallways() -> MapGenerator &;

  /// Perform the cellular automata simulation in the dungeon.
  ///
  /// @param generations - The number of generations to simulate.
  /// @return A reference to the MapGenerator object.
  auto cellular_automata(int generations) -> MapGenerator &;

  /// Generate the walls in the dungeon.
  ///
  /// @return A reference to the MapGenerator object.
  auto generate_walls() -> MapGenerator &;

  /// Place the obstacles in the dungeon.
  ///
  /// @return A reference to the MapGenerator object.
  auto place_obstacles() -> MapGenerator &;

  /// Place the player in the dungeon.
  ///
  /// @return A reference to the MapGenerator object.
  auto place_player() -> MapGenerator &;

  /// Place the items in the dungeon.
  ///
  /// @return A reference to the MapGenerator object.
  auto place_items() -> MapGenerator &;

  /// Place the goal in the dungeon.
  ///
  /// @return A reference to the MapGenerator object.
  auto place_goal() -> MapGenerator &;

  /// Place the lobby in the dungeon.
  ///
  /// @return A reference to the MapGenerator object.
  auto place_lobby() -> MapGenerator &;

  /// Get the grid.
  ///
  /// @return The grid.
  [[nodiscard]] auto get_grid() -> Grid & { return grid_; }

  /// Get the rooms.
  ///
  /// @return The rooms.
  [[nodiscard]] auto get_rooms() -> std::vector<Position> & { return rooms_; }

  /// Get the connections.
  ///
  /// @return The connections.
  [[nodiscard]] auto get_connections() -> std::vector<Connection> & { return connections_; }

  /// Get the enemy limit.
  ///
  /// @param level - The level of the dungeon.
  /// @return The enemy limit.
  [[nodiscard]] static auto get_enemy_limit(int level) -> int;

 private:
  /// The level of the dungeon.
  int level_;

  /// The 2D grid which represents the dungeon.
  Grid grid_;

  /// The random generator used to generate the map.
  std::mt19937 random_generator_;

  /// The rooms that have been generated.
  std::vector<Position> rooms_;

  /// The connections between the rooms.
  std::vector<Connection> connections_;
};
