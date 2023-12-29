// Ensure this file is only included once
#pragma once

// Std headers
#include <unordered_set>

// Local headers
#include "primitives.hpp"

// ----- FUNCTIONS ------------------------------
/// Calculate the shortest path in a grid from one pair to another using the A*
/// algorithm.
///
/// @details https://en.wikipedia.org/wiki/A%2A_search_algorithm
/// @param grid - The 2D grid which represents the dungeon.
/// @param start - The start position for the algorithm.
/// @param end - The end position for the algorithm.
/// @throws std::length_error - If the grid size is less than 0.
/// @return A vector of positions mapping out the shortest path from start to end.
auto calculate_astar_path(const Grid &grid, const Position &start, const Position &end) -> std::vector<Position>;

/// Generate a random position in a grid using the Dijkstra map algorithm.
///
/// @param grid - The grid to generate the Dijkstra map position for.
/// @param item_positions - The positions of the items to generate the Dijkstra map position for.
/// @param within - Whether to get a position within the minimum distance or not.
/// @throws std::length_error - If the grid size is less than 0.
/// @return A random Dijkstra map position.
auto generate_item_position(const Grid &grid, const std::unordered_set<Position> &item_positions, bool within)
    -> Position;
