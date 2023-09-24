// Ensure this file is only included once
#pragma once

// Custom includes
#include "primitives.hpp"

// ----- DEFINITIONS ------------------------------
/// Calculate the shortest path in a grid from one pair to another using the A*
/// algorithm.
///
/// @details https://en.wikipedia.org/wiki/A*_search_algorithm
///
/// @param grid - The 2D grid which represents the dungeon.
/// @param start - The start position for the algorithm.
/// @param end - The end position for the algorithm.
/// @throws std::length_error - Grid size must be bigger than 0.
/// @return A vector of positions mapping out the shortest path from start to end.
std::vector<Position> calculate_astar_path(Grid &grid, Position start, Position end);
