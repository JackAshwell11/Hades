// Ensure this file is only included once
#pragma once

// Custom includes
#include "primitives.hpp"

// ----- DEFINITIONS ------------------------------
/// Calculate the shortest path in a grid from one pair to another using the A*
/// algorithm.
///
/// Further reading which may be useful:
/// `The A* algorithm <https://en.wikipedia.org/wiki/A*_search_algorithm>`_
///
/// Parameters
/// ----------
/// grid - The 2D grid which represents the dungeon.
/// start - The start pair for the algorithm.
/// end - The end pair for the algorithm.
///
/// Throws
/// ------
/// std::length_error - Grid size must be bigger than 0.
///
/// Returns
/// -------
/// A vector of points mapping out the shortest path from start to end.
std::vector<Point> calculate_astar_path(Grid &grid, Point start, Point end);
