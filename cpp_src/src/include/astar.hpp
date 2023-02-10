// Ensure this file is only included once
#pragma once

// Custom includes
#include "primitives.hpp"

// ----- CONSTANTS ------------------------------
/* Represents the north, south, east, west, north-east, north-west, south-east
 * and south-west directions on a compass */
const std::vector<Point> INTERCARDINAL_OFFSETS = {
    {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1},
};

// ----- STRUCTURES ------------------------------
/// Represents a grid position and its costs from the start position
///
/// Parameters
/// ----------
/// cost - The cost to traverse to this neighbour.
/// pair - The position in the grid.
struct Neighbour {
  int cost;
  Point pair;

  inline bool operator<(const Neighbour nghbr) const {
    // The priority_queue data structure gets the maximum priority, so we need
    // to override that functionality to get the minimum priority
    return cost >= nghbr.cost;
  }
};

// ----- DEFINITIONS ------------------------------
/// Get a target's neighbours based on a given list of offsets.
///
/// Parameters
/// ----------
/// target - The target to get neighbours for.
/// height - The height of the grid.
/// width - The width of the grid.
/// offsets - The offsets to used to calculate the neighbours.
///
/// Returns
/// -------
/// A vector of the target's neighbours.
std::vector<Point> grid_bfs(Point &target, int height, int width);

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
std::vector<Point> calculate_astar_path(std::vector<std::vector<TileType>> &grid, Point &start, Point &end);
