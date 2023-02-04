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
    return cost > nghbr.cost;
  }
};

// ----- DEFINITIONS ------------------------------
std::vector<Point> grid_bfs(Point &target, int height, int width);

std::vector<Point> calculate_astar_path(std::vector<std::vector<TileType>> &grid, Point &start, Point &end);
