#ifndef PRIMITIVES_H
#define PRIMITIVES_H
// Custom includes
#include "primitives.h"

#endif

// Std includes
#include <queue>
#include <iostream>

// ----- CONSTANTS ------------------------------
/* The ID of a TileType obstacle */
const std::vector<Point> INTERCARDINAL_OFFSETS = {
    {-1, -1},
    {0,  -1},
    {1,  -1},
    {-1, 0},
    {1,  0},
    {-1, 1},
    {0,  1},
    {1,  1},
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
        // The priority_queue data structure gets the maximum priority, so we need to
        // override that functionality to get the minimum priority
        return cost > nghbr.cost;
    }
};


// ----- FUNCTIONS ------------------------------
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
std::vector<Point> grid_bfs(Point &target, int height, int width) {
    // Create a vector to store the neighbours
    std::vector<Point> result;

    // Iterate over each offset and check if it's a valid neighbour
    for (Point offset: INTERCARDINAL_OFFSETS) {
        int x = target.x + offset.x;
        int y = target.y + offset.y;
        if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
            result.emplace_back(x, y);
        }
    }

    // Return the result
    return result;
}


/// Calculate the shortest path in a grid from one pair to another using the A* algorithm.
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
/// Returns
/// -------
/// A vector of points mapping out the shortest path from start to end.
std::vector<Point> calculate_astar_path(std::vector<std::vector<int>> &grid, Point &start, Point &end) {
    // Set up a few variables needed for the pathfinding
    std::vector<Point> result;
    std::priority_queue<Neighbour> queue;
    std::unordered_map<Point, Point> came_from = {{start, start}};
    std::unordered_map<Point, int> distances = {{start, 0}};
    int height = (int) grid.capacity();
    int width = (int) grid[0].capacity();
    queue.push({0, start});

    // Loop until the priority queue is empty
    while (!queue.empty()) {
        // Get the lowest cost pair from the priority queue
        Point current = queue.top().pair;
        queue.pop();

        // Check if we've reached our target
        if (current == end) {
            // Backtrack through came_from to get the path
            while (!(came_from[current] == current)) {
                // Add the current pair to the result list
                result.emplace_back(current.x, current.y);

                // Get the next pair in the path
                current = came_from[current];
            }

            // Add the start pair and exit out of the loop
            result.emplace_back(start.x, start.y);
            break;
        }

        // Add all the neighbours to the heap with their cost being f = g + h:
        //   f - The total cost of traversing the neighbour.
        //   g - The distance between the start pair and the neighbour pair.
        //   h - The estimated distance from the neighbour pair to the end pair. We're using the Chebyshev distance for
        //       this.
        for (Point neighbour: grid_bfs(current, height, width)) {
            if (!came_from.count(neighbour)) {
                // Store the neighbour's parent and calculate its distance from the start pair
                came_from.emplace(neighbour, current);
                distances.emplace(neighbour, distances[current] + 1);

                // Check if the neighbour is an obstacle. If so, set the total cost to infinity, otherwise, set it to f = g + h
                int f_cost = (grid[neighbour.y][neighbour.x] == TileType::Obstacle) ? std::numeric_limits<int>::max() :
                             distances[neighbour] +
                             std::max(abs(neighbour.x - current.x), abs(neighbour.y - current.y));

                // Add the neighbour to the priority queue
                queue.push({f_cost, neighbour});
            }
        }
    }

    // Return result
    return result;
}
