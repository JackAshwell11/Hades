// Std includes
#include <execution>
#include <optional>
#include <queue>
#include <iostream>

// Custom includes
#include "include/astar.hpp"
#include "include/map.hpp"

// ----- FUNCTIONS ------------------------------
std::vector<Point> collect_positions(std::vector<std::vector<TileType>> &grid, TileType target) {
  // Iterate over grid and check each position
  std::vector<Point> result;
  for (int y = 0; y < grid.size(); y++) {
    for (int x = 0; x < grid[y].size(); x++) {
      if (grid[y][x] == target) {
        result.emplace_back(x, y);
      }
    }
  }

  // Return result
  return result;
}

void split_bsp(Leaf &bsp, std::vector<std::vector<TileType>> &grid,
               std::mt19937 &random_generator, int split_iteration) {
  // Start the splitting using a queue
  std::queue<Leaf *> queue;
  queue.push(&bsp);
  while (split_iteration > 0 && !queue.empty()) {
    // Get the current leaf from the deque object
    Leaf *current = queue.front();
    queue.pop();

    // Split the bsp if possible
    if (current->split(grid, random_generator, true) && current->left &&
        current->right) {
      // Add the child leafs so they can be split
      queue.push(current->left);
      queue.push(current->right);

      // Decrement the split iteration
      split_iteration -= 1;
    }
  }
}

std::vector<Rect> generate_rooms(Leaf &bsp,
                                 std::vector<std::vector<TileType>> &grid,
                                 std::mt19937 &random_generator) {
  // Create the rooms
  std::vector<Rect> rooms;
  std::queue<Leaf> queue;
  queue.push(bsp);
  while (!queue.empty()) {
    // Get the current leaf from the stack
    Leaf current = queue.front();
    queue.pop();

    // Check if a room already exists in this leaf
    if (current.room) {
      continue;
    }

    // Test if we can create a room in the current leaf
    if (current.left && current.right) {
      // Room creation not successful meaning there are child leafs so try again
      // on the child leafs
      queue.push(*current.left);
      queue.push(*current.right);
    } else {
      // Create a room in the current leaf and save the rect. If a room cannot
      // be created, the width to height ratio is outside of range so try again
      while (!current.create_room(grid, random_generator)) {
      }

      // Add the created room to the rooms list
      Rect new_room = *current.room;
      rooms.emplace_back(new_room);
    }
  }

  // Return all the created rooms
  return rooms;
}

std::unordered_set<Edge> create_connections(
    std::unordered_map<Rect, std::vector<Rect>> &complete_graph) {
  // Check if complete_graph is valid
  if (complete_graph.empty()) {
    throw std::length_error("Complete graph size must be bigger than 0.");
  }

  // Use Prim's algorithm to construct a minimum spanning tree from
  // complete_graph
  Rect start = complete_graph.begin()->first;
  std::priority_queue<Edge> unexplored;
  std::unordered_set<Rect> visited;
  std::unordered_set<Edge> mst;
  unexplored.emplace(0, start, start);
  while (mst.size() < complete_graph.size() && !unexplored.empty()) {
    // Get the neighbour with the lowest cost
    Edge lowest = unexplored.top();
    unexplored.pop();

    // Check if the neighbour is already visited or not
    if (visited.count(lowest.destination)) {
      continue;
    }

    // Neighbour isn't visited so mark it as visited and add its neighbours to
    // the heap
    visited.insert(lowest.destination);
    for (Rect neighbour : complete_graph.at(lowest.destination)) {
      if (!visited.count(neighbour)) {
        unexplored.emplace(
            lowest.destination.get_distance_to(neighbour),
            lowest.destination,
            neighbour
        );
      }
    }

    // Add a new edge towards the lowest cost neighbour onto the mst
    if (lowest.source != lowest.destination) {
      // Save the connection
      mst.insert(lowest);
    }
  }

  // Return the constructed minimum spanning tree
  return mst;
}

void place_tile(std::vector<std::vector<TileType>> &grid,
                std::mt19937 &random_generator, TileType target_tile,
                std::vector<Point> &possible_tiles) {
  // Check if at least one tile exists
  if (possible_tiles.empty()) {
    throw std::length_error("Possible tiles size must be bigger than 0.");
  }

  // Get a random floor position and place the target tile
  std::uniform_int_distribution<> possible_tiles_distribution(
      0, (int) possible_tiles.size() - 1);
  int possible_tile_index = possible_tiles_distribution(random_generator);
  Point possible_tile = possible_tiles.at(possible_tile_index);
  possible_tiles.erase(possible_tiles.begin() + possible_tile_index);
  grid[possible_tile.y][possible_tile.x] = target_tile;
}

void create_hallways(std::vector<std::vector<TileType>> &grid,
                     std::mt19937 &random_generator,
                     std::unordered_set<Edge> &connections,
                     int obstacle_count) {
  // Place random obstacles in the grid
  std::vector<Point> obstacle_positions =
      collect_positions(grid, TileType::Empty);
  for (int i = 0; i < obstacle_count; i++) {
    place_tile(grid, random_generator, TileType::Obstacle, obstacle_positions);
  }

  // Use the A* algorithm with to connect each pair of rooms making sure to
  // avoid the obstacles giving us natural looking hallways
  const int HALF_HALLWAY_SIZE = HALLWAY_SIZE / 2;
  std::vector<std::vector<Point>> path_points;
  path_points.resize(connections.size());
  std::transform(std::execution::par, connections.begin(), connections.end(),
                 path_points.begin(), [&grid](Edge connection) {
        return calculate_astar_path(grid, connection.source.center,
                                    connection.destination.center);
      });
  for (const std::vector<Point> &path : path_points) {
    for (Point path_point : path) {
      // Place a rect box around the path_point using HALLWAY_SIZE to determine
      // the width and height
      Rect{Point{
          path_point.x - HALF_HALLWAY_SIZE,
          path_point.y - HALF_HALLWAY_SIZE,
      },
           Point{
               path_point.x + HALF_HALLWAY_SIZE,
               path_point.y + HALF_HALLWAY_SIZE,
           }}
          .place_rect(grid);
    }
  }
}

std::pair<std::vector<std::vector<TileType>>, std::tuple<int, int, int>>
create_map(int level, std::optional<unsigned int> seed) {
  // Check that the level number is valid
  if (level < 0) {
    throw std::length_error("Level must be bigger than or equal to 0");
  }

  // Create the random generator. If seed is None, then get a random unsigned integer
  unsigned int valid_seed;
  if (!seed.has_value()) {
    std::random_device random_device;
    std::mt19937_64 seed_generator(random_device());
    std::uniform_int_distribution<unsigned int> seed_distribution;
    valid_seed = seed_distribution(seed_generator);
  } else {
    valid_seed = seed.value();
  }
  std::mt19937 random_generator(valid_seed);

  // Initialise a few variables needed for the map generation
  int grid_width = MAP_GENERATION_CONSTANTS.width.generate_value(level);
  int grid_height = MAP_GENERATION_CONSTANTS.height.generate_value(level);
  std::vector<std::vector<TileType>> grid(
      grid_height, std::vector<TileType>(grid_width, TileType::Empty));
  Leaf bsp = Leaf{Rect{Point{0, 0}, Point{grid_width - 1, grid_height - 1}}};

  // Split the bsp and create the rooms
  split_bsp(bsp, grid, random_generator,
            MAP_GENERATION_CONSTANTS.split_iteration.generate_value(level));
  std::vector<Rect> rooms = generate_rooms(bsp, grid, random_generator);

  // Create the hallways between the rooms
  std::unordered_map<Rect, std::vector<Rect>> complete_graph;
  for (Rect room : rooms) {
    std::vector<Rect> temp;
    for (const Rect &x : rooms) {
      if (x != room) {
        temp.push_back(x);
      }
    }
    complete_graph.insert({room, temp});
  }
  std::unordered_set<Edge> connections = create_connections(complete_graph);
  create_hallways(
      grid, random_generator, connections,
      MAP_GENERATION_CONSTANTS.obstacle_count.generate_value(level));

  // Get all the tiles which can support items being placed on them and then
  // place the player and all the items
  int item_limit = MAP_GENERATION_CONSTANTS.item_count.generate_value(level);
  std::vector<Point> possible_tiles = collect_positions(grid, TileType::Floor);
  place_tile(grid, random_generator, TileType::Player, possible_tiles);
  for (std::pair<TileType, double> item : ITEM_PROBABILITIES) {
    int tile_limit = (int) round((double) (item.second * item_limit));
    int tiles_placed = 0;
    while (tiles_placed < tile_limit) {
      place_tile(grid, random_generator, item.first, possible_tiles);
      tiles_placed += 1;
    }
  }

  // Return the grid and a LevelConstants object
  return std::make_pair(grid, std::make_tuple(level, grid_width, grid_height));
}
