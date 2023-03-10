/// Manages the generation of the dungeon and placing of game objects.
use crate::generation::astar::calculate_astar_path;
use crate::generation::bsp::Leaf;
use crate::generation::constants::{
    TileType, HALLWAY_SIZE, ITEM_PROBABILITIES, MAP_GENERATION_CONSTANTS,
};
use crate::generation::primitives::{Point, Rect};
use ahash::RandomState;
use ndarray::Array2;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// Holds the constants for a specific level.
///
/// # Parameters
/// * `level` - The level of this game.
/// * `width` - The width of the game map.
/// * `height` - The height of the game map.
#[pyclass(module = "hades.generation")]
pub struct LevelConstants {
    level: i32,
    width: i32,
    height: i32,
}

/// Represents an undirected weighted edge in a graph. It contains 3 elements: a cost, a source
/// vertex, and a destination vertex.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Edge(i32, Rect, Rect);

/// Collect all points in a given grid that match the target.
///
/// # Parameters
/// * `grid` - The 2D grid which represents the dungeon.
/// * `target` - The TileType to test for.
///
/// # Returns
/// A vector of points which match the target.
#[inline]
fn collect_positions(grid: &Array2<TileType>, target: &TileType) -> Vec<Point> {
    grid.indexed_iter()
        .filter(|((_, _), value)| *value == target)
        .map(|((x, y), _)| Point {
            x: x as i32,
            y: y as i32,
        })
        .collect()
}

/// Split the bsp based on the generated constants.
///
/// # Parameters
/// * `bsp` - The root leaf for the binary space partition.
/// * `grid` - The 2D grid which represents the dungeon.
/// * `random_generator` - The random generator used to generate the bsp.
/// * `split_iteration` - The number of splits to perform.
fn split_bsp(
    bsp: &mut Leaf,
    grid: &mut Array2<TileType>,
    random_generator: &mut StdRng,
    mut split_iteration: i32,
) {
    // Start the splitting using a queue
    let mut queue: VecDeque<&mut Leaf> = VecDeque::from([bsp]);
    while split_iteration > 0 && !queue.is_empty() {
        // Get the current leaf from the deque object
        let current: &mut Leaf = queue.pop_front().unwrap();

        // Split the bsp if possible
        if current.split(grid, random_generator, true)
            && current.left.is_some()
            && current.right.is_some()
        {
            // Add the child leafs so they can be split
            queue.push_back(current.left.as_mut().unwrap());
            queue.push_back(current.right.as_mut().unwrap());

            // Decrement the split iteration
            split_iteration -= 1;
        }
    }
}

/// Generate the rooms for a given game level using the bsp.
///
/// # Parameters
/// * `bsp` - The root leaf for the binary space partition.
/// * `grid` - The 2D grid which represents the dungeon.
/// * `random_generator` - The random generator used to generate the bsp.
///
/// # Returns
/// The generated rooms.
fn generate_rooms(
    bsp: &mut Leaf,
    grid: &mut Array2<TileType>,
    random_generator: &mut StdRng,
) -> Vec<Rect> {
    // Create the rooms
    let mut rooms: Vec<Rect> = vec![];
    let mut queue: VecDeque<&mut Leaf> = VecDeque::from([bsp]);
    while !queue.is_empty() {
        // Get the current leaf from the stack
        let current: &mut Leaf = queue.pop_back().unwrap();

        // Check if a room already exists in this leaf
        if current.room.is_some() {
            continue;
        }

        // Test if we can create a room in the current leaf
        if current.left.is_some() && current.right.is_some() {
            // Room creation not successful meaning there are child leafs so try again on the child
            // leafs
            queue.push_back(current.left.as_mut().unwrap());
            queue.push_back(current.right.as_mut().unwrap());
        } else {
            // Create a room in the current leaf and save the rect
            while !current.create_room(grid, random_generator) {
                // Width to height ratio is outside of range so try again
                continue;
            }

            // Add the created room to the rooms list
            rooms.push(current.room.unwrap());
        }
    }

    // Return all the created rooms
    return rooms;
}

/// Create a set of connections between all the rects ensuring that every rect is reachable.
///
/// Further reading which may be useful:
/// * `Prim's algorithm <https://en.wikipedia.org/wiki/Prim's_algorithm>`_
///
/// # Parameters
/// * `complete_graph` - An adjacency list which represents a complete graph.
///
/// # Returns
/// A set of edges which form the connections between rects.
fn create_connections(complete_graph: &HashMap<Rect, Vec<Rect>, RandomState>) -> HashSet<Edge> {
    // Use Prim's algorithm to construct a minimum spanning tree from complete_graph
    let start: &Rect = complete_graph.keys().next().unwrap();
    let mut unexplored: BinaryHeap<Edge> = BinaryHeap::from([Edge(0, *start, *start)]);
    let mut visited: HashSet<Rect> = HashSet::new();
    let mut mst: HashSet<Edge> = HashSet::new();
    while mst.len() < complete_graph.len() - 1 {
        // Get the neighbour with the lowest cost
        let lowest: Edge = unexplored.pop().unwrap();

        // Check if the neighbour is already visited or not
        if visited.contains(&lowest.2) {
            continue;
        }

        // Neighbour isn't visited so mark it as visited and add it's neighbours to the heap
        visited.insert(lowest.2);
        for neighbour in &complete_graph[&lowest.2] {
            if !visited.contains(&neighbour) {
                unexplored.push(Edge(
                    lowest.2.get_distance_to(&neighbour),
                    lowest.2,
                    *neighbour,
                ));
            }
        }

        // Add a new edge towards the lowest cost neighbour onto the mst
        if lowest.1 != lowest.2 {
            // Save the connection
            mst.insert(lowest);
        }
    }

    // Return the constructed minimum spanning tree
    return mst;
}

/// Create the hallways by placing random obstacles and pathfinding around them.
///
/// # Parameters
/// * `grid` - The 2D grid which represents the dungeon.
/// * `random_generator` - The random generator used to pick the positions for the obstacles.
/// * `connections` - The connections to pathfind using the A* algorithm.
/// * `obstacle_count` - The number of obstacles to place in the 2D grid.
fn create_hallways(
    grid: &mut Array2<TileType>,
    random_generator: &mut StdRng,
    connections: &HashSet<Edge>,
    obstacle_count: i32,
) {
    // Place random obstacles in the grid
    let mut obstacle_positions: Vec<Point> = collect_positions(grid, &TileType::Empty);
    for _ in 0..obstacle_count {
        let obstacle_point: Point =
            obstacle_positions.swap_remove(random_generator.gen_range(0..obstacle_positions.len()));
        grid[[obstacle_point.x as usize, obstacle_point.y as usize]] = TileType::Obstacle;
    }

    // Use the A* algorithm with to connect each pair of rooms making sure to avoid the obstacles
    // giving us natural looking hallways. Note that the width of the hallways will always be odd in
    // this implementation due to numpy indexing
    const HALF_HALLWAY_SIZE: i32 = HALLWAY_SIZE / 2;
    let path_points: Vec<Point> = connections
        .par_iter()
        .flat_map(|connection| {
            calculate_astar_path(grid, &connection.1.center, &connection.2.center)
        })
        .collect();
    for path_point in path_points {
        // Place a rect box around the path_point using HALLWAY_SIZE to determine the width and
        // height
        Rect::new(
            Point {
                x: path_point.x - HALF_HALLWAY_SIZE,
                y: path_point.y - HALF_HALLWAY_SIZE,
            },
            Point {
                x: path_point.x + HALF_HALLWAY_SIZE,
                y: path_point.y + HALF_HALLWAY_SIZE,
            },
        )
        .place_rect(grid);
    }
}

/// Places a given tile in the 2D grid.
///
/// # Parameters
/// * `grid` - The 2D grid which represents the dungeon.
/// * `random_generator` - The random generator used to pick the positions for the obstacles.
/// * `target_tile` - The tile to place in the 2D grid.
/// * `possible_tiles` - The possible tiles that the tile can be placed into.
fn place_tile(
    grid: &mut Array2<TileType>,
    random_generator: &mut StdRng,
    target_tile: &TileType,
    possible_tiles: &mut Vec<Point>,
) {
    // Get a random floor position and place the target tile
    let tile_point: Point =
        possible_tiles.swap_remove(random_generator.gen_range(0..possible_tiles.len()));
    grid[[tile_point.x as usize, tile_point.y as usize]] = target_tile.clone();
}

/// Generate the game map for a given game level.
///
/// # Parameters
/// * `level` - The game level to generate a map for.
/// * `seed` - The seed to initialise the random generator.
///
/// # Returns
/// A tuple containing the generated map and the level constants.
#[pyfunction]
pub fn create_map(level: i32, seed: u64) -> (Vec<TileType>, LevelConstants) {
    // Initialise a few variables needed for the map generation
    let grid_width: i32 = MAP_GENERATION_CONSTANTS.width.generate_value(level);
    let grid_height: i32 = MAP_GENERATION_CONSTANTS.height.generate_value(level);
    let mut random_generator: StdRng = StdRng::seed_from_u64(seed);
    let mut grid: Array2<TileType> =
        Array2::<TileType>::from_elem((grid_height as usize, grid_width as usize), TileType::Empty);
    let mut bsp: Leaf = Leaf::new(Rect::new(
        Point::new(0, 0),
        Point::new(grid_width - 1, grid_height - 1),
    ));

    // Split the bsp and create the rooms
    split_bsp(
        &mut bsp,
        &mut grid,
        &mut random_generator,
        MAP_GENERATION_CONSTANTS
            .split_iteration
            .generate_value(level),
    );
    let rooms: Vec<Rect> = generate_rooms(&mut bsp, &mut grid, &mut random_generator);

    // Create the hallways between the rooms
    let mut complete_graph: HashMap<Rect, Vec<Rect>, RandomState> = HashMap::default();
    for room in &rooms {
        complete_graph.insert(
            *room,
            rooms.iter().filter(|x| *x != room).copied().collect(),
        );
    }
    create_hallways(
        &mut grid,
        &mut random_generator,
        &create_connections(&complete_graph),
        MAP_GENERATION_CONSTANTS
            .obstacle_count
            .generate_value(level),
    );

    // Get all the tiles which can support items being placed on them and then place the player and
    // all the items
    let item_limit: i32 = MAP_GENERATION_CONSTANTS.item_count.generate_value(level);
    let mut possible_tiles: Vec<Point> = collect_positions(&grid, &TileType::Floor);
    place_tile(
        &mut grid,
        &mut random_generator,
        &TileType::Player,
        &mut possible_tiles,
    );
    for (tile, probability) in ITEM_PROBABILITIES {
        let tile_limit: i32 = (probability * item_limit as f32).round() as i32;
        let mut tiles_placed: i32 = 0;
        while tiles_placed < tile_limit {
            place_tile(&mut grid, &mut random_generator, &tile, &mut possible_tiles);
            tiles_placed += 1;
        }
    }

    // Return the grid and a LevelConstants object
    return (
        grid.into_raw_vec(),
        LevelConstants {
            level,
            width: grid_width,
            height: grid_width,
        },
    );
}
