/// Calculate the shortest path in a grid from one pair to another using the A* algorithm
use crate::generation::constants::TileType;
use crate::generation::primitives::Point;
use ahash::RandomState;
use ndarray::Array2;
use std::cmp::{max, Ordering};
use std::collections::{BinaryHeap, HashMap};

/// Represents a grid position and its costs from the start position
///
/// # Parameters
/// * `cost` - The cost to traverse to this neighbour.
/// * `pair` - The position in the grid.
#[derive(PartialEq, Eq)]
struct Neighbour {
    cost: i32,
    pair: Point,
}

// We have to implement this in order for the Ord trait to work
impl PartialOrd for Neighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.cost.partial_cmp(&self.cost)
    }
}

// We have to implement this in order for the min heap to work
impl Ord for Neighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Get a target's neighbours based on a given list of offsets.
///
/// # Parameters
/// * `target` - The target to get neighbours for.
/// * `height` - The height of the grid.
/// * `width` - The width of the grid.
/// * `offsets` - The offsets to used to calculate the neighbours.
///
/// # Returns
/// A vector of the target's neighbours.
fn grid_bfs(target: &Point, height: i32, width: i32) -> Vec<Point> {
    // Create a vector to store the neighbours
    let mut result: Vec<Point> = vec![];

    // Iterate over each offset and check if it's a valid neighbour
    for offset in [
        Point { x: -1, y: -1 },
        Point { x: 0, y: -1 },
        Point { x: 1, y: -1 },
        Point { x: -1, y: 0 },
        Point { x: 1, y: 0 },
        Point { x: -1, y: 1 },
        Point { x: 0, y: 1 },
        Point { x: 1, y: 1 },
    ] {
        let x: i32 = target.x + offset.x;
        let y: i32 = target.y + offset.y;
        if (x >= 0 && x < width) && (y >= 0 && y < height) {
            result.push(Point { x, y });
        }
    }

    // Return the result
    return result;
}

/// Calculate the shortest path in a grid from one pair to another using the A* algorithm.
///
/// Further reading which may be useful:
/// * `The A* algorithm <https://en.wikipedia.org/wiki/A*_search_algorithm>`_
///
/// # Parameters
/// * `grid` - The 2D grid which represents the dungeon.
/// * `start` - The start pair for the algorithm.
/// * `end` - The end pair for the algorithm.
///
/// # Returns
/// A vector of points mapping out the shortest path from start to end.
pub fn calculate_astar_path(grid: &Array2<TileType>, start: &Point, end: &Point) -> Vec<Point> {
    // Set up a few variables needed for the pathfinding
    let mut result: Vec<Point> = vec![];
    let mut queue: BinaryHeap<Neighbour> = BinaryHeap::from([Neighbour {
        cost: 0,
        pair: *start,
    }]);
    let mut came_from: HashMap<Point, Point, RandomState> = HashMap::default();
    let mut distances: HashMap<Point, i32, RandomState> = HashMap::default();
    let height: i32 = *grid.shape().get(0).unwrap() as i32;
    let width: i32 = *grid.shape().get(1).unwrap() as i32;
    came_from.insert(*start, *start);
    distances.insert(*start, 0);

    // Loop until the priority queue is empty
    while !queue.is_empty() {
        // Get the lowest cost pair from the priority queue
        let mut current: Point = queue.pop().unwrap().pair;

        // Check if we've reached our target
        if current == *end {
            // Backtrack through came_from to get the path
            while came_from[&current] != current {
                // Add the current pair to the result list
                result.push(Point {
                    x: current.x,
                    y: current.y,
                });

                // Get the next pair in the path
                current = came_from[&current];
            }

            // Add the start pair and exit out of the loop
            result.push(Point {
                x: start.x,
                y: start.y,
            });
            break;
        }

        // Add all the neighbours to the heap with their cost being f = g + h:
        //   f - The total cost of traversing the neighbour.
        //   g - The distance between the start pair and the neighbour pair.
        //   h - The estimated distance from the neighbour pair to the end pair. We're using the
        //       Chebyshev distance for this.
        for neighbour in grid_bfs(&current, height, width) {
            if !came_from.contains_key(&neighbour) {
                // Store the neighbour's parent and calculate its distance from the start pair
                came_from.insert(neighbour, current);
                distances.insert(neighbour, distances.get(&current).unwrap() + 1);

                // Check if the neighbour is an obstacle. If so, set the total cost to infinity,
                // otherwise, set it to f = g + h
                let f_cost: i32 =
                    if grid[[neighbour.y as usize, neighbour.x as usize]] == TileType::Obstacle {
                        i32::MAX
                    } else {
                        distances[&neighbour]
                            + max(
                                (neighbour.x - current.x).abs(),
                                (neighbour.y - current.y).abs(),
                            )
                    };

                // Add the neighbour to the priority queue
                queue.push(Neighbour {
                    cost: f_cost,
                    pair: neighbour,
                });
            }
        }
    }

    // Return the result
    return result;
}
