/// Calculate the shortest path in a grid from one pair to another using the A* algorithm
use crate::generation::constants::TileType;
use crate::generation::primitives::Point;
use ahash::RandomState;
use ndarray::Array2;
use std::cmp::{max, Ordering};
use std::collections::{BinaryHeap, HashMap};

// Represents the north, south, east, west, north-east, north-west, south-east and south-west directions on a compass
const INTERCARDINAL_OFFSETS: [Point; 8] = [
    Point { x: -1, y: -1 },
    Point { x: 0, y: -1 },
    Point { x: 1, y: -1 },
    Point { x: -1, y: 0 },
    Point { x: 1, y: 0 },
    Point { x: -1, y: 1 },
    Point { x: 0, y: 1 },
    Point { x: 1, y: 1 },
];

/// Represents a grid position and its costs from the start position
///
/// # Parameters
/// * `cost` - The cost to traverse to this neighbour.
/// * `pair` - The position in the grid.
#[derive(PartialEq, Eq)]
struct Neighbour {
    cost: i32,
    pair: Point,
    parent: Point,
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

#[inline]
fn walkable(grid: &Array2<TileType>, x: i32, y: i32) -> bool {
    (0 <= x)
        && (x < *grid.shape().get(1).unwrap() as i32)
        && (0 <= y)
        && (y < *grid.shape().get(0).unwrap() as i32)
        && (grid[[y as usize, x as usize]] != TileType::Obstacle)
}

fn jump(grid: &Array2<TileType>, current: Point, parent: Point, end: &Point) -> Option<Point> {
    if !walkable(grid, current.x, current.y) {
        return None;
    }

    if current == *end {
        return Option::from(current);
    }

    let (dx, dy) = (current.x - parent.x, current.y - parent.y);
    if dx != 0 && dy != 0 {
        if (!walkable(grid, current.x - dx, current.y)
            && walkable(grid, current.x - dx, current.y + dy))
            || (!walkable(grid, current.x, current.y - dy)
                && walkable(grid, current.x + dx, current.y - dy))
        {
            return Option::from(current);
        }

        if jump(grid, Point::new(current.x + dx, current.y), current, end).is_some()
            || jump(grid, Point::new(current.x, current.y + dy), current, end).is_some()
        {
            return Option::from(current);
        }
    } else if dx != 0 {
        if (!walkable(grid, current.x, current.y - 1)
            && walkable(grid, current.x + dx, current.y - 1))
            || (!walkable(grid, current.x, current.y + 1)
                && walkable(grid, current.x + dx, current.y + 1))
        {
            return Option::from(current);
        }
    } else if dy != 0 {
        if (!walkable(grid, current.x - 1, current.y)
            && walkable(grid, current.x - 1, current.y - dy))
            || (!walkable(grid, current.x + 1, current.y)
                && walkable(grid, current.x + 1, current.y - dy))
        {
            return Option::from(current);
        }
    }

    if walkable(grid, current.x + dx, current.y) || walkable(grid, current.x, current.y + dy) {
        return jump(
            grid,
            Point::new(current.x + dx, current.y + dy),
            current,
            end,
        );
    }

    return None;
}

fn prune_neighbours(current: &Point, parent: &Point) -> Vec<Point> {
    let mut neighbours: Vec<Point> = Vec::new();
    let (dx, dy) = (
        (current.x - parent.x) / max((current.x - parent.x).abs(), 1),
        (current.y - parent.y) / max((current.y - parent.y).abs(), 1),
    );

    if dx != 0 && dy != 0 {
        neighbours.push(Point::new(current.x + dx, current.y));
        neighbours.push(Point::new(current.x, current.y + dy));
        neighbours.push(Point::new(current.x + dx, current.y + dy));

        // not sure about these
        neighbours.push(Point::new(current.x + dx, current.y - dy));
        neighbours.push(Point::new(current.x - dx, current.y + dy));
    } else if dx != 0 {
        neighbours.push(Point::new(current.x + dx, current.y - 1));
        neighbours.push(Point::new(current.x + dx, current.y));
        neighbours.push(Point::new(current.x + dx, current.y + 1));
    } else if dy != 0 {
        neighbours.push(Point::new(current.x - 1, current.y + dy));
        neighbours.push(Point::new(current.x, current.y + dy));
        neighbours.push(Point::new(current.x + 1, current.y + dy));
    } else {
        let bfs_neighbours: Vec<Point> = INTERCARDINAL_OFFSETS
            .iter()
            .map(|offset| Point::new(current.x + offset.x, current.y + offset.y))
            .collect();
        neighbours.extend(bfs_neighbours);
    }

    return neighbours;
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
        parent: *start,
    }]);
    let mut came_from: HashMap<Point, Point, RandomState> = HashMap::default();
    let mut distances: HashMap<Point, i32, RandomState> = HashMap::default();
    came_from.insert(*start, *start);
    distances.insert(*start, 0);

    // Loop until the priority queue is empty
    while !queue.is_empty() {
        // Get the lowest cost pair from the priority queue
        let next: Neighbour = queue.pop().unwrap();
        let (mut current, mut parent) = (next.pair, next.parent);

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
        for neighbour in prune_neighbours(&current, &parent) {
            let jump_point: Option<Point> = jump(&grid, neighbour, current, &end);
            if jump_point.is_some() && !came_from.contains_key(&jump_point.unwrap()) {
                // Store the neighbour's parent and calculate its distance from the start pair
                came_from.insert(jump_point.unwrap(), current);
                distances.insert(jump_point.unwrap(), distances.get(&current).unwrap() + 1);

                // Check if the neighbour is an obstacle. If so, set the total cost to infinity,
                // otherwise, set it to f = g + h
                let f_cost: i32 = distances[&jump_point.unwrap()]
                    + max(
                        (jump_point.unwrap().x - current.x).abs(),
                        (jump_point.unwrap().y - current.y).abs(),
                    );

                // Add the neighbour to the priority queue
                queue.push(Neighbour {
                    cost: f_cost,
                    pair: jump_point.unwrap(),
                    parent: current,
                });
            }
        }
    }

    // Return the result
    return result;
}
