/// Stores objects that are shared between all generation classes.
use crate::generation::constants::{TileType, REPLACEABLE_TILES};
use ndarray::{s, Array2};
use std::cmp::{max, min};

/// Represents a point in the grid.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    /// Construct a Point object.
    ///
    /// # Parameters
    /// * `x` - The x position.
    /// * `y` - The y position.
    ///
    /// # Returns
    /// A Point object.
    pub fn new(x: i32, y: i32) -> Point {
        Point { x, y }
    }

    /// Calculate the sum of two points.
    ///
    /// # Parameters
    /// * `other` - The point to calculate the sum with.
    ///
    /// # Returns
    /// The sum of two points.
    #[inline]
    pub fn sum(&self, other: &Point) -> (i32, i32) {
        (self.x + other.x, self.y + other.y)
    }

    /// Calculate the absolute difference between two points.
    ///
    /// # Parameters
    /// * `other` - The point to calculate the absolute difference with.
    ///
    /// # Returns
    /// The absolute difference between two points.
    #[inline]
    pub fn abs_diff(&self, other: &Point) -> (i32, i32) {
        ((self.x - other.x).abs(), (self.y - other.y).abs())
    }
}

/// Represents a rectangle of any size useful for the interacting with the 2D grid.
///
/// When creating a container, the split wall is included in the rect size, whereas,
/// rooms don't so MIN_CONTAINER_SIZE must be bigger than MIN_ROOM_SIZE.
///
/// # Attributes
/// * `center` - The center position of the rect.
/// * `width` - The width of the rect.
/// * `height` - The height of the rect.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Rect {
    // Parameters
    pub top_left: Point,
    pub bottom_right: Point,

    // Attributes
    pub center: Point,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    /// Construct a Rect object.
    ///
    /// # Parameters
    /// * `top-left` - The top-left position.
    /// * `bottom-right` - The bottom-right position.
    ///
    /// # Returns
    /// A Rect object.
    pub fn new(top_left: Point, bottom_right: Point) -> Rect {
        let sum: (i32, i32) = top_left.sum(&bottom_right);
        let diff: (i32, i32) = top_left.abs_diff(&bottom_right);
        Rect {
            top_left,
            bottom_right,
            center: Point::new(
                ((sum.0 as f32) / 2.0).round() as i32,
                ((sum.1 as f32) / 2.0).round() as i32,
            ),
            width: diff.0,
            height: diff.1,
        }
    }

    /// Get the Chebyshev distance to another rect.
    ///
    /// # Parameters
    /// * `other` - The rect to find the distance to.
    ///
    /// # Returns
    /// The Chebyshev distance between this rect and the given rect.
    #[inline]
    pub fn get_distance_to(&self, other: &Rect) -> i32 {
        return max(
            (self.center.x - other.center.x).abs(),
            (self.center.y - other.center.y).abs(),
        );
    }

    /// Place the rect in the 2D grid.
    ///
    /// # Parameters
    /// * 'grid' - The 2D grid which represents the dungeon.
    pub fn place_rect(&self, grid: &mut Array2<TileType>) {
        // Get the width and height of the grid
        let grid_height: usize = *grid.shape().get(0).unwrap();
        let grid_width: usize = *grid.shape().get(1).unwrap();

        // Place the walls
        grid.slice_mut(s![
            max(self.top_left.y, 0)..min(self.bottom_right.y + 1, grid_height as i32),
            max(self.top_left.x, 0)..min(self.bottom_right.x + 1, grid_width as i32)
        ])
        .map_inplace(move |x| {
            // TODO: DETERMINE IF MOVE IS NEEDED
            if REPLACEABLE_TILES.contains(x) {
                *x = TileType::Wall.clone()
            }
        });

        // Place the floors. The ranges must be -1 in all directions since we don't want to
        // overwrite the walls keeping the player in, but we still want to overwrite walls that
        // block the path for hallways
        grid.slice_mut(s![
            max(self.top_left.y + 1, 1)..min(self.bottom_right.y, (grid_height - 1) as i32),
            max(self.top_left.x + 1, 1)..min(self.bottom_right.x, (grid_width - 1) as i32)
        ])
        .fill(TileType::Floor);
    }
}
