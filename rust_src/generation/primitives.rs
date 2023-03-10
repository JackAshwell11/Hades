use pyo3::{pyclass, pymethods};
use std::cmp::max;

/// Represents a point in the grid.
///
/// Parameters
/// ----------
/// x: int
///     The x position.
/// y: int
///     The y position.
#[pyclass(module = "hades_extensions.generation")]
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct Point {
    #[pyo3(get)]
    pub x: i32,
    #[pyo3(get)]
    pub y: i32,
}

#[pymethods]
impl Point {
    #[new]
    fn new(x: i32, y: i32) -> Point {
        Point { x, y }
    }

    /// Return a human-readable representation of this object in Python.
    fn __repr__(&self) -> String {
        format!("Point(x={}, y={})", self.x, self.y)
    }

    /// Calculates the sum of two points.
    ///
    /// Parameters
    /// ----------
    /// other: Rect
    ///     The point to calculate the sum with.
    pub fn sum(&self, other: &Point) -> (i32, i32) {
        (self.x + other.x, self.y + other.y)
    }

    /// Calculates the absolute difference between two points.
    ///
    /// Parameters
    /// ----------
    /// other: Point
    ///     The point to calculate the absolute difference with.
    pub fn abs_diff(&self, other: &Point) -> (i32, i32) {
        ((self.x - other.x).abs(), (self.y - other.y).abs())
    }
}

/// Represents a rectangle of any size useful for the interacting with the 2D grid.
///
/// When creating a container, the split wall is included in the rect size, whereas,
/// rooms don't so MIN_CONTAINER_SIZE must be bigger than MIN_ROOM_SIZE.
///
/// Parameters
/// ----------
/// top-left: Point
///     The top-left position.
/// bottom-right: Point
///     The bottom-right position.
#[pyclass(module = "hades_extensions.generation")]
pub struct Rect {
    #[pyo3(get)]
    pub top_left: Point,
    #[pyo3(get)]
    pub bottom_right: Point,
    #[pyo3(get)]
    pub width: i32,
    #[pyo3(get)]
    pub height: i32,
    #[pyo3(get)]
    pub center: Point,
}

#[pymethods]
impl Rect {
    #[new]
    fn new(top_left: Point, bottom_right: Point) -> Rect {
        let sum: (i32, i32) = top_left.sum(&bottom_right);
        let diff: (i32, i32) = bottom_right.abs_diff(&top_left);
        Rect {
            top_left,
            bottom_right,
            width: diff.0,
            height: diff.1,
            center: Point {
                x: ((sum.0 as f32) / 2.0).round() as i32,
                y: ((sum.1 as f32) / 2.0).round() as i32,
            },
        }
    }

    /// Return a human-readable representation of this object in Python.
    fn __repr__(&self) -> String {
        format!(
            "Rect(top_left={}, bottom_right={}, width={}, height={}, center={})",
            self.top_left.__repr__(),
            self.bottom_right.__repr__(),
            self.width,
            self.height,
            self.center.__repr__()
        )
    }

    /// Get the Chebyshev distance to another rect.
    ///
    /// Parameters
    /// ----------
    /// other: Rect
    ///     The rect to find the distance to.
    ///
    /// Returns
    /// -------
    /// int
    ///     The Chebyshev distance between this rect and the given rect.
    pub fn get_distance_to(&self, other: &Rect) -> i32 {
        return max(
            (self.center.x - other.center.x).abs(),
            (self.center.y - other.center.y).abs(),
        );
    }

    /// Places the rect in the 2D grid.
    ///
    /// Parameters
    /// ----------
    /// grid: list[list[int]]
    ///     The 2D grid which represents the dungeon.
    pub fn place_rect(&self, grid: Vec<Vec<i32>>) {}
}
