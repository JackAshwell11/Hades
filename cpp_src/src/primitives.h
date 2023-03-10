// Custom includes
#include "constants.h"

// External includes
#include <unordered_map>

// ----- STRUCTURES ------------------------------
/// Allows multiple hashes to be combined for a struct
template<class T>
inline void hash_combine(size_t &seed, const T &v) {
    std::hash <T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/// Represents a point in the grid.
struct Point {
    int x, y;

    inline bool operator==(const Point pnt) const {
        return x == pnt.x && y == pnt.y;
    }

    inline bool operator!=(const Point pnt) const {
        return x != pnt.x && y != pnt.y;
    }

    /// Default constructor for a Point object. This should not be used.
    Point() = default;

    /// Construct a Point object.
    ///
    /// Parameters
    /// ----------
    /// x - The x position.
    /// y - The y position.
    ///
    /// Returns
    /// -------
    /// A Point object.
    Point(int x_val, int y_val) {
        x = x_val;
        y = y_val;
    }

    /// Calculate the sum of two points.
    ///
    /// Parameters
    /// ----------
    /// other - The point to calculate the sum with.
    ///
    /// Returns
    /// -------
    /// The sum of two points.
    inline std::pair<int, int> sum(Point &other) const {
        return std::make_pair(x + other.x, y + other.y);
    }

    /// Calculate the absolute difference between two points.
    ///
    /// Parameters
    /// ----------
    /// other - The point to calculate the absolute difference with.
    ///
    /// Returns
    /// -------
    /// The absolute difference between two points.
    inline std::pair<int, int> abs_diff(Point &other) const {
        return std::make_pair(abs(x - other.x), abs(y - other.y));
    }
};


/// Allows the point struct to be hashed in a map.
template<>
struct std::hash<Point> {
    size_t operator()(const Point &pnt) const {
        size_t res = 0;
        hash_combine(res, pnt.x);
        hash_combine(res, pnt.y);
        return res;
    }
};

/// Represents a rectangle of any size useful for the interacting with the 2D grid.
///
/// When creating a container, the split wall is included in the rect size, whereas,
/// rooms don't so MIN_CONTAINER_SIZE must be bigger than MIN_ROOM_SIZE.
///
/// Attributes
/// ----------
/// center - The center position of the rect.
/// width - The width of the rect.
/// height - The height of the rect.
struct Rect {
    // Parameters
    Point top_left, bottom_right;

    // Attributes
    Point center;
    int width, height;

    inline bool operator==(const Rect rct) const {
        return top_left == rct.top_left && bottom_right == rct.bottom_right;
    }

    inline bool operator!=(const Rect rct) const {
        return top_left != rct.top_left && bottom_right != rct.bottom_right;
    }

    /// Default constructor for a Rect object. This should not be used.
    Rect() = default;

    /// Construct a Rect object.
    ///
    /// Parameters
    /// ----------
    /// top-left - The top-left position.
    /// bottom-right - The bottom-right position.
    ///
    /// Returns
    /// -------
    /// A Rect object.
    Rect(Point top_left_val, Point bottom_right_val) {
        std::pair<int, int> sum = top_left_val.sum(bottom_right_val);
        std::pair<int, int> diff = top_left_val.abs_diff(bottom_right_val);
        top_left = top_left_val;
        bottom_right = bottom_right_val;
        center = Point((int) round(((double) sum.first) / 2.0), (int) round(((double) sum.second) / 2.0));
        width = diff.first;
        height = diff.second;
    }

    /// Get the Chebyshev distance to another rect.
    ///
    /// Parameters
    /// ----------
    /// other - The rect to find the distance to.
    ///
    /// Returns
    /// -------
    /// The Chebyshev distance between this rect and the given rect.
    inline int get_distance_to(Rect &other) const {
        return std::max(abs(center.x - other.center.x), abs(center.y - other.center.y));
    }

    /// Place the rect in the 2D grid.
    ///
    /// Parameters
    /// ----------
    /// grid - The 2D grid which represents the dungeon.
    void place_rect(std::vector<std::vector<int>> &grid) const {
        // Get the width and height of the grid
        int grid_height = (int) grid.size();
        int grid_width = (int) grid[0].size();

        // Place the walls
        for (int y = std::max(top_left.y, 0); y < std::min(bottom_right.y + 1, grid_height); y++) {
            for (int x = std::max(top_left.x, 0); x < std::min(bottom_right.x + 1, grid_width); x++) {
                if (std::count(std::begin(REPLACEABLE_TILES), std::end(REPLACEABLE_TILES), grid[y][x]) > 0) {
                    grid[y][x] = TileType::Wall;
                }
            }
        }

        // Place the floors. The ranges must be -1 in all directions since we don't want to
        // overwrite the walls keeping the player in, but we still want to overwrite walls that
        // block the path for hallways
        for (int y = std::max(top_left.y + 1, 1); y < std::min(bottom_right.y, grid_height - 1); y++) {
            for (int x = std::max(top_left.x + 1, 1); x < std::min(bottom_right.x, grid_width - 1); x++) {
                grid[y][x] = TileType::Floor;
            }
        }
    }
};

/// Allows the rect struct to be hashed in a map.
template<>
struct std::hash<Rect> {
    size_t operator()(const Rect &rct) const {
        size_t res = 0;
        hash_combine(res, rct.top_left);
        hash_combine(res, rct.bottom_right);
        return res;
    }
};
