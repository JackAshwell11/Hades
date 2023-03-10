#ifndef PRIMITIVES_H
#define PRIMITIVES_H
// Custom includes
#include "primitives.h"

#endif

// External includes
#include <random>

// ----- STRUCTURES ------------------------------
/// A binary spaced partition leaf used to generate the dungeon's rooms.
///
/// Attributes
/// ----------
/// left - The left container of this leaf. If both values are -1, we have reached the end of the
/// branch.
/// right - The right container of this leaf. If both values are -1, we have reached the end of the
/// branch.
/// room - The rect object for representing the room inside this leaf.
/// split_vertical - Whether the leaf was split vertically or not.
struct Leaf {
    // Parameters
    Rect container;

    // Attributes
    Leaf *left, *right;
    Rect *room;
    bool split_vertical;

    /// Default constructor for a Leaf object. This should not be used.
    Leaf() {}

    /// Constructs a Leaf object.
    ///
    /// Parameters
    /// ----------
    /// container - The rect object for representing this leaf.
    ///
    /// Returns
    /// -------
    /// A Leaf object.
    Leaf(Rect container_val) {
        container = container_val;
        left = nullptr;
        right = nullptr;
        room = nullptr;
    }

    /// Split a container either horizontally or vertically.
    ///
    /// Parameters
    /// ----------
    /// grid - The 2D grid which represents the dungeon.
    /// random_generator - The random generator used to generate the bsp.
    /// min_container_size - The minimum size one side of a container can be.
    /// debug_game - Whether the game is in debug mode or not.
    ///
    /// Returns
    /// -------
    /// Whether the split was successful or not.
    bool split(std::vector<std::vector<int>> &grid, std::mt19937 &random_generator, bool debug_game) {
        // Check if this leaf is already split or not
        if (left && right) {
            return false;
        }

        // To determine the direction of split, we test if the width is 25% larger than the height,
        // if so we split vertically. However, if the height is 25% larger than the width, we split
        // horizontally. Otherwise, we split randomly
        std::uniform_int_distribution<> split_vertical_distribution(0, 1);
        bool split_vertical_val = split_vertical_distribution(random_generator);
        if ((container.width > container.height) && (((double) container.width / container.height) >= 1.25)) {
            split_vertical_val = true;
        } else if ((container.height > container.width) && (((double) container.height / container.width) >= 1.25)) {
            split_vertical_val = false;
        }

        // To determine the range of values that we could split on, we need to find out if the
        // container is too small. Once we've done that, we can use the x1, y1, x2 and y2
        // coordinates to specify the range of values
        int max_size = (split_vertical_val) ? container.width - MIN_CONTAINER_SIZE : container.height -
                                                                                     MIN_CONTAINER_SIZE;
        if (max_size <= MIN_CONTAINER_SIZE) {
            // Container too small to split
            return false;
        }

        // Create the split position. This ensures that there will be MIN_CONTAINER_SIZE on each
        // side
        std::uniform_int_distribution<> pos_distribution(MIN_CONTAINER_SIZE, max_size);
        int pos = pos_distribution(random_generator);

        // Split the container
        if (split_vertical_val) {
            // Split vertically making sure to adjust pos, so it can be within range of the actual
            // container
            pos += container.top_left.x;
            if (debug_game) {
                for (int y = container.top_left.y; y < container.bottom_right.y + 1; y++) {
                    grid[y][pos] = TileType::DebugWall;
                }
            }

            // Create the child leafs
            left = new Leaf{
                Rect{
                    Point{
                        container.top_left.x, container.top_left.y
                    },
                    Point{
                        pos - 1, container.bottom_right.y,
                    }
                }
            };
            right = new Leaf{
                Rect{
                    Point{
                        pos + 1, container.top_left.y,
                    },
                    Point{
                        container.bottom_right.x, container.bottom_right.y
                    }
                }
            };
        } else {
            // Split horizontally making sure to adjust pos, so it can be within range of the actual
            // container
            pos += container.top_left.y;
            if (debug_game) {
                for (int x = container.top_left.x; x < container.bottom_right.x + 1; x++) {
                    grid[pos][x] = TileType::DebugWall;
                }
            }

            // Create the child leafs
            left = new Leaf{
                Rect{
                    Point{
                        container.top_left.x, container.top_left.y
                    },
                    Point{
                        container.bottom_right.x, pos - 1,
                    }
                }
            };
            right = new Leaf{
                Rect{
                    Point{
                        container.top_left.x, pos + 1,
                    },
                    Point{
                        container.bottom_right.x, container.bottom_right.y
                    }
                }
            };
        }

        // Set the leaf's split direction
        split_vertical = split_vertical_val;

        // Successful split
        return true;
    }

    /// Create a random sized room inside a container.
    ///
    /// Parameters
    /// ----------
    /// grid - The 2D grid which represents the dungeon.
    /// random_generator - The random generator used to generate the bsp.
    ///
    /// Returns
    /// -------
    /// Whether the room creation was successful or not.
    bool create_room(std::vector<std::vector<int>> &grid, std::mt19937 &random_generator) {
        // Test if this container is already split or not. If it is, we do not want to create a room
        // inside it otherwise it will overwrite other rooms
        if (left && right) {
            return false;
        }

        // Pick a random width and height making sure it is at least min_room_size but doesn't
        // exceed the container
        std::uniform_int_distribution<> width_distribution(MIN_ROOM_SIZE, container.width);
        int width = width_distribution(random_generator);
        std::uniform_int_distribution<> height_distribution(MIN_ROOM_SIZE, container.height);
        int height = height_distribution(random_generator);

        // Use the width and height to find a suitable x and y position which can create the room
        std::uniform_int_distribution<> x_pos_distribution(container.top_left.x, container.bottom_right.x - width);
        int x_pos = x_pos_distribution(random_generator);
        std::uniform_int_distribution<> y_pos_distribution(container.top_left.y, container.bottom_right.y - height);
        int y_pos = y_pos_distribution(random_generator);

        // Create the room rect and test if its width to height ratio will make an oddly-shaped room
        Rect rect = Rect{
            Point{
                x_pos, y_pos
            },
            Point{
                x_pos + width - 1, y_pos + height - 1
            }
        };
        if ((((double) std::min(rect.width, rect.height)) / ((double) std::min(rect.width, rect.height))) <
            ROOM_RATIO) {
            return false;
        }

        // Width to height ratio is fine so place the rect in the 2D grid and store it
        rect.place_rect(grid);
        room = &rect;

        // Successful room creation
        return true;
    }
};
