/// Creates a binary space partition used for generating the rooms.
use crate::generation::constants::{TileType, MIN_CONTAINER_SIZE, MIN_ROOM_SIZE, ROOM_RATIO};
use crate::generation::primitives::{Point, Rect};
use ndarray::{s, Array2};
use rand::rngs::StdRng;
use rand::Rng;
use std::cmp::{max, min};

/// A binary spaced partition leaf used to generate the dungeon's rooms.
///
/// # Attributes
/// * `left` - The left container of this leaf. If this is None, we have reached the end of the
/// branch.
/// * `right` - The right container of this leaf. If this is None, we have reached the end of the
/// branch.
/// * `room` - The rect object for representing the room inside this leaf.
/// * `split_vertical` - Whether the leaf was split vertically or not. By default, this is None
/// (not split).
#[derive(Clone)]
pub struct Leaf {
    // Parameters
    container: Rect,

    // Attributes
    pub left: Option<Box<Leaf>>,
    pub right: Option<Box<Leaf>>,
    pub room: Option<Rect>,
    split_vertical: Option<bool>,
}

impl Leaf {
    /// Constructs a Leaf object.
    ///
    /// # Parameters
    /// * `container` - The rect object for representing this leaf.
    ///
    /// # Returns
    /// A Leaf object.
    pub fn new(container: Rect) -> Leaf {
        Leaf {
            container,
            left: None,
            right: None,
            room: None,
            split_vertical: None,
        }
    }

    /// Split a container either horizontally or vertically.
    ///
    /// # Parameters
    /// * `grid` - The 2D grid which represents the dungeon.
    /// * `random_generator` - The random generator used to generate the bsp.
    /// * `min_container_size` - The minimum size one side of a container can be.
    /// * `debug_game` - Whether the game is in debug mode or not.
    ///
    /// # Returns
    /// Whether the split was successful or not.
    pub fn split(
        &mut self,
        grid: &mut Array2<TileType>,
        random_generator: &mut StdRng,
        debug_game: bool,
    ) -> bool {
        // Check if this leaf is already split or not
        if self.left.is_some() && self.right.is_some() {
            return false;
        }

        // To determine the direction of split, we test if the width is 25% larger than the height,
        // if so we split vertically. However, if the height is 25% larger than the width, we split
        // horizontally. Otherwise, we split randomly
        let split_vertical: bool = if (self.container.width > self.container.height)
            && ((self.container.width / self.container.height) as f64 >= 1.25)
        {
            true
        } else if (self.container.height > self.container.width)
            && ((self.container.height / self.container.width) as f64 >= 1.25)
        {
            false
        } else {
            random_generator.gen::<bool>()
        };

        // To determine the range of values that we could split on, we need to find out if the
        // container is too small. Once we've done that, we can use the x1, y1, x2 and y2
        // coordinates to specify the range of values
        let max_size: i32 = if split_vertical {
            self.container.width - MIN_CONTAINER_SIZE
        } else {
            self.container.height - MIN_CONTAINER_SIZE
        };
        if max_size <= MIN_CONTAINER_SIZE {
            // Container too small to split
            return false;
        }

        // Create the split position. This ensures that there will be MIN_CONTAINER_SIZE on each
        // side
        let mut pos: i32 = random_generator.gen_range(MIN_CONTAINER_SIZE..max_size + 1);

        // Split the container
        if split_vertical {
            // Split vertically making sure to adjust pos, so it can be within range of the actual
            // container
            pos += self.container.top_left.x;
            if debug_game {
                grid.slice_mut(s![
                    self.container.top_left.y..self.container.bottom_right.y + 1,
                    pos
                ])
                .fill(TileType::DebugWall);
            }

            // Create the child leafs
            self.left = Option::from(Box::new(Leaf::new(Rect::new(
                Point::new(self.container.top_left.x, self.container.top_left.y),
                Point::new(pos - 1, self.container.bottom_right.y),
            ))));
            self.right = Option::from(Box::new(Leaf::new(Rect::new(
                Point::new(pos + 1, self.container.top_left.y),
                Point::new(self.container.bottom_right.x, self.container.bottom_right.y),
            ))));
        } else {
            // Split horizontally making sure to adjust pos, so it can be within range of the actual
            // container
            pos += self.container.top_left.y;
            if debug_game {
                grid.slice_mut(s![
                    pos,
                    self.container.top_left.x..self.container.bottom_right.x + 1
                ])
                .fill(TileType::DebugWall);
            }

            // Create the child leafs
            self.left = Option::from(Box::new(Leaf::new(Rect::new(
                Point::new(self.container.top_left.x, self.container.top_left.y),
                Point::new(self.container.bottom_right.x, pos - 1),
            ))));
            self.right = Option::from(Box::new(Leaf::new(Rect::new(
                Point::new(self.container.top_left.x, pos + 1),
                Point::new(self.container.bottom_right.x, self.container.bottom_right.y),
            ))));
        }

        // Set the leaf's split direction
        self.split_vertical = Option::from(split_vertical);

        // Successful split
        return true;
    }

    /// Create a random sized room inside a container.
    ///
    /// # Parameters
    /// * `grid` - The 2D grid which represents the dungeon.
    /// * `random_generator` - The random generator used to generate the bsp.
    ///
    /// # Returns
    /// Whether the room creation was successful or not.
    pub fn create_room(
        &mut self,
        grid: &mut Array2<TileType>,
        random_generator: &mut StdRng,
    ) -> bool {
        // Test if this container is already split or not. If it is, we do not want to create a room
        // inside it otherwise it will overwrite other rooms
        if self.left.is_some() && self.right.is_some() {
            return false;
        }

        // Pick a random width and height making sure it is at least min_room_size but doesn't
        // exceed the container
        let width: i32 = random_generator.gen_range(MIN_ROOM_SIZE..self.container.width + 1);
        let height: i32 = random_generator.gen_range(MIN_ROOM_SIZE..self.container.height + 1);

        // Use the width and height to find a suitable x and y position which can create the room
        let x_pos: i32 = random_generator
            .gen_range(self.container.top_left.x..self.container.bottom_right.x - width + 1);
        let y_pos: i32 = random_generator
            .gen_range(self.container.top_left.y..self.container.bottom_right.y - height + 1);

        // Create the room rect and test if its width to height ratio will make an oddly-shaped room
        let rect: Rect = Rect::new(
            Point::new(x_pos, y_pos),
            Point::new(x_pos + width - 1, y_pos + height - 1),
        );
        if ((min(rect.width, rect.height) as f32) / (max(rect.width, rect.height) as f32))
            < ROOM_RATIO
        {
            return false;
        }

        // Width to height ratio is fine so place the rect in the 2D grid and store it
        rect.place_rect(grid);
        self.room = Option::from(rect);

        // Successful room creation
        return true;
    }
}
