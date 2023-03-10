/// Stores various constants related to the dungeon generation.
use pyo3::prelude::*;
use std::cmp::min;

/// Stores the different types of tiles in the game map.
#[pyclass]
#[derive(PartialEq, Eq, Hash, Clone)]
pub enum TileType {
    DebugWall,
    Empty,
    Floor,
    Wall,
    Obstacle,
    Player,
    HealthPotion,
    ArmourPotion,
    HealthBoostPotion,
    ArmourBoostPotion,
    SpeedBoostPotion,
    FireRateBoostPotion,
}

/// Stores a map generation constant which can be calculated.
///
/// # Parameters
/// * `base_value` - The base value for the exponential calculation.
/// * `increase` - The percentage increase for the constant.
/// * `max_value` - The max value for the exponential calculation.
pub struct MapGenerationConstant {
    base_value: i32,
    max_value: i32,
    increase: f32,
}

impl MapGenerationConstant {
    /// Generate a value based on the exponential equation.
    ///
    /// # Parameters
    /// * `level` - The game level to generate a value for.
    ///
    /// # Returns
    /// The generated valued.
    #[inline]
    pub fn generate_value(&self, level: i32) -> i32 {
        min(
            (self.base_value as f32 * self.increase.powi(level)).round() as i32,
            self.max_value,
        )
    }
}

/// Stores the map generation constants
///
/// # Parameters
/// * `width` - The width of the 2D grid.
/// * `height` - The height of the 2D grid.
/// * `split_iteration` - The amount of splits to perform.
/// * `obstacle_count` - The amount of obstacles to place in the 2D grid.
/// * `item_count` - The amount of items to place in the 2D grid.
pub struct MapGenerationConstants {
    pub width: MapGenerationConstant,
    pub height: MapGenerationConstant,
    pub split_iteration: MapGenerationConstant,
    pub obstacle_count: MapGenerationConstant,
    pub item_count: MapGenerationConstant,
}

// Defines the constants for the map generation
pub const MAP_GENERATION_CONSTANTS: MapGenerationConstants = MapGenerationConstants {
    width: MapGenerationConstant {
        base_value: 30,
        max_value: 150,
        increase: 1.2,
    },
    height: MapGenerationConstant {
        base_value: 20,
        max_value: 100,
        increase: 1.2,
    },
    split_iteration: MapGenerationConstant {
        base_value: 5,
        max_value: 25,
        increase: 1.5,
    },
    obstacle_count: MapGenerationConstant {
        base_value: 20,
        max_value: 200,
        increase: 1.3,
    },
    item_count: MapGenerationConstant {
        base_value: 5,
        max_value: 30,
        increase: 1.1,
    },
};

// Defines the probabilities for each item
pub const ITEM_PROBABILITIES: [(TileType, f32); 6] = [
    (TileType::HealthPotion, 0.3),
    (TileType::ArmourPotion, 0.3),
    (TileType::HealthBoostPotion, 0.2),
    (TileType::ArmourBoostPotion, 0.1),
    (TileType::SpeedBoostPotion, 0.05),
    (TileType::FireRateBoostPotion, 0.05),
];

// Defines constants for the binary space partition
pub const CONTAINER_RATIO: f64 = 1.25;
pub const MIN_CONTAINER_SIZE: i32 = 5;
pub const MIN_ROOM_SIZE: i32 = 4;
pub const ROOM_RATIO: f32 = 0.625;

// Defines constants for hallway and entity generation
pub const REPLACEABLE_TILES: [TileType; 3] =
    [TileType::Empty, TileType::Obstacle, TileType::DebugWall];
pub const HALLWAY_SIZE: i32 = 5;
