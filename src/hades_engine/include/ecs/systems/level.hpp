// Ensure this file is only included once
#pragma once

// Local headers
#include "ecs/bases.hpp"

/// Allows a game object to have a level with experience.
struct PlayerLevel final : ComponentBase {
  /// The current level of the game object.
  int level{1};

  /// The current experience of the game object.
  double experience{0.0};
};
