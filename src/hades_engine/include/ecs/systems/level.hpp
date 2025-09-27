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

  /// Serialise the component to a JSON object.
  ///
  /// @param json - The JSON object to serialise to.
  void to_file(nlohmann::json& json) const override;

  /// Deserialise the component from a JSON object.
  ///
  /// @param json - The JSON object to deserialise from.
  void from_file(const nlohmann::json& json) override;
};
