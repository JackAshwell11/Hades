// Std includes
#include <algorithm>

// Custom includes
#include "game_objects/components.hpp"

// ----- STRUCTURES ------------------------------
float GameObjectAttributeBase::value() const {
  return value_;
}

void GameObjectAttributeBase::value(float new_value) {
  value_ = std::max(std::min(new_value, max_value), 0.0f);
}
