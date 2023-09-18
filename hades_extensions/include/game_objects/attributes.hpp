// Ensure this file is only included once
#pragma once

#include <optional>
#include "registry.hpp"

// TODO: Could look at mixins
//  (https://stackoverflow.com/questions/18773367/what-are-mixins-as-a-concept)
//  along with redesigning GameObjectAtt*ributeBase to improve the design

// TODO: Could always talk to chatgpt about structuring the components

// TODO: Thinking about damagesystem, statuseffectsystem, instanteffectsystem,
//  upgradesystem, using mixins

// ----- STRUCTURES ------------------------------




class GameObjectAttributeBase : public ComponentBase {
 public:
  int level_limit;
  double max_value;
  int current_level = 0;

  GameObjectAttributeBase(double initial_value, int level_limit)
      : value_(initial_value),
        max_value(has_maximum() ? initial_value : std::numeric_limits<double>::infinity()),
        level_limit(level_limit) {}

  [[nodiscard]] virtual bool has_instant_effect() const { return true; }

  [[nodiscard]] virtual bool has_maximum() const { return true; }

  [[nodiscard]] virtual bool has_status_effect() const { return true; }

  [[nodiscard]] virtual bool is_upgradable() const { return true; }

  [[nodiscard]] inline double value() const {
    return value_;
  }

  inline void value(double new_value) {
    value_ = std::max(std::min(new_value, max_value), 0.0);
  }

 private:
  double value_;
};
