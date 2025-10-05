// Ensure this file is only included once
#pragma once

// Std headers
#include <memory>
#include <string>
#include <vector>

// Local headers
#include "ecs/bases.hpp"
#include "game_object.hpp"

/// Allows a game object to have currency.
struct Money final : ComponentBase {
  /// The amount of money the game object has.
  int money;
};

/// Represents an offering in the shop.
struct ShopOffering {
  /// Initialise the object.
  ///
  /// @param name - The name of the offering.
  /// @param description - The description of the offering.
  /// @param base_cost - The base cost of the offering.
  /// @param cost_multiplier - The cost multiplier of the offering.
  ShopOffering(std::string name, std::string description, const double base_cost, const double cost_multiplier)
      : name(std::move(name)),
        description(std::move(description)),
        base_cost(base_cost),
        cost_multiplier(cost_multiplier) {}

  /// Apply the offering to the buyer.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  /// @return true if the application was successful, false otherwise.
  auto apply(const Registry* registry, GameObjectID buyer_id) const -> bool;

  /// Get the cost of the offering.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @return The cost of the offering.
  [[nodiscard]] auto get_cost(const Registry* registry, GameObjectID buyer_id) const -> int;

  /// The name of the offering.
  std::string name;

  /// The description of the offering.
  std::string description;

  /// The base cost of the offering.
  double base_cost;

  /// The cost multiplier of the offering.
  double cost_multiplier;
};

/// Provides facilities to manage a shop system.
class ShopSystem final : public SystemBase {
 public:
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit ShopSystem(Registry* registry) : SystemBase(registry) {}

  /// Add offerings to the shop from a stream.
  ///
  /// @param stream - The input stream containing the JSON data for the shop offerings.
  /// @param player_id - The game object ID of the player.
  /// @throws std::runtime_error if there was an error parsing the JSON file or the offering type is unknown.
  void add_offerings(std::istream& stream, GameObjectID player_id);

  /// Get an offering by its index.
  ///
  /// @param offering_index - The index of the offering.
  /// @throws std::out_of_range if the index is out of bounds.
  /// @return The offering at the specified index, or nullptr if the index is out of bounds.
  [[nodiscard]] auto get_offering(int offering_index) const -> const ShopOffering&;

  /// Purchase an offering from the shop for a buyer.
  ///
  /// @param buyer_id - The ID of the buyer.
  /// @param offering_index - The index of the offering to purchase.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  /// @return Whether the purchase was successful or not.
  [[nodiscard]] auto purchase(GameObjectID buyer_id, int offering_index) const -> bool;

 private:
  /// The offerings available in the shop.
  std::vector<ShopOffering> offerings_;
};
