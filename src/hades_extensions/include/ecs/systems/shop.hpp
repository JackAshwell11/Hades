// Ensure this file is only included once
#pragma once

// Std headers
#include <typeindex>

// Local headers
#include "ecs/bases.hpp"
#include "game_object.hpp"

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

  /// The virtual destructor.
  virtual ~ShopOffering() = default;

  /// The copy constructor.
  ShopOffering(const ShopOffering &) = default;

  /// The move constructor.
  ShopOffering(ShopOffering &&) = default;

  /// The copy assignment operator.
  auto operator=(const ShopOffering &) -> ShopOffering & = default;

  /// The move assignment operator.
  auto operator=(ShopOffering &&) -> ShopOffering & = default;

  /// Apply the offering to the buyer.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  /// @return true if the application was successful, false otherwise.
  virtual auto apply(const Registry *registry, GameObjectID buyer_id) const -> bool = 0;

  /// Get the cost of the offering.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @return The cost of the offering.
  [[nodiscard]] virtual auto get_cost(const Registry *registry, GameObjectID buyer_id) const -> double;

  /// The name of the offering.
  std::string name;

  /// The description of the offering.
  std::string description;

  /// The base cost of the offering.
  double base_cost;

  /// The cost multiplier of the offering.
  double cost_multiplier;
};

/// Represents an upgradable stat offering in the shop.
struct StatUpgradeOffering final : ShopOffering {
  /// Initialise the object.
  ///
  /// @param name - The name of the offering.
  /// @param description - The description of the offering.
  /// @param component_type - The type of the component to upgrade.
  /// @param base_cost - The base cost of the offering.
  /// @param cost_multiplier - The cost multiplier of the offering.
  /// @param base_value - The base value of the offering.
  /// @param value_multiplier - The value multiplier of the offering.
  StatUpgradeOffering(const std::string &name, const std::string &description, const std::type_index component_type,
                      const double base_cost, const double cost_multiplier, const double base_value,
                      const double value_multiplier)
      : ShopOffering(name, description, base_cost, cost_multiplier),
        component_type(component_type),
        base_value(base_value),
        value_multiplier(value_multiplier) {}

  /// Apply the offering to the buyer.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  /// @return true if the application was successful, false otherwise.
  auto apply(const Registry *registry, GameObjectID buyer_id) const -> bool override;

  /// Get the cost of the offering.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @return The cost of the offering.
  [[nodiscard]] auto get_cost(const Registry *registry, GameObjectID buyer_id) const -> double override;

  /// The type of the component to upgrade.
  std::type_index component_type;

  /// The base value of the offering.
  double base_value;

  /// The value multiplier of the offering.
  double value_multiplier;
};

/// Represents a one-time component unlock offering in the shop.
struct ComponentUnlockOffering final : ShopOffering {
  /// Initialise the object.
  ///
  /// @param name - The name of the offering.
  /// @param description - The description of the offering.
  /// @param cost - The cost of the offering.
  /// @param cost_multiplier - The cost multiplier of the offering.
  ComponentUnlockOffering(const std::string &name, const std::string &description, const double cost,
                          const double cost_multiplier)
      : ShopOffering(name, description, cost, cost_multiplier) {}

  /// Apply the offering to the buyer.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  /// @return true if the application was successful, false otherwise.
  auto apply(const Registry *registry, GameObjectID buyer_id) const -> bool override;
};

/// Represents a repeatable item offering in the shop.
struct ItemOffering final : ShopOffering {
  /// Initialise the object.
  ///
  /// @param name - The name of the offering.
  /// @param description - The description of the offering.
  /// @param base_cost - The base cost of the offering.
  /// @param cost_multiplier - The cost multiplier of the offering.
  ItemOffering(const std::string &name, const std::string &description, const double base_cost,
               const double cost_multiplier)
      : ShopOffering(name, description, base_cost, cost_multiplier) {}

  /// Apply the offering to the buyer.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  /// @param buyer_id - The ID of the buyer.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  /// @return true if the application was successful, false otherwise.
  auto apply(const Registry *registry, GameObjectID buyer_id) const -> bool override;
};

/// Allows a game object to have currency.
struct Money final : ComponentBase {
  /// The amount of money the game object has.
  int money;

  /// Serialise the component to a JSON object.
  ///
  /// @param json - The JSON object to serialise to.
  void to_file(nlohmann::json &json) const override;

  /// Deserialise the component from a JSON object.
  ///
  /// @param json - The JSON object to deserialise from.
  void from_file(const nlohmann::json &json) override;
};

/// Provides facilities to manage a shop system.
class ShopSystem final : public SystemBase {
 public:
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit ShopSystem(Registry *registry) : SystemBase(registry) {}

  /// Add offerings to the shop from a stream.
  ///
  /// @param stream - The input stream containing the JSON data for the shop offerings.
  /// @param player_id - The game object ID of the player.
  /// @throws std::runtime_error if there was an error parsing the JSON file or the offering type is unknown.
  void add_offerings(std::istream &stream, GameObjectID player_id);

  /// Get an offering by its index.
  ///
  /// @param offering_index - The index of the offering.
  [[nodiscard]] auto get_offering(int offering_index) const -> const ShopOffering *;

  /// Get the cost of an offering.
  ///
  /// @param offering_index - The index of the offering.
  /// @param buyer_id - The ID of the buyer.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  /// @return The cost of the offering.
  [[nodiscard]] auto get_offering_cost(int offering_index, GameObjectID buyer_id) const -> int;

  /// Purchase an offering from the shop for a buyer.
  ///
  /// @param buyer_id - The ID of the buyer.
  /// @param offering_index - The index of the offering to purchase.
  /// @throws RegistryError - If the game object does not exist or does not have the required components.
  /// @return Whether the purchase was successful or not.
  [[nodiscard]] auto purchase(GameObjectID buyer_id, int offering_index) const -> bool;

 private:
  /// The offerings available in the shop.
  std::vector<std::unique_ptr<ShopOffering>> offerings_;
};
