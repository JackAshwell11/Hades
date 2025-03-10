// Ensure this file is only included once
#pragma once

// Local headers
#include "ecs/registry.hpp"

/// Allows a game object to interact with the physics system.
struct KinematicComponent final : ComponentBase {
  /// The Chipmunk2D body of the game object.
  ChipmunkHandle<cpBody, cpBodyFree> body;

  /// The Chipmunk2D shape of the game object.
  ChipmunkHandle<cpShape, cpShapeFree> shape;

  /// The rotation angle of the game object in radians.
  double rotation{0};

  /// Whether the game object has been collected by the player or not.
  bool collected{false};

  /// Initialise the object.
  ///
  /// @param is_static - Whether the body is static or not.
  explicit KinematicComponent(const bool is_static = false)
      : body(is_static ? cpBodyNewStatic() : cpBodyNewKinematic()),
        shape(cpBoxShapeNew(*body, SPRITE_SIZE, SPRITE_SIZE, 0.0)) {
    cpShapeSetSensor(*shape, !is_static);
  }

  /// Initialise the object.
  ///
  /// @param vertices - The vertices of the shape.
  explicit KinematicComponent(const std::vector<cpVect> &vertices)
      : body(cpBodyNew(1, std::numeric_limits<cpFloat>::infinity())),
        shape(cpPolyShapeNew(*body, static_cast<int>(vertices.size()), vertices.data(), cpTransformIdentity, 0.0)) {}
};

/// Provides facilities to manipulate a game object's physics.
struct PhysicsSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit PhysicsSystem(Registry *registry) : SystemBase(registry) {}

  /// Process update logic for the physics system.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(const double delta_time) const override { cpSpaceStep(get_registry()->get_space(), delta_time); }

  /// Add a force to a game object.
  ///
  /// @param game_object_id - The ID of the game object.
  /// @param force - The force to add.
  /// @throws RegistryError if the game object does not exist or does not have a kinematic component.
  void add_force(GameObjectID game_object_id, const cpVect &force) const;

  /// Add a bullet to the physics engine.
  ///
  /// @param bullet - The bullet's position and velocity.
  /// @param damage - The damage the bullet inflicts.
  /// @param source_type - The game object type that fired the bullet.
  void add_bullet(const std::pair<cpVect, cpVect> &bullet, double damage, GameObjectType source_type) const;

  /// Get the nearest item to a game object.
  ///
  /// @param game_object_id - The ID of the game object to get the nearest item to.
  /// @throws RegistryError if the game object does not exist or does not have a kinematic component.
  /// @return The ID of the nearest item.
  [[nodiscard]] auto get_nearest_item(GameObjectID game_object_id) const -> GameObjectID;
};
