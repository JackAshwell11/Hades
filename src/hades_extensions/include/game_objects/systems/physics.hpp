// Ensure this file is only included once
#pragma once

// Local headers
#include "game_objects/registry.hpp"
#include "game_objects/stats.hpp"

#include <iostream>

// ----- COMPONENTS ------------------------------
/// Allows a game object to interact with the physics system.
struct KinematicComponent final : ComponentBase {
  /// The Chipmunk2D body of the game object.
  ChipmunkHandle<cpBody, cpBodyFree> body{cpBodyNew(1, INFINITY)};

  /// The Chipmunk2D shape of the game object.
  ChipmunkHandle<cpShape, cpShapeFree> shape;

  /// Initialise the object.
  ///
  /// @param vertices - The vertices of the shape.
  /// @param physics_type - The type of physics object.
  explicit KinematicComponent(const std::vector<cpVect> &&/*vertices*/, const PhysicsType physics_type)
    //  : shape(cpPolyShapeNew(*body, static_cast<int>(vertices.size()), vertices.data(), cpTransformIdentity, 0.0)) {
      : shape(cpBoxShapeNew(*body, SPRITE_SIZE, SPRITE_SIZE, 0.0)) {
    cpShapeSetCollisionType(*shape, static_cast<cpCollisionType>(physics_type));
  }
};

// ----- SYSTEMS ------------------------------
/// Provides facilities to manipulate a game object's physics.
struct PhysicsSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit PhysicsSystem(const Registry *registry) : SystemBase(registry) {}

  /// Process update logic for the physics system.
  ///
  /// @param delta_time - The time interval since the last time the function was called.
  void update(const double delta_time) const override {
    cpSpaceStep(get_registry()->get_space(), delta_time);
  }

  /// Add a force to a game object.
  ///
  /// @param game_object_id - The ID of the game object.
  /// @param force - The force to add.
  /// @throws RegistryError if the game object does not exist or does not have the target component.
  void add_force(const GameObjectID game_object_id, const cpVect &force) const {
    cpBodyApplyForceAtLocalPoint(
        *get_registry()->get_component<KinematicComponent>(game_object_id)->body,
        cpvnormalize(force) * get_registry()->get_component<MovementForce>(game_object_id)->get_value(), cpvzero);
  }
};
