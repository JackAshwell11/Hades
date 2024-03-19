// Ensure this file is only included once
#pragma once

// Local headers
#include "game_objects/registry.hpp"

// ----- COMPONENTS ------------------------------
/// Allows a game object to interact with the physics system.
struct KinematicComponent final : ComponentBase {
  /// The Chipmunk2D body of the game object.
  ChipmunkHandle<cpBody, cpBodyFree> body{cpBodyNewKinematic()};

  /// The Chipmunk2D shape of the game object.
  ChipmunkHandle<cpShape, cpShapeFree> shape;

  /// Initialise the object.
  ///
  /// @param vertices - The vertices of the shape.
  explicit KinematicComponent(const std::vector<cpVect> &&vertices)
      : shape(cpPolyShapeNew(*body, static_cast<int>(vertices.size()), vertices.data(), cpTransformIdentity, 0.0)) {}
};

// ----- SYSTEMS ------------------------------
/// Provides facilities to manipulate
struct PhysicsSystem final : SystemBase {
  /// Initialise the object.
  ///
  /// @param registry - The registry that manages the game objects, components, and systems.
  explicit PhysicsSystem(const Registry *registry) : SystemBase(registry) {}
};
