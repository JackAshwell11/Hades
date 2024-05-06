// Ensure this file is only included once
#pragma once

// Local headers
#include "game_objects/registry.hpp"
#include "game_objects/stats.hpp"

// ----- COMPONENTS ------------------------------
/// Allows a game object to interact with the physics system.
struct KinematicComponent final : ComponentBase {
  /// The Chipmunk2D body of the game object.
  ChipmunkHandle<cpBody, cpBodyFree> body;

  /// The Chipmunk2D shape of the game object.
  ChipmunkHandle<cpShape, cpShapeFree> shape;

  /// The rotation angle of the game object in degrees.
  double rotation{0};

  /// Initialise the object.
  ///
  /// @param is_static - Whether the body is static or not.
  explicit KinematicComponent(const bool is_static = false)
      : body(is_static ? cpBodyNewStatic() : cpBodyNewKinematic()),
        shape(cpBoxShapeNew(*body, SPRITE_SIZE, SPRITE_SIZE, 0.0)) {}

  /// Initialise the object.
  ///
  /// @param vertices - The vertices of the shape.
  explicit KinematicComponent(const std::vector<cpVect> &&vertices)
      : body(cpBodyNew(1, std::numeric_limits<cpFloat>::infinity())),
        shape(cpPolyShapeNew(*body, static_cast<int>(vertices.size()), vertices.data(), cpTransformIdentity, 0.0)) {}
};

// ----- SYSTEMS ------------------------------
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
  void add_force(const GameObjectID game_object_id, const cpVect &force) const {
    cpBodyApplyForceAtLocalPoint(
        *get_registry()->get_component<KinematicComponent>(game_object_id)->body,
        cpvnormalize(force) * get_registry()->get_component<MovementForce>(game_object_id)->get_value(), cpvzero);
  }

  /// Add a bullet to the physics engine.
  ///
  /// @param bullet - The bullet's position and velocity.
  /// @return The game object ID for the bullet.
  [[nodiscard]] auto add_bullet(const std::pair<cpVect, cpVect> &bullet) const -> GameObjectID {
    const auto bullet_id = get_registry()->create_game_object(GameObjectType::Bullet, get<0>(bullet),
                                                              {std::make_shared<KinematicComponent>()});
    cpBodySetVelocity(*get_registry()->get_component<KinematicComponent>(bullet_id)->body, get<1>(bullet));
    return bullet_id;
  }
};
