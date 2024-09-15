// Related header
#include "ecs/systems/physics.hpp"

// Local headers
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/sprite.hpp"

void PhysicsSystem::add_force(const GameObjectID game_object_id, const cpVect &force) const {
  cpBodyApplyForceAtLocalPoint(
      *get_registry()->get_component<KinematicComponent>(game_object_id)->body,
      cpvnormalize(force) * get_registry()->get_component<MovementForce>(game_object_id)->get_value(), cpvzero);
}

auto PhysicsSystem::add_bullet(const std::pair<cpVect, cpVect> &bullet, const double damage) const -> GameObjectID {
  const auto bullet_id{
      get_registry()->create_game_object(GameObjectType::Bullet, get<0>(bullet),
                                         {std::make_shared<Damage>(damage, -1), std::make_shared<KinematicComponent>(),
                                          std::make_shared<PythonSprite>()})};
  cpBodySetVelocity(*get_registry()->get_component<KinematicComponent>(bullet_id)->body, get<1>(bullet));
  return bullet_id;
}
