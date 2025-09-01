// Ensure this file is only included once
#pragma once

// Ensure Chipmunk2D uses the correct types
#define CP_BOOL_TYPE bool
#define cpTrue true
#define cpFalse false

// Std headers
#include <memory>

// External headers
#include <chipmunk/chipmunk.h>

/// The scale factor for the sprite.
constexpr double SPRITE_SCALE{0.5};

/// The pixel size of the sprite.
constexpr double SPRITE_SIZE{128 * SPRITE_SCALE};

/// The != operator.
inline auto operator!=(const cpVect& lhs, const cpVect& rhs) -> bool { return !cpveql(lhs, rhs); }

/// The + operator.
inline auto operator+(const cpVect& lhs, const float val) -> cpVect { return cpvadd(lhs, {val, val}); }

/// The += operator.
inline auto operator+=(cpVect& lhs, const cpVect& rhs) -> cpVect {
  lhs = cpvadd(lhs, rhs);
  return lhs;
}

/// Allows for the RAII management of a Chipmunk2D object.
///
/// @tparam T - The type of Chipmunk2D object to manage.
/// @tparam Destructor - The destructor function for the Chipmunk2D object.
template <typename T, void (*Destructor)(T*)>
class ChipmunkHandle {
 public:
  /// Initialise the object.
  ///
  /// @param obj - The Chipmunk2D object.
  explicit ChipmunkHandle(T* obj) : obj_(obj, Destructor) {}

  /// The destructor.
  ~ChipmunkHandle() = default;

  /// The copy constructor.
  ChipmunkHandle(const ChipmunkHandle&) = delete;

  /// The copy assignment operator.
  auto operator=(const ChipmunkHandle&) -> ChipmunkHandle& = delete;

  /// The move constructor.
  ChipmunkHandle(ChipmunkHandle&& other) noexcept : obj_(std::move(other.obj_)) {}

  /// The move assignment operator.
  auto operator=(ChipmunkHandle&& other) noexcept -> ChipmunkHandle& {
    if (this != &other) {
      obj_ = std::move(other.obj_);
    }
    return *this;
  }

  /// The dereference operator.
  auto operator*() const -> T* { return obj_.get(); }

 private:
  /// The Chipmunk2D object.
  std::unique_ptr<T, void (*)(T*)> obj_;
};
