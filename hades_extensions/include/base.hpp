// Ensure this file is only included once
#pragma once

// Std includes
#include <cmath>

// ----- HASHES ------------------------------
/// Allows multiple hashes to be combined for a struct
///
/// @param seed - The seed for initialising the hasher.
/// @param v - The value to hash.
template<typename T>
inline void hash_combine(size_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// TODO: Look over generation/ (header, source and test) to make sure it all
//  conforms to standards set by game_objects/

// TODO: Look over all includes (need to decide if each file includes
//  everything (even duplicates) or only what it needs (takes from other
//  includes)). Leaning towards each file only including what it needs and not
//  including what has already been included

// TODO: See if const, consteval, constexpr, inline and references can be used
//  more

// TODO: Move all fixtures to local files (for independence)

// TODO: Rename all Fixtures to Fixture

// TODO: Switch to docstrings for explaining tests
