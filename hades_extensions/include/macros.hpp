// Ensure this file is only included once
#pragma once

// External includes
#include "gtest/gtest.h"

// ----- MACROS ------------------------------
/// Assert that a statement throws an exception of a given type with a given message.
#define ASSERT_THROW_MESSAGE(statement, exception_type, expected_msg) \
    try { \
      statement; \
      ADD_FAILURE() << "Expected exception of type " #exception_type; \
    } catch (const exception_type &e) { \
      ASSERT_STREQ(e.what(), expected_msg); \
      SUCCEED(); \
    } catch (...) { \
      ADD_FAILURE() << "Expected exception of type " #exception_type; \
    }

// TODO: Maybe move test structs, classes and fixtures to separate include
//  directory

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
