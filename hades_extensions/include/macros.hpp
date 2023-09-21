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
//  conforms to standards set by game_objects/. This includes const, consteval,
//  constexpr, inline, references, docs, includes, local fixtures, etc
