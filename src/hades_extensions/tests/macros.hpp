// Ensure this file is only included once
#pragma once

// External headers
#include <gtest/gtest.h>

// ----- MACROS ------------------------------
/// Assert that a statement throws an exception of a given type with a given message.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ASSERT_THROW_MESSAGE(statement, exception_type, expected_msg) \
  try {                                                               \
    static_cast<void>(statement);                                     \
    ADD_FAILURE() << "Expected exception of type " #exception_type;   \
  } catch (const exception_type &e) {                                 \
    ASSERT_STREQ(e.what(), expected_msg);                             \
    SUCCEED();                                                        \
  } catch (...) {                                                     \
    ADD_FAILURE() << "Expected exception of type " #exception_type;   \
  }
