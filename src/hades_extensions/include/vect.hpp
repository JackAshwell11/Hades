// Ensure this file is only included once
#pragma once

// External headers
#include <chipmunk/chipmunk.h>

/// The != operator.
inline auto operator!=(const cpVect &lhs, const cpVect &rhs) -> bool { return lhs.x != rhs.x || lhs.y != rhs.y; }
