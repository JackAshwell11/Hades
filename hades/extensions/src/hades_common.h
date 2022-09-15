#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits>
#include <unordered_map>
#include <vector>


struct IntPair {
    /* Represents a namedtuple describing a pair of integers */
    int x, y;

    inline bool operator==(const IntPair pair) const {
        // If two hashes are the same, we need to check if the two pairs are the same
        return x == pair.x && y == pair.y;
    }
};


/* Represents the north, south, east and west directions on a compass */
std::vector<IntPair> CARDINAL_OFFSETS = {
        {0, -1},
        {-1, 0},
        {1, 0},
        {0, 1},
};


/* Represents the north, south, east, west, north-east, north-west, south-east and
south-west directions on a compass */
std::vector<IntPair> INTERCARDINAL_OFFSETS = {
        {-1, -1},
        {0, -1},
        {1, -1},
        {-1, 0},
        {1, 0},
        {-1, 1},
        {0, 1},
        {1, 1},
};


template <class T>
inline void hash_combine(size_t& seed, const T& v) {
    /* Allows multiple hashes to be combined for a struct */
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}


template<>
struct std::hash<IntPair> {
    /* Allows the pair struct to be hashed in a map */
    size_t operator()(const IntPair &pair) const {
        size_t res = 0;
        hash_combine(res, pair.x);
        hash_combine(res, pair.y);
        return res;
    }
};


std::vector<IntPair> grid_bfs(IntPair target, int height, int width, std::vector<IntPair> offsets = CARDINAL_OFFSETS) {
    /* Gets a target's neighbours in a grid */
    std::vector<IntPair> result;
    for (int i = 0; i < offsets.size(); i++) {
        int x = target.x + offsets[i].x;
        int y = target.y + offsets[i].y;
        if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
            result.push_back({x, y});
        }
    }
    return result;
}
