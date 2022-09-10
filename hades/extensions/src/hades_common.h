template <class T>
inline void hash_combine(size_t& seed, const T& v) {
    /* Allows multiple hashes to be combined for a struct */
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}


struct Pair {
    /* Represents a namedtuple describing a pair of integers */
    int x, y;

    inline bool operator==(const Pair pair) const {
        // If two hashes are the same, we need to check if the two pairs are the same
        return x == pair.x && y == pair.y;
    }
};


std::vector<Pair> CARDINAL_OFFSETS = {
        {0, -1},
        {-1, 0},
        {1, 0},
        {0, 1},
};

std::vector<Pair> INTERCARDINAL_OFFSETS = {
        {-1, -1},
        {0, -1},
        {1, -1},
        {-1, 0},
        {1, 0},
        {-1, 1},
        {0, 1},
        {1, 1},
};


template<>
struct std::hash<Pair> {
    /* Allows the pair struct to be hashed in a map */
    size_t operator()(const Pair& pnt) const {
        size_t res = 0;
        hash_combine(res, pnt.x);
        hash_combine(res, pnt.y);
        return res;
    }
};


std::vector<Pair> grid_bfs(Pair target, int height, int width, std::vector<Pair> offsets = CARDINAL_OFFSETS) {
    /* Gets a target's neighbours in a grid */
    std::vector<Pair> result;
    for (int i = 0; i < offsets.size(); i++) {
        int x = target.x + offsets[i].x;
        int y = target.y + offsets[i].y;
        if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
            result.push_back({x, y});
        }
    }
    return result;
}
