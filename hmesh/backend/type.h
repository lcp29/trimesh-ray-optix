#pragma once

#include <cstdint>

namespace hmesh {

template <typename T, int N> struct vec { T p[N]; };

using vec2i = vec<int32_t, 2>;
using vec2f = vec<float, 2>;
using vec3i = vec<int32_t, 3>;
using vec3f = vec<float, 3>;

} // namespace hmesh
