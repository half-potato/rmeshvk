#define_import_path rmesh::intersect

// Face (a, b, c, opposite_vertex) — opposite used to flip normal inward.
// Matches TET_FACES in Rust camera module.
const FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);
