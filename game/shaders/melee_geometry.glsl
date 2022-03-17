#version 330
// This is how we access Arcade's global projection matrix
uniform Projection {
    uniform mat4 matrix;
} proj;
// The position we measure distance from
uniform vec2 origin;
// The position we measure distance from accounting for the camera scroll
uniform vec2 origin_relative;
// The maximum distance
uniform float maxDistance;
// Sampler for reading wall data
uniform sampler2D walls;
// These configure the geometry shader to process a points
// and also emit single points of data
// in the spritelist.
layout (points) in;
layout (points, max_vertices = 1) out;
// The position input from vertex shader.
// It's an array because geo shader can take more than one input
in vec2 v_position[];
// This shader outputs all the sprite indices it selected.
// NOTE: We use floats here for compatibility since integers are always working
out float spriteIndex;
// Helper function converting screen coordinates to texture coordinates.
// Texture coordinates are normalized (0.0 -> 1.0) were 0,0 is in the
vec2 screen2texcoord(vec2 pos) {
    return vec2(pos / textureSize(walls, 0).xy);
}
void main() {
    // Only emit a line between the sprite and origin when within the distance
    if (distance(v_position[0], origin) > maxDistance) return;
    // Read samples from the wall texture in a line looking for obstacles
    // We simply make a vector between the origina and the sprite location
    // and trace pixels in this path with a reasonable step.
    int steps = int(maxDistance / 2);
    vec2 dir = v_position[0] - origin;
    for (int i = 0; i < steps; i++) {
        // Read pixels along the vector
        vec2 pos = origin_relative + dir * (float(i) / float(steps));
        vec4 color = texture(walls, screen2texcoord(pos));
        // If we find non-zero pixel data we have obstacles in our path!
        if (color != vec4(0.0)) return;
    }
    // We simply return the primitive index.
    // This is a built in counter in geometry shaders
    // started at 0 incrementing by 1 for every invocation.
    // It should always match the spritelist index.
    spriteIndex = float(gl_PrimitiveIDIn);
    EmitVertex();
    EndPrimitive();
}
