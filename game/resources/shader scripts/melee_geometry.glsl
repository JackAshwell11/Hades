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
uniform float max_distance;
// How many times we should check along the ray
uniform int resolution;
// The origin's direction
uniform float direction;
// Half of the angle range that the target can be within
uniform int half_angle_range;
// Sampler for reading wall data
uniform sampler2D walls;
// These configure the geometry shader to process a point and also emit single points of data in the spritelist
layout (points) in;
layout (points, max_vertices = 1) out;
// The position input from vertex shader. It's an array because the geometry shader can take more than one input
in vec2 v_position[];
// This shader outputs all the sprite indices it selected. NOTE: We use floats here for compatibility since integers are
// always working
out float sprite_index;
// Helper function converting screen coordinates to texture coordinates. Texture coordinates are normalized (0.0 -> 1.0)
vec2 screen_to_texcoord(vec2 pos) {
    return vec2(pos / textureSize(walls, 0).xy);
}
// Main geometry shader function
void main() {
    // Only emit a line between the sprite and the origin when the sprite is within the distance
    if (distance(v_position[0], origin) > max_distance) return;
    // Get the angle between the origin and the target. This needs to be in the format arcade uses
    float x_diff = v_position[0].x - origin.x;
    float y_diff = v_position[0].y - origin.y;
    float angle = -degrees(atan(x_diff, y_diff)) + 90;
    // Make sure the angle is between 0 and 360
    if (angle < 0) angle += 360;
    // Check if the target is within the angle range of the origin
    if (((direction - half_angle_range) > angle) || (angle > (direction + half_angle_range))) return;
    // Read samples from the wall texture in a line looking for obstacles. We simply make a vector between the origin
    // and the sprite location and trace pixels in this path with a reasonable step calculated using the resolution
    int steps = int(max_distance / resolution);
    vec2 vector_direction = v_position[0] - origin;
    for (int i = 0; i < steps; i++) {
        // Read pixels along the vector. We multiply by (float(i)/float(steps)) since vector_distance is the whole
        // distance between the player and the target, but we only want a portion of it hence the percentage
        vec2 pos = origin_relative + vector_direction * (float(i) / float(steps));
        vec4 color = texture(walls, screen_to_texcoord(pos));
        // If we find non-zero pixel data then we have obstacles in our path
        if (color != vec4(0.0)) return;
    }
    // We simply return the primitive index. This is a built in counter in geometry shaders starting at 0 and
    // incrementing by 1 for every invocation. It should always match the spritelist index
    sprite_index = float(gl_PrimitiveIDIn);
    EmitVertex();
    EndPrimitive();
}
