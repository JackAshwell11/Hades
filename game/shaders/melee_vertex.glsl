#version 330
// Sprite positions from spritelist
in vec2 in_pos;
// Output to geometry shader
out vec2 v_position;
// Main vertex shader function
void main() {
    // This shader just forwards data to the geometry shader
    v_position = in_pos;
}
