#version 330
// Sprite positions from SpriteList
in vec2 in_pos;
// Output to geometry shader
out vec2 v_position;
void main() {
    // This shader just forwards info to geo shader
    v_position = in_pos;
}
