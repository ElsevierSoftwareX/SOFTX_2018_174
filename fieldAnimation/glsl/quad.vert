#version 330

layout (location = 0) in vec2 a_pos;
layout (location = 1) in vec2 a_tex;

out vec2 v_tex_pos;

void main() {
    v_tex_pos = a_tex;
    gl_Position = vec4(a_pos, 0, 1);
}
