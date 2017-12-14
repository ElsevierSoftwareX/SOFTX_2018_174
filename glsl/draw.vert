#version 330
// Vertex texture fetch method: https://www.khronos.org/opengl/wiki/Vertex_Texture_Fetch
layout (location = 0) in float a_index;

uniform sampler2D u_particles;
uniform float u_particles_res;
// Model-View-Projection matrix
uniform mat4 MVP;
uniform float pointSize;

out vec2 v_particle_pos;

void main() {
    vec4 color = texture(u_particles, vec2(
        fract(a_index / u_particles_res),
        floor(a_index / u_particles_res) / u_particles_res));

    // decode current particle position from the pixel's RGBA value
    // Ranbge from 0  to 1.0
    v_particle_pos = vec2(
        color.r / 255.0 + color.b,
        color.g / 255.0 + color.a);

    gl_PointSize = pointSize;
    gl_Position = MVP * vec4(v_particle_pos.x, v_particle_pos.y, 0, 1);
}
