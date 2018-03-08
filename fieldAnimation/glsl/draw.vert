#version 330
// Vertex texture fetch method:
// https://www.khronos.org/opengl/wiki/Vertex_Texture_Fetch
layout (location = 0) in float a_index;

uniform sampler2D u_tracers;
uniform float u_tracersRes;

// Model-View-Projection matrix
uniform mat4 MVP;
uniform float pointSize;

out vec2 tracerPos;

void main() {
    vec4 color = texture(u_tracers, vec2(
        fract(a_index / u_tracersRes),
        floor(a_index / u_tracersRes) / u_tracersRes));

    // decode current tracer position from the pixel's RGBA value
    // Ranbge from 0  to 1.0
    tracerPos = vec2(
        color.r / 255.0 + color.b,
        color.g / 255.0 + color.a);

    gl_PointSize = pointSize;
    gl_Position = MVP * vec4(tracerPos.x, tracerPos.y, 0, 1);
}
