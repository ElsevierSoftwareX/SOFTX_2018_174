#version 330

layout (location = 0) in vec2 Position;
layout (location = 1) in vec2 texCoords;

uniform mat4 MVP;

out vec2 texCoords0;

void main()
{
    texCoords0 = texCoords;
    gl_Position = MVP * vec4(Position, 0.0, 1.0);
}
