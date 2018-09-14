#version 330

uniform sampler2D gMap;

out vec4 fragColor;
in vec2 texCoords0;

//-----------------------------------------------------------------------------
void main()
 {
    vec3 value = texture(gMap, texCoords0).rgb;
    fragColor = vec4(value, 1.0);
}
