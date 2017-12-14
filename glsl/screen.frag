# version 330

uniform sampler2D u_screen;
uniform float u_opacity;

in vec2 v_tex_pos;
out vec4 FragColor;

void main() {

     vec4 color = texture(u_screen, v_tex_pos);
     FragColor = vec4(floor(255.0 * color * u_opacity) / 255.0);
}
