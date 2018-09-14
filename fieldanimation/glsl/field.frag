#version 330

const float pi = 3.141592653589793238462643383279;

uniform sampler2D gMap;

// CubeHelix parameters
// START colour (1=red, 2=green, 3=blue; e.g. 0.5=purple);
uniform float start;
// ROTS  rotations in colour (typically -1.5 to 1.5, e.g. -1.0
//       is one blue->green->red cycle);
uniform float rot;
// Hue (colour saturation) 0 = greyscale.
uniform float gamma;
uniform bool reverse;
uniform bool useHue;
uniform float minSat;
uniform float maxSat;
uniform float minLight;
uniform float maxLight;
uniform float startHue;
uniform float endHue;

out vec4 fragColor;
in vec2 texCoords0;

vec3 cubeHelix(float x)
{
   // Implement D.A. Green's cubehelix colour scheme
   // Input x ranges from 0 to 1.
   float red, green, blue, frc, satar, amp, angle, newstart, newrot;

   if (useHue)
   {
        newstart = (startHue / 360. - 1) * 3.;
        newrot = endHue / 360. - newstart / 3. - 1.;
   }
   else{
       newstart = start;
       newrot = rot;
   }

   frc = int(x*255.)*(maxLight-minLight)/255.+minLight;
   angle = 2.0 * pi * (newstart / 3.0 + newrot * frc + 1.);
   frc = pow(frc, gamma);

   satar = int(x*255.)*(maxSat-minSat)/255.+minSat;
   amp = satar * frc * (1. - frc) / 2.;

   red   = frc + amp * (-0.14861 * cos(angle) + 1.78277 * sin(angle));
   green = frc + amp * (-0.29227 * cos(angle) - 0.90649 * sin(angle));
   blue  = frc + amp * (1.97294 * cos(angle));

   if (red > 1.)
       red = 1.;
   else if (red < 0.)
       red = 0.;
   if (green > 1.)
       green = 1.;
   else if (green < 0.)
       green = 0.;
   if (blue > 1.)
       blue = 1.;
   else if (blue < 0.)
       blue = 0.;


    if (reverse){
        red = 1.0-red;
        green = 1.0-green;
        blue = 1.0-blue;
    }

    return vec3(red, green, blue);

}


//-----------------------------------------------------------------------------
void main()
 {
    float value = texture(gMap, texCoords0).r;
    fragColor = vec4(cubeHelix(value), 1.0);
}
