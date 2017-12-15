#version 330

#extension GL_EXT_gpu_shader4_1 : enable

uniform sampler2D u_particles;
uniform sampler2D u_wind;
uniform vec2 u_wind_res;
uniform vec2 u_wind_min;
uniform vec2 u_wind_max;
uniform float u_rand_seed;
uniform float u_speed_factor;
uniform float u_drop_rate;
uniform float u_drop_rate_bump;

uniform float unknown_const;
uniform bool periodic;

in vec2 v_tex_pos;
out vec4 fragColor;

//////////////////////////////////
// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}
// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random_float( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random_vec2( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random_vec3( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random_vec4( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
//////////////////////////////////

// pseudo-random generator
const vec3 rand_constants = vec3(12.9898, 78.233, 4375.85453);
float rand(const vec2 co) {
    float t = dot(co, rand_constants.xy);
//     float t = dot(rand_constants.xy, co);
    return fract(sin(t) * (rand_constants.z + t));
}

highp float rand1(vec2 co)
{
    highp float a = 12.9898;
    highp float b = 78.233;
    highp float c = 43758.5453;
    highp float dt= dot(co.xy ,vec2(a,b));
    highp float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}


// float rand(vec2 n) { 
// 	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
// }

// float noise(vec2 p){
// 	vec2 ip = floor(p);
// 	vec2 u = fract(p);
// 	u = u*u*(3.0-2.0*u);
// 	
// 	float res = mix(
// 		mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
// 		mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
// 	return res*res;
// }

// int random(int seed, int iterations)      
//     {                                                                                                           
//         int value = seed;                                                                                       
//         int n;                                                                                                  
//                                                                                                                 
//         for (n = 0; n < iterations; n++) {                                                                      
//             value = ((value >> 7) ^ (value << 9)) * 15485863;                                                   
//         }                                                                                                       
//                                                                                                                 
//         return value;                                                                                           
//     }                                                                                                           
//                                                                                                                 
// vec2 random_vector(int seed)                                                                                
// {                                                                                                           
//     int r = random(seed, 4);                                                                       
//     int g = random(r, 2);                                                                                   
//                                                                                                             
//     return vec2(float(r & 0x3FF) / 1024.0,                                                                  
//                 float(g & 0x3FF) / 1024.0);   
//                 }


vec2 random_vector(float seed)                                                                                
 {                                                                                                           
     float r = random_float(seed);                                                                       
     float g = random_float(r);
                                                                                                             
     return vec2(r, g);
     }


// wind speed lookup; use manual bilinear filtering based on 4 adjacent pixels for smooth interpolation
vec2 lookup_wind(const vec2 uv) {
    // return texture2D(u_wind, uv).rg; // lower-res hardware filtering
    vec2 px = 1.0 / u_wind_res;
    vec2 vc = (floor(uv * u_wind_res)) * px;
    vec2 f = fract(uv * u_wind_res);
    vec2 tl = texture(u_wind, vc).rg;
    vec2 tr = texture(u_wind, vc + vec2(px.x, 0)).rg;
    vec2 bl = texture(u_wind, vc + vec2(0, px.y)).rg;
    vec2 br = texture(u_wind, vc + px).rg;
    return mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y);
}

void main() {
    vec4 color = texture(u_particles, v_tex_pos);
    vec2 pos = vec2(
        color.r / 255.0 + color.b,
        color.g / 255.0 + color.a); // decode particle position from pixel RGBA

    vec2 velocity = mix(u_wind_min, u_wind_max, lookup_wind(pos));
    // relative speed in range [0, 1]
    float speed_t = length(velocity) / length(u_wind_max);

    // take EPSG:4236 distortion into account for calculating where the particle moved
     float distortion = 1.0;
//     float distortion = cos(radians(pos.y * 180.0 - 90.0));
     // TODO: Check y velocity sign
     vec2 offset = vec2(velocity.x / distortion, velocity.y) * unknown_const * u_speed_factor;
//      vec2 offset = vec2(0.0, 0.0) * u_speed_factor;

    // update particle position, wrapping around the date line. Periodic boundary along x and y
    if (periodic)
        {
            pos = fract(1.0 + pos + offset);
        }
    else{
            pos = pos + offset;
        }
//      float aa = gl_InstanceID;
    // a random seed to use for the particle drop
     vec2 seed = (pos + v_tex_pos) * gl_FragCoord.xy;
//     vec2 seed = (pos + v_tex_pos) * u_rand_seed;
    
    // drop rate is a chance a particle will restart at random position, to avoid degeneration
    // Solve the problem of areas with fast points that are denser than areas with slow points
    // increase reset rate for fast particles   
    float new_seed = random_vec2(seed);
    float drop_rate = u_drop_rate + speed_t * u_drop_rate_bump;
    float drop = step(1.0 - drop_rate, new_seed);
//     float drop = step(1.0 - drop_rate, rand(seed));
       
    vec2 random_pos = random_vector(new_seed);
//     vec2 random_pos = vec2(
//           rand(seed+1.3),
//           rand(seed+2.1));
    pos = mix(pos, random_pos, drop);
    
    // encode the new particle position back into RGBA
    fragColor = vec4(
        fract(pos * 255.0),
        floor(pos * 255.0)/255.0);
}
