#version 330

#extension GL_EXT_gpu_shader4_1 : enable

uniform sampler2D u_tracers;
uniform sampler2D u_field;
uniform vec2 u_fieldRes;
uniform vec2 u_fieldMin;
uniform vec2 u_fieldMax;
uniform float u_rand_seed;
uniform float u_speed_factor;
uniform float u_drop_rate;
uniform float u_drop_rate_bump;

uniform float fieldScaling;
uniform bool periodic;

in vec2 v_tex_pos;
out vec4 fragColor;

//-----------------------------------------------------------------------------
// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

//-----------------------------------------------------------------------------
// Compound versions of the hashing algorithm I whipped together.
uint hash(uvec2 v) {return hash( v.x ^ hash(v.y)                         );}
uint hash(uvec3 v) {return hash( v.x ^ hash(v.y) ^ hash(v.z)             );}
uint hash(uvec4 v) {return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) );}

//-----------------------------------------------------------------------------
// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable
// value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;           // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

//-----------------------------------------------------------------------------
// Pseudo-random value in half-open range [0:1].
float random_float(float x) {
    return floatConstruct(hash(floatBitsToUint(x)));
}
//-----------------------------------------------------------------------------
float random_vec2(vec2  v) {
    return floatConstruct(hash(floatBitsToUint(v)));
}

//-----------------------------------------------------------------------------
vec2 random_vector(float seed) {
     float r = random_float(seed);
     float g = random_float(r);

     return vec2(r, g);
}

//-----------------------------------------------------------------------------
// Field modulus lookup; use manual bilinear filtering based on 4 adjacent
// pixels for smooth interpolation
vec2 lookupField(const vec2 uv) {
    // return texture2D(u_field, uv).rg; // lower-res hardware filtering
    vec2 px = 1.0 / u_fieldRes;
    vec2 vc = (floor(uv * u_fieldRes)) * px;
    vec2 f = fract(uv * u_fieldRes);
    vec2 tl = texture(u_field, vc).rg;
    vec2 tr = texture(u_field, vc + vec2(px.x, 0)).rg;
    vec2 bl = texture(u_field, vc + vec2(0, px.y)).rg;
    vec2 br = texture(u_field, vc + px).rg;
    return mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y);
}

//-----------------------------------------------------------------------------
void main() {
    vec4 color = texture(u_tracers, v_tex_pos);
    vec2 pos = vec2(
        color.r / 255.0 + color.b,
        color.g / 255.0 + color.a); // decode tracer position from pixel RGBA

    vec2 velocity = mix(u_fieldMin, u_fieldMax, lookupField(pos));
    float speed_t = length(velocity) / length(u_fieldMax);
    vec2 offset = vec2(velocity.x, velocity.y) * 1e-4* fieldScaling
         * u_speed_factor;

    // update tracer position, wrapping around the date line.
    // Periodic boundary along x and y
    if (periodic)
        {
            pos = fract(1.0 + pos + offset);
        }
    else{
            pos = pos + offset;
        }
    vec2 seed = (pos + v_tex_pos) * u_rand_seed;
    //vec2 seed = pos * u_rand_seed;

    // drop rate is a chance a tracer will restart at random position,
    // to avoid degeneration.
    // Solve the problem of areas with fast points that are denser than
    // areas with slow points.
    // Increase reset rate for fast tracers
    float new_seed = random_vec2(seed);
    //float drop_rate = u_drop_rate + speed_t;/* * u_drop_rate_bump;*/
    float drop_rate = u_drop_rate + speed_t * u_drop_rate_bump;
    float drop = step(1.0 - drop_rate, new_seed);

    vec2 random_pos = random_vector(new_seed);
    pos = mix(pos, random_pos, drop);

    // encode the new tracer position back into RGBA
    fragColor = vec4(
        fract(pos * 255.0),
        floor(pos * 255.0)/255.0);
}
