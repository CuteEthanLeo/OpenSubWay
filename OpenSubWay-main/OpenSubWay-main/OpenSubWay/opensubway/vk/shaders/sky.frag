#version 450

layout(location = 0) in vec3 vDir;
layout(location = 1) in float vTime;
layout(location = 0) out vec4 outColor;

// Golden-hour rig — keep these constants in sync with scene.frag.
const vec3 SUN_DIR      = normalize(vec3(0.62, 0.30, 0.42));
const vec3 SUN_COLOR    = vec3(1.00, 0.70, 0.40);
const vec3 SKY_ZENITH   = vec3(0.13, 0.20, 0.40);
const vec3 HORIZON_WARM = vec3(0.98, 0.64, 0.35);
const vec3 HORIZON_COOL = vec3(0.44, 0.48, 0.62);

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Four-octave FBM: enough structure for cumulus without melting the CPU.
float fbm(vec2 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        s += a * vnoise(p);
        p = p * 2.17 + vec2(19.7, 7.3);
        a *= 0.5;
    }
    return s;
}

// Cheaper two-octave variant for the sun-shading resample.
float fbm2(vec2 p) {
    return vnoise(p) * 0.5 + vnoise(p * 2.17 + vec2(19.7, 7.3)) * 0.25 + 0.125;
}

// Narkowicz ACES filmic approximation.
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec3 d = normalize(vDir);
    float y = d.y;
    float sd = clamp(dot(d, SUN_DIR), 0.0, 1.0);

    // Horizon shifts amber toward the sun, cool grey-blue away from it.
    vec3 horizon = mix(HORIZON_COOL, HORIZON_WARM, pow(sd, 2.0));
    vec3 sky = mix(horizon, SKY_ZENITH, pow(clamp(y, 0.0, 1.0), 0.5));
    // Below the horizon, settle into a soft dusk haze.
    sky = mix(horizon * 0.85, sky, smoothstep(-0.08, 0.02, y));

    // Sun disc, inner halo, and a broad golden glow hugging the horizon.
    float horizGlow = 1.0 - clamp(abs(y) * 4.0, 0.0, 1.0);
    sky += SUN_COLOR * (smoothstep(0.9991, 0.9997, sd) * 7.0
                        + pow(sd, 150.0) * 0.9
                        + pow(sd, 5.0) * (0.10 + 0.18 * horizGlow));

    // Crepuscular volumetric shafts around the low sun. Angular noise makes
    // broad rays that drift slowly, while the horizon mask keeps the effect in
    // the humid lower atmosphere instead of painting the whole sky.
    float rayAngle = atan(d.z, d.x) * 13.0;
    float rayBands = 0.58 + 0.42 * sin(rayAngle + vTime * 0.035
                                       + vnoise(d.xz * 19.0) * 5.0);
    float shafts = pow(sd, 13.0) * smoothstep(-0.02, 0.34, y)
                   * (1.0 - smoothstep(0.48, 0.82, y));
    sky += SUN_COLOR * shafts * rayBands * 0.32;

    // --- Cumulus layer with pseudo-volumetric shading. ---
    if (y > 0.015) {
        // Project onto an overhead plane; drift slowly with time.
        vec2 cp = d.xz / (y + 0.15) * 1.25 + vec2(37.0) + vTime * 0.006;
        float den = fbm(cp);
        float cover = smoothstep(0.44, 0.66, den);

        if (cover > 0.001) {
            // Sample the field again displaced toward the sun: where density
            // drops sunward, the cloud face is lit; where it rises, shadowed.
            vec2 sunStep = normalize(SUN_DIR.xz) * 0.55;
            float den2 = fbm2(cp + sunStep);
            float lit = clamp((den - den2) * 3.2 + 0.55, 0.0, 1.0);

            // Grey-mauve shadowed bases -> warm sunlit tops (golden hour).
            vec3 cloudDark = vec3(0.34, 0.33, 0.38);
            vec3 cloudLit = vec3(1.15, 0.92, 0.72);
            vec3 cloud = mix(cloudDark, cloudLit, lit);

            // Silver lining: thin edges near the sun glow from behind.
            float edge = cover * (1.0 - cover) * 4.0;
            cloud += SUN_COLOR * edge * pow(sd, 3.0) * 0.8;

            float fade = smoothstep(0.015, 0.10, y);   // sink into the haze
            sky = mix(sky, cloud, cover * fade * 0.92);
        }
    }

    outColor = vec4(pow(aces(sky * 1.15), vec3(1.0 / 2.2)), 1.0);
}
