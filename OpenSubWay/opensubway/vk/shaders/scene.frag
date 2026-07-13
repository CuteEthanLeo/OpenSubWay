#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 2) in float vEmissive;
layout(location = 3) in vec3 vWorldPos;
layout(location = 4) in vec2 vMaterial;   // x = gloss, y = translucency

layout(push_constant) uniform Push {
    mat4 mvp;
    mat4 model;
    vec4 camPos;   // xyz = camera, w = time
} pc;

layout(location = 0) out vec4 outColor;

// Golden-hour rig — keep these constants in sync with sky.frag.
const vec3 SUN_DIR       = normalize(vec3(0.62, 0.30, 0.42));
const vec3 SUN_COLOR     = vec3(1.00, 0.70, 0.40) * 3.2;
const vec3 SKY_ZENITH    = vec3(0.13, 0.20, 0.40);
const vec3 HORIZON_WARM  = vec3(0.98, 0.64, 0.35);
const vec3 HORIZON_COOL  = vec3(0.44, 0.48, 0.62);
const vec3 GROUND_BOUNCE = vec3(0.26, 0.20, 0.14);

// Narkowicz ACES filmic approximation.
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

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

void main() {
    vec3 n = normalize(vNormal);
    vec3 toFrag = vWorldPos - pc.camPos.xyz;
    float dist = length(toFrag);
    vec3 v = -toFrag / max(dist, 1e-4);
    float gloss = vMaterial.x;
    float transl = vMaterial.y;

    // --- Procedural albedo detail (breaks up the flat toy-plastic look). ---
    // Planar-map along the dominant normal axis so walls don't get streaks.
    vec3 an = abs(n);
    vec2 uv = (an.y >= an.x && an.y >= an.z) ? vWorldPos.xz
            : (an.x >= an.z) ? vWorldPos.zy : vWorldPos.xy;
    // Fine grain only near the camera (it aliases away at range anyway).
    float detail = vnoise(uv * 1.9) * 0.6
                 + ((dist < 70.0) ? vnoise(uv * 8.7) * 0.4 : 0.2);
    // Vertex colors are authored as sRGB; light in linear space (gamma-2
    // approximation) so darks stay rich instead of washing out pastel.
    vec3 albedo = vColor * vColor;
    // Rough materials show strong grain; polished paint stays clean.
    albedo *= 1.0 + (detail - 0.5) * mix(0.45, 0.08, gloss);

    // --- Direct sun (warm, low). ---
    float ndl = clamp(dot(n, SUN_DIR), 0.0, 1.0);
    vec3 direct = SUN_COLOR * ndl;

    // --- Ambient: dusk sky hemisphere + warm ground bounce, with a cheap
    // contact-occlusion term that darkens vertical surfaces near the ground.
    float up = n.y * 0.5 + 0.5;
    vec3 skyAmb = mix(HORIZON_COOL, SKY_ZENITH, up);
    float ao = mix(clamp(0.55 + vWorldPos.y * 0.22, 0.55, 1.0), 1.0,
                   clamp(n.y, 0.0, 1.0));
    vec3 ambient = mix(GROUND_BOUNCE, skyAmb, up) * 0.42 * ao;

    vec3 color = albedo * (direct + ambient);

    // --- Specular: gloss drives tightness and strength (steel, paint, glass).
    vec3 h = normalize(SUN_DIR + v);
    float ndv = max(dot(n, v), 0.0);
    float fres = pow(1.0 - ndv, 5.0);
    float specPow = mix(24.0, 220.0, gloss);
    float spec = pow(max(dot(n, h), 0.0), specPow) * ndl * (0.05 + 1.1 * gloss);
    color += SUN_COLOR * spec * (1.0 + 2.0 * fres);
    // Fresnel sky reflection — strong on glass/paint, faint on matte.
    color += skyAmb * fres * (0.08 + 0.55 * gloss);

    // --- Translucency: warm sunlight bleeding through foliage when the sun
    // is behind it relative to the viewer ("light through the trees").
    if (transl > 0.0) {
        float back = pow(clamp(dot(-v, SUN_DIR), 0.0, 1.0), 4.0);
        float thin = 1.0 - ndl;   // strongest on the shaded side
        color += SUN_COLOR * albedo * back * transl * (0.35 + 0.85 * thin);
        // Soft rim where low sun grazes the canopy silhouette.
        color += SUN_COLOR * albedo * fres * transl * 0.25;
    }

    // Self-illuminated surfaces (signs, headlights) ignore the light rig.
    color = mix(color, vColor * vColor * 2.2, clamp(vEmissive, 0.0, 1.0));

    // --- Aerial perspective: warm amber toward the sun, cool grey away.
    float sunAmount = pow(max(dot(-v, SUN_DIR), 0.0), 4.0);
    vec3 fogColor = mix(HORIZON_COOL, HORIZON_WARM, sunAmount) * 0.95;
    float fog = 1.0 - exp(-dist * 0.0012);
    color = mix(color, fogColor, clamp(fog, 0.0, 0.85));

    outColor = vec4(pow(aces(color), vec3(1.0 / 2.2)), 1.0);
}
