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

float distributionGGX(vec3 n, vec3 h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float ndh = max(dot(n, h), 0.0);
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * d * d, 1e-5);
}

float geometrySchlickGGX(float nd, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return nd / max(nd * (1.0 - k) + k, 1e-5);
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
    float underground = 1.0 - smoothstep(-6.0, -1.0, vWorldPos.y);

    // The huge surface terrain plane physically sits above the bored tunnel.
    // It must not enter the underground camera's depth buffer; otherwise any
    // tiny gap at a segment/station join turns the entire ceiling grass green.
    bool isSurfaceGrass = abs(vWorldPos.y) < 0.18
        && abs(n.y) > 0.92
        && distance(vColor, vec3(0.31, 0.41, 0.19)) < 0.055;
    if (pc.camPos.y < -1.0 && isSurfaceGrass) {
        discard;
    }

    // --- Procedural albedo detail (breaks up the flat toy-plastic look). ---
    // Planar-map along the dominant normal axis so walls don't get streaks.
    vec3 an = abs(n);
    vec2 uv = (an.y >= an.x && an.y >= an.z) ? vWorldPos.xz
            : (an.x >= an.z) ? vWorldPos.zy : vWorldPos.xy;
    // Stable world-space coverage dither gives glass and foliage real holes
    // without requiring a costly sorted transparency pass.  Low translucency
    // materials (train glazing) remain solid and retain clean reflections.
    if (transl > 0.28) {
        float coverage = hash(floor(uv * 37.0));
        if (coverage < (transl - 0.28) * 0.45) {
            discard;
        }
    }
    // Fine grain only near the camera (it aliases away at range anyway).
    float detail = vnoise(uv * 1.9) * 0.6
                 + ((dist < 70.0) ? vnoise(uv * 8.7) * 0.4 : 0.2);
    // Vertex colors are authored as sRGB; light in linear space (gamma-2
    // approximation) so darks stay rich instead of washing out pastel.
    vec3 albedo = vColor * vColor;
    // Rough materials show strong grain; polished paint stays clean.
    albedo *= 1.0 + (detail - 0.5) * mix(0.45, 0.08, gloss);

    // Shield-tunnel segment joints.  The tunnel shell uses a deliberately
    // distinct 0.17 gloss value, letting us add fine ring seams without
    // striping station walls or the polished train.  This costs no extra
    // geometry across the 39 km underground section.
    float tunnelShell = underground
        * (1.0 - smoothstep(0.018, 0.045, abs(gloss - 0.17)))
        * smoothstep(0.12, 0.72, 1.0 - abs(n.y));
    float ringPhase = vWorldPos.x * 3.35 + vWorldPos.z * 0.055;
    float ringJoint = pow(0.5 + 0.5 * cos(ringPhase), 26.0);
    float segmentMottle = vnoise(vWorldPos.xz * vec2(0.055, 0.21));
    albedo *= 1.0 - tunnelShell * (0.16 * ringJoint
                                  + 0.055 * (segmentMottle - 0.5));

    // --- Direct sun (warm, low). ---
    float ndl = clamp(dot(n, SUN_DIR), 0.0, 1.0);
    vec3 direct = SUN_COLOR * ndl * mix(1.0, 0.015, underground);

    // --- Ambient: dusk sky hemisphere + warm ground bounce, with a cheap
    // contact-occlusion term that darkens vertical surfaces near the ground.
    float up = n.y * 0.5 + 0.5;
    vec3 skyAmb = mix(HORIZON_COOL, SKY_ZENITH, up);
    float ao = mix(clamp(0.55 + vWorldPos.y * 0.22, 0.55, 1.0), 1.0,
                   clamp(n.y, 0.0, 1.0));
    vec3 ambient = mix(GROUND_BOUNCE, skyAmb, up) * 0.42 * ao;
    ambient *= mix(1.0, 0.16, underground);

    vec3 color = albedo * (direct + ambient);

    // Underground fluorescent pools: repeated cool-white fixtures illuminate
    // walls, rails and train bodies instead of impossible sunlight in tunnels.
    float fixtureWave = 0.5 + 0.5 * sin(vWorldPos.x * 0.2618
                                       + vWorldPos.z * 0.016);
    float fixturePool = 0.55 + 1.25 * pow(fixtureWave, 10.0);
    float floorBounce = 0.78 + 0.22 * clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 tunnelLight = vec3(0.58, 0.72, 0.92) * fixturePool * floorBounce;
    color += albedo * tunnelLight * underground;

    // --- Cook-Torrance microfacet specular for steel, paint and glass. ---
    vec3 h = normalize(SUN_DIR + v);
    float ndv = max(dot(n, v), 0.0);
    float fres = pow(1.0 - ndv, 5.0);
    float roughness = mix(0.92, 0.10, gloss);
    float ndf = distributionGGX(n, h, roughness);
    float geometry = geometrySchlickGGX(ndv, roughness)
                   * geometrySchlickGGX(ndl, roughness);
    vec3 f0 = mix(vec3(0.035), vec3(0.62), gloss * gloss);
    vec3 F = f0 + (1.0 - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    vec3 microfacet = (ndf * geometry * F) / max(4.0 * ndv * ndl, 1e-4);
    color += SUN_COLOR * microfacet * ndl;
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

    // --- Volumetric sunlight. ---
    // Analytic height fog integrates density along the camera ray.  A strongly
    // forward-scattering phase term turns the low sun into visible golden
    // shafts between buildings/trees without an expensive 3D froxel buffer.
    vec3 rayDir = toFrag / max(dist, 1e-4);
    float heightDensity = exp(-max(0.0, (pc.camPos.y + vWorldPos.y) * 0.5) * 0.075);
    float mistNoise = mix(0.72, 1.18,
        vnoise(vWorldPos.xz * 0.028 + vec2(pc.camPos.w * 0.008, 0.0)));
    float opticalDepth = (1.0 - exp(-dist * 0.0085)) * heightDensity * mistNoise
                         * (1.0 - underground);
    float sunPhase = pow(clamp(dot(rayDir, SUN_DIR), 0.0, 1.0), 10.0);
    vec3 inScatter = SUN_COLOR * opticalDepth * (0.055 + 0.62 * sunPhase);
    color = color * exp(-opticalDepth * 0.22) + inScatter;

    // --- Aerial perspective: warm amber toward the sun, cool grey away.
    float sunAmount = pow(max(dot(-v, SUN_DIR), 0.0), 4.0);
    vec3 fogColor = mix(HORIZON_COOL, HORIZON_WARM, sunAmount) * 0.95;
    fogColor = mix(fogColor, vec3(0.20, 0.27, 0.34), underground);
    float fog = 1.0 - exp(-dist * 0.0012);
    color = mix(color, fogColor, clamp(fog, 0.0, 0.85));

    outColor = vec4(pow(aces(color), vec3(1.0 / 2.2)), 1.0);
}
