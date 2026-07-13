#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in float inEmissive;
layout(location = 4) in vec2 inMaterial;   // x = gloss, y = translucency

layout(push_constant) uniform Push {
    mat4 mvp;      // clip * proj * view * model
    mat4 model;    // model matrix (world-space lighting/fog)
    vec4 camPos;   // xyz = camera world position, w = time (seconds)
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;
layout(location = 2) out float vEmissive;
layout(location = 3) out vec3 vWorldPos;
layout(location = 4) out vec2 vMaterial;

void main() {
    vec4 world = pc.model * vec4(inPos, 1.0);
    gl_Position = pc.mvp * vec4(inPos, 1.0);
    vNormal = mat3(pc.model) * inNormal;
    vColor = inColor;
    vEmissive = inEmissive;
    vWorldPos = world.xyz;
    vMaterial = inMaterial;
}
