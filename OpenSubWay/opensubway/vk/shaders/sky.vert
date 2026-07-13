#version 450

// Fullscreen triangle (no vertex buffer); emits a world-space view ray so the
// fragment shader can shade the sky analytically. Drawn at depth 1.0 with
// LESS_OR_EQUAL so it only fills pixels no geometry covered.

layout(push_constant) uniform Push {
    mat4 invViewProj;   // inverse of (clip * proj * view)
    vec4 camPos;        // xyz = camera world position, w = time (seconds)
} pc;

layout(location = 0) out vec3 vDir;
layout(location = 1) out float vTime;

void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec2 ndc = uv * 2.0 - 1.0;
    gl_Position = vec4(ndc, 1.0, 1.0);

    vec4 far = pc.invViewProj * vec4(ndc, 1.0, 1.0);
    vDir = far.xyz / far.w - pc.camPos.xyz;
    vTime = pc.camPos.w;
}
