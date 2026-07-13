#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec4 vColor;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D atlas;

void main() {
    float coverage = texture(atlas, vUV).r;   // R8 glyph coverage / solid = 1
    outColor = vec4(vColor.rgb, vColor.a * coverage);
}
