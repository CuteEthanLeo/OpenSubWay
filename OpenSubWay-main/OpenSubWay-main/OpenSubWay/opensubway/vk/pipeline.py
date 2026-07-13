"""Graphics pipeline(s). Milestone 2 provides the 3D scene pipeline."""

from __future__ import annotations

from pathlib import Path

from vulkan import *  # noqa: F401,F403

SHADER_DIR = Path(__file__).resolve().parent / "shaders"

# Scene vertex layout:
# position(3)+normal(3)+color(3)+emissive(1)+material(2) floats = 48 bytes.
SCENE_VERTEX_STRIDE = 12 * 4
PUSH_CONSTANT_SIZE = 144  # two mat4 (mvp + model) + vec4 camPos

# Sky push constants: mat4 invViewProj + vec4 camPos.
SKY_PUSH_SIZE = 80

# Text vertex layout: pos(2) + uv(2) + rgba(4) floats = 32 bytes.
TEXT_VERTEX_STRIDE = 8 * 4


def load_shader_module(ctx, spv_path: Path):
    code = spv_path.read_bytes()
    info = VkShaderModuleCreateInfo(
        sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        codeSize=len(code),
        pCode=code,
    )
    return vkCreateShaderModule(ctx.device, info, None)


class ScenePipeline:
    def __init__(self, ctx, render_pass):
        self.ctx = ctx
        self.pipeline = None
        self.layout = None
        self._build(render_pass)

    def _build(self, render_pass):
        ctx = self.ctx
        vert = load_shader_module(ctx, SHADER_DIR / "scene.vert.spv")
        frag = load_shader_module(ctx, SHADER_DIR / "scene.frag.spv")

        stages = [
            VkPipelineShaderStageCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=VK_SHADER_STAGE_VERTEX_BIT,
                module=vert,
                pName="main",
            ),
            VkPipelineShaderStageCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=VK_SHADER_STAGE_FRAGMENT_BIT,
                module=frag,
                pName="main",
            ),
        ]

        binding = VkVertexInputBindingDescription(
            binding=0, stride=SCENE_VERTEX_STRIDE, inputRate=VK_VERTEX_INPUT_RATE_VERTEX
        )
        attrs = [
            VkVertexInputAttributeDescription(
                location=0, binding=0, format=VK_FORMAT_R32G32B32_SFLOAT, offset=0
            ),
            VkVertexInputAttributeDescription(
                location=1, binding=0, format=VK_FORMAT_R32G32B32_SFLOAT, offset=12
            ),
            VkVertexInputAttributeDescription(
                location=2, binding=0, format=VK_FORMAT_R32G32B32_SFLOAT, offset=24
            ),
            VkVertexInputAttributeDescription(
                location=3, binding=0, format=VK_FORMAT_R32_SFLOAT, offset=36
            ),
            VkVertexInputAttributeDescription(
                location=4, binding=0, format=VK_FORMAT_R32G32_SFLOAT, offset=40
            ),
        ]
        vertex_input = VkPipelineVertexInputStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions=[binding],
            vertexAttributeDescriptionCount=len(attrs),
            pVertexAttributeDescriptions=attrs,
        )

        input_assembly = VkPipelineInputAssemblyStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=VK_FALSE,
        )

        viewport_state = VkPipelineViewportStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            scissorCount=1,
        )

        rasterizer = VkPipelineRasterizationStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=VK_FALSE,
            rasterizerDiscardEnable=VK_FALSE,
            polygonMode=VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=VK_CULL_MODE_NONE,
            frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthBiasEnable=VK_FALSE,
        )

        multisample = VkPipelineMultisampleStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sampleShadingEnable=VK_FALSE,
            rasterizationSamples=ctx.msaa_samples,
        )

        depth_stencil = VkPipelineDepthStencilStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable=VK_TRUE,
            depthWriteEnable=VK_TRUE,
            depthCompareOp=VK_COMPARE_OP_LESS,
            depthBoundsTestEnable=VK_FALSE,
            stencilTestEnable=VK_FALSE,
        )

        blend_attachment = VkPipelineColorBlendAttachmentState(
            colorWriteMask=VK_COLOR_COMPONENT_R_BIT
            | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT
            | VK_COLOR_COMPONENT_A_BIT,
            blendEnable=VK_FALSE,
        )
        color_blend = VkPipelineColorBlendStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=VK_FALSE,
            attachmentCount=1,
            pAttachments=[blend_attachment],
        )

        dynamic_state = VkPipelineDynamicStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount=2,
            pDynamicStates=[VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR],
        )

        push_range = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            offset=0,
            size=PUSH_CONSTANT_SIZE,
        )
        layout_info = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=0,
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_range],
        )
        self.layout = vkCreatePipelineLayout(ctx.device, layout_info, None)

        pipeline_info = VkGraphicsPipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=len(stages),
            pStages=stages,
            pVertexInputState=vertex_input,
            pInputAssemblyState=input_assembly,
            pViewportState=viewport_state,
            pRasterizationState=rasterizer,
            pMultisampleState=multisample,
            pDepthStencilState=depth_stencil,
            pColorBlendState=color_blend,
            pDynamicState=dynamic_state,
            layout=self.layout,
            renderPass=render_pass,
            subpass=0,
        )
        self.pipeline = vkCreateGraphicsPipelines(ctx.device, None, 1, [pipeline_info], None)[0]

        vkDestroyShaderModule(ctx.device, vert, None)
        vkDestroyShaderModule(ctx.device, frag, None)

    def destroy(self):
        ctx = self.ctx
        if self.pipeline:
            vkDestroyPipeline(ctx.device, self.pipeline, None)
            self.pipeline = None
        if self.layout:
            vkDestroyPipelineLayout(ctx.device, self.layout, None)
            self.layout = None


class SkyPipeline:
    """Procedural sky: a fullscreen triangle at depth 1.0 (no vertex buffer).

    Drawn after the scene with LESS_OR_EQUAL depth so it only shades pixels
    that no geometry covered.
    """

    def __init__(self, ctx, render_pass):
        self.ctx = ctx
        self.pipeline = None
        self.layout = None
        self._build(render_pass)

    def _build(self, render_pass):
        ctx = self.ctx
        vert = load_shader_module(ctx, SHADER_DIR / "sky.vert.spv")
        frag = load_shader_module(ctx, SHADER_DIR / "sky.frag.spv")

        stages = [
            VkPipelineShaderStageCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=VK_SHADER_STAGE_VERTEX_BIT, module=vert, pName="main",
            ),
            VkPipelineShaderStageCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=VK_SHADER_STAGE_FRAGMENT_BIT, module=frag, pName="main",
            ),
        ]

        vertex_input = VkPipelineVertexInputStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=0,
            vertexAttributeDescriptionCount=0,
        )
        input_assembly = VkPipelineInputAssemblyStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=VK_FALSE,
        )
        viewport_state = VkPipelineViewportStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1, scissorCount=1,
        )
        rasterizer = VkPipelineRasterizationStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=VK_FALSE, rasterizerDiscardEnable=VK_FALSE,
            polygonMode=VK_POLYGON_MODE_FILL, lineWidth=1.0,
            cullMode=VK_CULL_MODE_NONE, frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthBiasEnable=VK_FALSE,
        )
        multisample = VkPipelineMultisampleStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sampleShadingEnable=VK_FALSE, rasterizationSamples=ctx.msaa_samples,
        )
        depth_stencil = VkPipelineDepthStencilStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable=VK_TRUE, depthWriteEnable=VK_FALSE,
            depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL,
            depthBoundsTestEnable=VK_FALSE, stencilTestEnable=VK_FALSE,
        )
        blend_attachment = VkPipelineColorBlendAttachmentState(
            colorWriteMask=VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            blendEnable=VK_FALSE,
        )
        color_blend = VkPipelineColorBlendStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=VK_FALSE, attachmentCount=1, pAttachments=[blend_attachment],
        )
        dynamic_state = VkPipelineDynamicStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount=2,
            pDynamicStates=[VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR],
        )

        push_range = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_VERTEX_BIT, offset=0, size=SKY_PUSH_SIZE
        )
        layout_info = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=0,
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_range],
        )
        self.layout = vkCreatePipelineLayout(ctx.device, layout_info, None)

        pipeline_info = VkGraphicsPipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=len(stages), pStages=stages,
            pVertexInputState=vertex_input, pInputAssemblyState=input_assembly,
            pViewportState=viewport_state, pRasterizationState=rasterizer,
            pMultisampleState=multisample, pDepthStencilState=depth_stencil,
            pColorBlendState=color_blend, pDynamicState=dynamic_state,
            layout=self.layout, renderPass=render_pass, subpass=0,
        )
        self.pipeline = vkCreateGraphicsPipelines(ctx.device, None, 1, [pipeline_info], None)[0]

        vkDestroyShaderModule(ctx.device, vert, None)
        vkDestroyShaderModule(ctx.device, frag, None)

    def destroy(self):
        ctx = self.ctx
        if self.pipeline:
            vkDestroyPipeline(ctx.device, self.pipeline, None)
            self.pipeline = None
        if self.layout:
            vkDestroyPipelineLayout(ctx.device, self.layout, None)
            self.layout = None


class TextPipeline:
    """2D overlay pipeline: alpha-blended textured quads, no depth test.

    Samples a single R8 font atlas via one combined-image-sampler descriptor.
    """

    def __init__(self, ctx, render_pass):
        self.ctx = ctx
        self.pipeline = None
        self.layout = None
        self.descriptor_set_layout = None
        self._build(render_pass)

    def _build(self, render_pass):
        ctx = self.ctx
        vert = load_shader_module(ctx, SHADER_DIR / "text.vert.spv")
        frag = load_shader_module(ctx, SHADER_DIR / "text.frag.spv")

        stages = [
            VkPipelineShaderStageCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=VK_SHADER_STAGE_VERTEX_BIT, module=vert, pName="main",
            ),
            VkPipelineShaderStageCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=VK_SHADER_STAGE_FRAGMENT_BIT, module=frag, pName="main",
            ),
        ]

        binding = VkVertexInputBindingDescription(
            binding=0, stride=TEXT_VERTEX_STRIDE, inputRate=VK_VERTEX_INPUT_RATE_VERTEX
        )
        attrs = [
            VkVertexInputAttributeDescription(
                location=0, binding=0, format=VK_FORMAT_R32G32_SFLOAT, offset=0
            ),
            VkVertexInputAttributeDescription(
                location=1, binding=0, format=VK_FORMAT_R32G32_SFLOAT, offset=8
            ),
            VkVertexInputAttributeDescription(
                location=2, binding=0, format=VK_FORMAT_R32G32B32A32_SFLOAT, offset=16
            ),
        ]
        vertex_input = VkPipelineVertexInputStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions=[binding],
            vertexAttributeDescriptionCount=len(attrs),
            pVertexAttributeDescriptions=attrs,
        )

        input_assembly = VkPipelineInputAssemblyStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=VK_FALSE,
        )
        viewport_state = VkPipelineViewportStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1, scissorCount=1,
        )
        rasterizer = VkPipelineRasterizationStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=VK_FALSE, rasterizerDiscardEnable=VK_FALSE,
            polygonMode=VK_POLYGON_MODE_FILL, lineWidth=1.0,
            cullMode=VK_CULL_MODE_NONE, frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthBiasEnable=VK_FALSE,
        )
        multisample = VkPipelineMultisampleStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sampleShadingEnable=VK_FALSE, rasterizationSamples=ctx.msaa_samples,
        )
        depth_stencil = VkPipelineDepthStencilStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable=VK_FALSE, depthWriteEnable=VK_FALSE,
            depthCompareOp=VK_COMPARE_OP_ALWAYS,
            depthBoundsTestEnable=VK_FALSE, stencilTestEnable=VK_FALSE,
        )
        blend_attachment = VkPipelineColorBlendAttachmentState(
            colorWriteMask=VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            blendEnable=VK_TRUE,
            srcColorBlendFactor=VK_BLEND_FACTOR_SRC_ALPHA,
            dstColorBlendFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            colorBlendOp=VK_BLEND_OP_ADD,
            srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            alphaBlendOp=VK_BLEND_OP_ADD,
        )
        color_blend = VkPipelineColorBlendStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=VK_FALSE, attachmentCount=1, pAttachments=[blend_attachment],
        )
        dynamic_state = VkPipelineDynamicStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount=2,
            pDynamicStates=[VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR],
        )

        # Descriptor set layout: one combined image sampler for the atlas.
        sampler_binding = VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT,
        )
        dsl_info = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=1, pBindings=[sampler_binding],
        )
        self.descriptor_set_layout = vkCreateDescriptorSetLayout(ctx.device, dsl_info, None)

        layout_info = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1, pSetLayouts=[self.descriptor_set_layout],
            pushConstantRangeCount=0,
        )
        self.layout = vkCreatePipelineLayout(ctx.device, layout_info, None)

        pipeline_info = VkGraphicsPipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=len(stages), pStages=stages,
            pVertexInputState=vertex_input, pInputAssemblyState=input_assembly,
            pViewportState=viewport_state, pRasterizationState=rasterizer,
            pMultisampleState=multisample, pDepthStencilState=depth_stencil,
            pColorBlendState=color_blend, pDynamicState=dynamic_state,
            layout=self.layout, renderPass=render_pass, subpass=0,
        )
        self.pipeline = vkCreateGraphicsPipelines(ctx.device, None, 1, [pipeline_info], None)[0]

        vkDestroyShaderModule(ctx.device, vert, None)
        vkDestroyShaderModule(ctx.device, frag, None)

    def destroy(self):
        ctx = self.ctx
        if self.pipeline:
            vkDestroyPipeline(ctx.device, self.pipeline, None)
            self.pipeline = None
        if self.layout:
            vkDestroyPipelineLayout(ctx.device, self.layout, None)
            self.layout = None
        if self.descriptor_set_layout:
            vkDestroyDescriptorSetLayout(ctx.device, self.descriptor_set_layout, None)
            self.descriptor_set_layout = None
