"""Frame orchestration: acquire -> record -> submit -> present."""

from __future__ import annotations

import struct

import glfw
import glm
from vulkan import *  # noqa: F401,F403
from vulkan import ffi

from .. import config
from ..render import camera as cam
from ..render.hud import FontAtlas, HudBuilder
from . import memory
from .commands import CommandResources
from .pipeline import TEXT_VERTEX_STRIDE, ScenePipeline, SkyPipeline, TextPipeline
from .swapchain import SwapchainBundle
from .texture import Texture

# Room for the HUD text mesh (vertices); comfortably above what we emit
# (status HUD + control panel + the full-screen Options overlay together).
TEXT_MAX_VERTICES = 24000

# Swapchain-out-of-date signalling differs between binding versions; catch broadly.
try:
    from vulkan import VkErrorOutOfDateKhr  # type: ignore
except Exception:  # pragma: no cover
    VkErrorOutOfDateKhr = None


def _is_out_of_date(exc) -> bool:
    if VkErrorOutOfDateKhr is not None and isinstance(exc, VkErrorOutOfDateKhr):
        return True
    return "OUT_OF_DATE" in repr(exc) or "SUBOPTIMAL" in repr(exc)


class GpuMesh:
    """A :class:`~opensubway.render.mesh.Mesh` uploaded to GPU buffers."""

    def __init__(self, ctx, mesh):
        self.ctx = ctx
        self.index_count = mesh.index_count
        self.vertex_buffer, self.vertex_memory = memory.create_host_buffer(
            ctx, mesh.vertex_bytes(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
        )
        self.index_buffer, self.index_memory = memory.create_host_buffer(
            ctx, mesh.index_bytes(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT
        )

    def destroy(self):
        ctx = self.ctx
        vkDestroyBuffer(ctx.device, self.vertex_buffer, None)
        vkFreeMemory(ctx.device, self.vertex_memory, None)
        vkDestroyBuffer(ctx.device, self.index_buffer, None)
        vkFreeMemory(ctx.device, self.index_memory, None)


class Renderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.swap = SwapchainBundle(ctx)
        self.scene_pipeline = ScenePipeline(ctx, self.swap.render_pass)
        self.sky_pipeline = SkyPipeline(ctx, self.swap.render_pass)
        self.text_pipeline = TextPipeline(ctx, self.swap.render_pass)
        self.commands = CommandResources(ctx)
        # A present-wait semaphore belongs to a swapchain image, not merely a
        # CPU frame slot. Reusing frame-indexed semaphores can race presentation
        # when images are acquired in a different order (caught by validation
        # on NVIDIA and capable of producing intermittent device loss).
        self._render_finished = []
        self._create_present_semaphores()
        self.current_frame = 0
        self.framebuffer_resized = False

        self._acquire = vkGetDeviceProcAddr(ctx.device, "vkAcquireNextImageKHR")
        self._present = vkGetDeviceProcAddr(ctx.device, "vkQueuePresentKHR")
        self.frames_presented = 0

        # --- HUD text resources ---
        atlas = FontAtlas()
        self.hud_builder = HudBuilder(atlas)
        self.atlas_texture = Texture(ctx, atlas.width, atlas.height, atlas.pixels)
        self._create_text_descriptor()

        cap = TEXT_MAX_VERTICES * TEXT_VERTEX_STRIDE
        self._text_buffers = []
        self._text_memories = []
        for _ in range(config.MAX_FRAMES_IN_FLIGHT):
            buf, mem = memory.create_buffer(
                ctx, cap, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )
            self._text_buffers.append(buf)
            self._text_memories.append(mem)
        self._text_capacity = cap
        self._text_counts = [0] * config.MAX_FRAMES_IN_FLIGHT

    def _create_text_descriptor(self):
        ctx = self.ctx
        pool_size = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount=1
        )
        pool_info = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1, poolSizeCount=1, pPoolSizes=[pool_size],
        )
        self._descriptor_pool = vkCreateDescriptorPool(ctx.device, pool_info, None)

        alloc = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self._descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.text_pipeline.descriptor_set_layout],
        )
        self.text_descriptor_set = vkAllocateDescriptorSets(ctx.device, alloc)[0]

        image_info = VkDescriptorImageInfo(
            sampler=self.atlas_texture.sampler,
            imageView=self.atlas_texture.view,
            imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )
        write = VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.text_descriptor_set,
            dstBinding=0, dstArrayElement=0,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=1, pImageInfo=[image_info],
        )
        vkUpdateDescriptorSets(ctx.device, 1, [write], 0, None)

    def _create_present_semaphores(self):
        info = VkSemaphoreCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO)
        self._render_finished = [
            vkCreateSemaphore(self.ctx.device, info, None) for _ in self.swap.images
        ]

    def _destroy_present_semaphores(self):
        for semaphore in self._render_finished:
            vkDestroySemaphore(self.ctx.device, semaphore, None)
        self._render_finished = []

    # ----------------------------------------------------------- swapchain
    def recreate_swapchain(self):
        # Wait until the window has a non-zero size (e.g. not minimised).
        w, h = glfw.get_framebuffer_size(self.ctx.window)
        while w == 0 or h == 0:
            glfw.wait_events()
            w, h = glfw.get_framebuffer_size(self.ctx.window)

        vkDeviceWaitIdle(self.ctx.device)
        self._destroy_present_semaphores()
        self.swap.destroy()
        self.swap.create()
        self._create_present_semaphores()
        # Pipeline remains valid: the new render pass is compatible with the old.

    # ---------------------------------------------------------------- draw
    def draw(self, camera, items, hud_lines=None, buttons=None, overlay=None, telemetry=None):
        """``items``: (GpuMesh, model) list. ``hud_lines``: (text, rgba) list.
        ``buttons``: control-panel Button list. ``overlay``: modal menu lines."""
        ctx = self.ctx
        frame = self.commands.frames[self.current_frame]

        vkWaitForFences(ctx.device, 1, [frame.in_flight], VK_TRUE, 2**64 - 1)

        try:
            image_index = self._acquire(
                ctx.device, self.swap.swapchain, 2**64 - 1,
                frame.image_available, None,
            )
        except Exception as exc:
            if _is_out_of_date(exc):
                self.recreate_swapchain()
                return
            raise

        render_finished = self._render_finished[image_index]

        # Build & upload HUD geometry for this frame (safe: its fence is signalled).
        self._upload_hud(hud_lines, buttons, overlay, telemetry)

        vkResetFences(ctx.device, 1, [frame.in_flight])
        vkResetCommandBuffer(frame.command_buffer, 0)
        self._record(frame.command_buffer, image_index, camera, items)

        submit = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            waitSemaphoreCount=1,
            pWaitSemaphores=[frame.image_available],
            pWaitDstStageMask=[VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
            commandBufferCount=1,
            pCommandBuffers=[frame.command_buffer],
            signalSemaphoreCount=1,
            pSignalSemaphores=[render_finished],
        )
        vkQueueSubmit(ctx.graphics_queue, 1, [submit], frame.in_flight)

        present_info = VkPresentInfoKHR(
            sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=[render_finished],
            swapchainCount=1,
            pSwapchains=[self.swap.swapchain],
            pImageIndices=[image_index],
        )
        try:
            self._present(ctx.present_queue, present_info)
        except Exception as exc:
            if _is_out_of_date(exc):
                self.framebuffer_resized = False
                self.recreate_swapchain()
            else:
                raise
        else:
            self.frames_presented += 1
            if self.framebuffer_resized:
                self.framebuffer_resized = False
                self.recreate_swapchain()

        self.current_frame = (self.current_frame + 1) % config.MAX_FRAMES_IN_FLIGHT

    def _upload_hud(self, hud_lines, buttons=None, overlay=None, telemetry=None):
        i = self.current_frame
        if not hud_lines and not buttons and not overlay:
            self._text_counts[i] = 0
            return
        data, count = self.hud_builder.build_frame(
            hud_lines or [], buttons or [], self.swap.extent.width,
            self.swap.extent.height, overlay=overlay, telemetry=telemetry,
        )
        if len(data) > self._text_capacity:  # clamp defensively
            max_verts = self._text_capacity // TEXT_VERTEX_STRIDE
            data = data[: max_verts * TEXT_VERTEX_STRIDE]
            count = max_verts
        memory.upload_to_memory(self.ctx, self._text_memories[i], data)
        self._text_counts[i] = count

    def _record(self, cmd, image_index, camera, items):
        swap = self.swap
        begin = VkCommandBufferBeginInfo(sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
        vkBeginCommandBuffer(cmd, begin)

        clear_color = VkClearValue(
            color=VkClearColorValue(float32=list(config.CLEAR_COLOR))
        )
        clear_depth = VkClearValue(
            depthStencil=VkClearDepthStencilValue(depth=1.0, stencil=0)
        )
        rp_begin = VkRenderPassBeginInfo(
            sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=swap.render_pass,
            framebuffer=swap.framebuffers[image_index],
            renderArea=VkRect2D(offset=VkOffset2D(x=0, y=0), extent=swap.extent),
            clearValueCount=2,
            pClearValues=[clear_color, clear_depth],
        )
        vkCmdBeginRenderPass(cmd, rp_begin, VK_SUBPASS_CONTENTS_INLINE)

        viewport = VkViewport(
            x=0.0, y=0.0,
            width=float(swap.extent.width), height=float(swap.extent.height),
            minDepth=0.0, maxDepth=1.0,
        )
        scissor = VkRect2D(offset=VkOffset2D(x=0, y=0), extent=swap.extent)
        vkCmdSetViewport(cmd, 0, 1, [viewport])
        vkCmdSetScissor(cmd, 0, 1, [scissor])

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, self.scene_pipeline.pipeline)

        aspect = swap.extent.width / max(1, swap.extent.height)
        proj = cam.perspective(aspect, getattr(camera, "fov", None))
        view = camera.view()
        vp = proj * view
        eye = camera.eye_position()
        # w = elapsed seconds: drives slow cloud drift in the sky shader.
        cam_bytes = struct.pack("4f", eye.x, eye.y, eye.z, glfw.get_time() % 100000.0)

        for gpu_mesh, model in items:
            mvp = vp * model
            push = cam.mat4_bytes(mvp) + cam.mat4_bytes(model) + cam_bytes
            vkCmdPushConstants(
                cmd, self.scene_pipeline.layout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, len(push),
                ffi.from_buffer(push),
            )
            vkCmdBindVertexBuffers(cmd, 0, 1, [gpu_mesh.vertex_buffer], [0])
            vkCmdBindIndexBuffer(cmd, gpu_mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32)
            vkCmdDrawIndexed(cmd, gpu_mesh.index_count, 1, 0, 0, 0)

        # Procedural sky: fullscreen triangle at depth 1.0, fills the pixels
        # no geometry covered (LESS_OR_EQUAL against the cleared depth).
        sky_push = cam.mat4_bytes(glm.inverse(vp)) + cam_bytes
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, self.sky_pipeline.pipeline)
        vkCmdPushConstants(
            cmd, self.sky_pipeline.layout,
            VK_SHADER_STAGE_VERTEX_BIT, 0, len(sky_push),
            ffi.from_buffer(sky_push),
        )
        vkCmdDraw(cmd, 3, 1, 0, 0)

        # HUD overlay (same render pass, alpha-blended, drawn on top).
        text_count = self._text_counts[self.current_frame]
        if text_count > 0:
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, self.text_pipeline.pipeline)
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, self.text_pipeline.layout,
                0, 1, [self.text_descriptor_set], 0, None,
            )
            vkCmdBindVertexBuffers(cmd, 0, 1, [self._text_buffers[self.current_frame]], [0])
            vkCmdDraw(cmd, text_count, 1, 0, 0)

        vkCmdEndRenderPass(cmd)
        vkEndCommandBuffer(cmd)

    # ------------------------------------------------------------ teardown
    def destroy(self):
        ctx = self.ctx
        vkDeviceWaitIdle(ctx.device)
        for buf in self._text_buffers:
            vkDestroyBuffer(ctx.device, buf, None)
        for mem in self._text_memories:
            vkFreeMemory(ctx.device, mem, None)
        self._destroy_present_semaphores()
        vkDestroyDescriptorPool(ctx.device, self._descriptor_pool, None)
        self.atlas_texture.destroy()
        self.text_pipeline.destroy()
        self.commands.destroy()
        self.sky_pipeline.destroy()
        self.scene_pipeline.destroy()
        self.swap.destroy()
