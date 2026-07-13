"""Command pool, per-frame command buffers, and synchronization primitives."""

from __future__ import annotations

from vulkan import *  # noqa: F401,F403

from .. import config


class FrameSync:
    """Per-frame-in-flight command buffer + sync objects."""

    def __init__(self, ctx, command_pool):
        self.ctx = ctx
        alloc = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self.command_buffer = vkAllocateCommandBuffers(ctx.device, alloc)[0]

        sem_info = VkSemaphoreCreateInfo(sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO)
        fence_info = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, flags=VK_FENCE_CREATE_SIGNALED_BIT
        )
        self.image_available = vkCreateSemaphore(ctx.device, sem_info, None)
        self.render_finished = vkCreateSemaphore(ctx.device, sem_info, None)
        self.in_flight = vkCreateFence(ctx.device, fence_info, None)

    def destroy(self):
        ctx = self.ctx
        vkDestroySemaphore(ctx.device, self.image_available, None)
        vkDestroySemaphore(ctx.device, self.render_finished, None)
        vkDestroyFence(ctx.device, self.in_flight, None)


def run_single_time(ctx, record_fn):
    """Allocate a one-shot command buffer, run ``record_fn(cmd)``, submit, wait.

    Used for setup work like image layout transitions and staging copies.
    """
    pool_info = VkCommandPoolCreateInfo(
        sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        queueFamilyIndex=ctx.graphics_family,
    )
    pool = vkCreateCommandPool(ctx.device, pool_info, None)
    try:
        alloc = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vkAllocateCommandBuffers(ctx.device, alloc)[0]
        begin = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vkBeginCommandBuffer(cmd, begin)
        record_fn(cmd)
        vkEndCommandBuffer(cmd)

        submit = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        vkQueueSubmit(ctx.graphics_queue, 1, [submit], None)
        vkQueueWaitIdle(ctx.graphics_queue)
    finally:
        vkDestroyCommandPool(ctx.device, pool, None)


class CommandResources:
    def __init__(self, ctx):
        self.ctx = ctx
        pool_info = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=ctx.graphics_family,
        )
        self.command_pool = vkCreateCommandPool(ctx.device, pool_info, None)
        self.frames = [
            FrameSync(ctx, self.command_pool) for _ in range(config.MAX_FRAMES_IN_FLIGHT)
        ]

    def destroy(self):
        ctx = self.ctx
        for frame in self.frames:
            frame.destroy()
        self.frames = []
        vkDestroyCommandPool(ctx.device, self.command_pool, None)
