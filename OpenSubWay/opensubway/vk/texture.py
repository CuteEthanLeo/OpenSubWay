"""Single-channel (R8) texture upload for the HUD font atlas."""

from __future__ import annotations

from vulkan import *  # noqa: F401,F403

from . import memory
from .commands import run_single_time


class Texture:
    """An R8_UNORM sampled image (used for the font atlas)."""

    def __init__(self, ctx, width: int, height: int, data: bytes):
        self.ctx = ctx
        self.width = width
        self.height = height

        fmt = VK_FORMAT_R8_UNORM

        # Staging buffer with the pixel data.
        staging, staging_mem = memory.create_host_buffer(
            ctx, data, VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        )

        self.image, self.memory = memory.create_image(
            ctx, width, height, fmt,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )

        def record(cmd):
            self._transition(
                cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            )
            region = VkBufferImageCopy(
                bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                imageSubresource=VkImageSubresourceLayers(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel=0, baseArrayLayer=0, layerCount=1,
                ),
                imageOffset=VkOffset3D(x=0, y=0, z=0),
                imageExtent=VkExtent3D(width=width, height=height, depth=1),
            )
            vkCmdCopyBufferToImage(
                cmd, staging, self.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, [region],
            )
            self._transition(
                cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            )

        run_single_time(ctx, record)

        vkDestroyBuffer(ctx.device, staging, None)
        vkFreeMemory(ctx.device, staging_mem, None)

        self.view = memory.create_image_view(ctx, self.image, fmt, VK_IMAGE_ASPECT_COLOR_BIT)
        self.sampler = self._create_sampler()

    def _transition(self, cmd, old, new, src_access, dst_access, src_stage, dst_stage):
        barrier = VkImageMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            oldLayout=old, newLayout=new,
            srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            image=self.image,
            subresourceRange=VkImageSubresourceRange(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
            ),
            srcAccessMask=src_access, dstAccessMask=dst_access,
        )
        vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, None, 0, None, 1, [barrier])

    def _create_sampler(self):
        info = VkSamplerCreateInfo(
            sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            magFilter=VK_FILTER_LINEAR,
            minFilter=VK_FILTER_LINEAR,
            addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            anisotropyEnable=VK_FALSE,
            borderColor=VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
            unnormalizedCoordinates=VK_FALSE,
            compareEnable=VK_FALSE,
            mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR,
        )
        return vkCreateSampler(self.ctx.device, info, None)

    def destroy(self):
        ctx = self.ctx
        vkDestroySampler(ctx.device, self.sampler, None)
        vkDestroyImageView(ctx.device, self.view, None)
        vkDestroyImage(ctx.device, self.image, None)
        vkFreeMemory(ctx.device, self.memory, None)
