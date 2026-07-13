"""Buffer / image allocation helpers built on a :class:`VulkanContext`."""

from __future__ import annotations

from vulkan import *  # noqa: F401,F403
from vulkan import ffi


def find_memory_type(ctx, type_filter, properties):
    mem_props = vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)
    for i in range(mem_props.memoryTypeCount):
        if (type_filter & (1 << i)) and (
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties
        ):
            return i
    raise RuntimeError("Failed to find a suitable memory type.")


def create_buffer(ctx, size, usage, properties):
    """Create a buffer + allocate/bind memory. Returns (buffer, memory)."""
    info = VkBufferCreateInfo(
        sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=size,
        usage=usage,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    )
    buffer = vkCreateBuffer(ctx.device, info, None)

    req = vkGetBufferMemoryRequirements(ctx.device, buffer)
    alloc = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=req.size,
        memoryTypeIndex=find_memory_type(ctx, req.memoryTypeBits, properties),
    )
    memory = vkAllocateMemory(ctx.device, alloc, None)
    vkBindBufferMemory(ctx.device, buffer, memory, 0)
    return buffer, memory


def upload_to_memory(ctx, memory, data: bytes, offset: int = 0):
    """Map host-visible memory and copy ``data`` in."""
    size = len(data)
    ptr = vkMapMemory(ctx.device, memory, offset, size, 0)
    ffi.memmove(ptr, data, size)
    vkUnmapMemory(ctx.device, memory)


def create_host_buffer(ctx, data: bytes, usage):
    """Create a HOST_VISIBLE|COHERENT buffer initialised with ``data``."""
    size = len(data)
    buffer, memory = create_buffer(
        ctx,
        size,
        usage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )
    upload_to_memory(ctx, memory, data)
    return buffer, memory


def create_image(ctx, width, height, fmt, tiling, usage, properties,
                 samples=VK_SAMPLE_COUNT_1_BIT):
    """Create a 2D image + memory. Returns (image, memory)."""
    info = VkImageCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        imageType=VK_IMAGE_TYPE_2D,
        extent=VkExtent3D(width=width, height=height, depth=1),
        mipLevels=1,
        arrayLayers=1,
        format=fmt,
        tiling=tiling,
        initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        usage=usage,
        samples=samples,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    )
    image = vkCreateImage(ctx.device, info, None)

    req = vkGetImageMemoryRequirements(ctx.device, image)
    alloc = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=req.size,
        memoryTypeIndex=find_memory_type(ctx, req.memoryTypeBits, properties),
    )
    memory = vkAllocateMemory(ctx.device, alloc, None)
    vkBindImageMemory(ctx.device, image, memory, 0)
    return image, memory


def create_image_view(ctx, image, fmt, aspect):
    info = VkImageViewCreateInfo(
        sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        image=image,
        viewType=VK_IMAGE_VIEW_TYPE_2D,
        format=fmt,
        components=VkComponentMapping(
            r=VK_COMPONENT_SWIZZLE_IDENTITY,
            g=VK_COMPONENT_SWIZZLE_IDENTITY,
            b=VK_COMPONENT_SWIZZLE_IDENTITY,
            a=VK_COMPONENT_SWIZZLE_IDENTITY,
        ),
        subresourceRange=VkImageSubresourceRange(
            aspectMask=aspect,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        ),
    )
    return vkCreateImageView(ctx.device, info, None)
