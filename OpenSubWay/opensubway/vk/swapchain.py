"""Swapchain, depth buffer, render pass and framebuffers.

Grouped in :class:`SwapchainBundle`, which is destroyed and recreated whenever
the window is resized.
"""

from __future__ import annotations

import glfw
from vulkan import *  # noqa: F401,F403

from .. import config
from . import memory


class SwapchainBundle:
    def __init__(self, ctx):
        self.ctx = ctx
        self.swapchain = None
        self.images = []
        self.image_views = []
        self.image_format = None
        self.extent = None
        self.depth_image = None
        self.depth_memory = None
        self.depth_view = None
        self.color_image = None      # MSAA color target (samples > 1)
        self.color_memory = None
        self.color_view = None
        self.render_pass = None
        self.framebuffers = []
        self.samples = ctx.msaa_samples

        # KHR entry points
        self._vkCreateSwapchainKHR = vkGetDeviceProcAddr(ctx.device, "vkCreateSwapchainKHR")
        self._vkDestroySwapchainKHR = vkGetDeviceProcAddr(ctx.device, "vkDestroySwapchainKHR")
        self._vkGetSwapchainImagesKHR = vkGetDeviceProcAddr(ctx.device, "vkGetSwapchainImagesKHR")
        self._vkGetSurfaceCaps = vkGetInstanceProcAddr(
            ctx.instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"
        )
        self._vkGetSurfaceFormats = vkGetInstanceProcAddr(
            ctx.instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"
        )
        self._vkGetPresentModes = vkGetInstanceProcAddr(
            ctx.instance, "vkGetPhysicalDeviceSurfacePresentModesKHR"
        )

        self.create()

    # ------------------------------------------------------------------ build
    def _choose_format(self):
        formats = self._vkGetSurfaceFormats(self.ctx.physical_device, self.ctx.surface)
        for f in formats:
            if (
                f.format == VK_FORMAT_B8G8R8A8_UNORM
                and f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            ):
                return f
        return formats[0]

    def _choose_present_mode(self):
        modes = self._vkGetPresentModes(self.ctx.physical_device, self.ctx.surface)
        if VK_PRESENT_MODE_MAILBOX_KHR in modes:
            return VK_PRESENT_MODE_MAILBOX_KHR
        return VK_PRESENT_MODE_FIFO_KHR  # always available

    def _choose_extent(self, caps):
        # Return an *independent* VkExtent2D. Returning caps.currentExtent
        # directly would alias memory owned by `caps`, which is freed once caps
        # is garbage-collected -> later reads yield 0.
        if caps.currentExtent.width != 0xFFFFFFFF:
            return VkExtent2D(
                width=caps.currentExtent.width, height=caps.currentExtent.height
            )
        w, h = glfw.get_framebuffer_size(self.ctx.window)
        w = max(caps.minImageExtent.width, min(caps.maxImageExtent.width, w))
        h = max(caps.minImageExtent.height, min(caps.maxImageExtent.height, h))
        return VkExtent2D(width=w, height=h)

    def create(self):
        ctx = self.ctx
        caps = self._vkGetSurfaceCaps(ctx.physical_device, ctx.surface)
        surface_format = self._choose_format()
        present_mode = self._choose_present_mode()
        extent = self._choose_extent(caps)

        image_count = caps.minImageCount + 1
        if caps.maxImageCount > 0:
            image_count = min(image_count, caps.maxImageCount)

        families = {ctx.graphics_family, ctx.present_family}
        if len(families) > 1:
            sharing = VK_SHARING_MODE_CONCURRENT
            indices = list(families)
        else:
            sharing = VK_SHARING_MODE_EXCLUSIVE
            indices = []

        info = VkSwapchainCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            surface=ctx.surface,
            minImageCount=image_count,
            imageFormat=surface_format.format,
            imageColorSpace=surface_format.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode=sharing,
            queueFamilyIndexCount=len(indices),
            pQueueFamilyIndices=indices,
            preTransform=caps.currentTransform,
            compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=present_mode,
            clipped=VK_TRUE,
            oldSwapchain=None,
        )
        self.swapchain = self._vkCreateSwapchainKHR(ctx.device, info, None)
        self.image_format = surface_format.format
        self.extent = extent

        self.images = self._vkGetSwapchainImagesKHR(ctx.device, self.swapchain)
        self.image_views = [
            memory.create_image_view(ctx, img, self.image_format, VK_IMAGE_ASPECT_COLOR_BIT)
            for img in self.images
        ]

        if self.samples > VK_SAMPLE_COUNT_1_BIT:
            self._create_color_resources()
        self._create_depth_resources()
        self._create_render_pass()
        self._create_framebuffers()

    def _create_color_resources(self):
        """Multisampled color target that resolves into the swapchain image."""
        ctx = self.ctx
        self.color_image, self.color_memory = memory.create_image(
            ctx,
            self.extent.width,
            self.extent.height,
            self.image_format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            samples=self.samples,
        )
        self.color_view = memory.create_image_view(
            ctx, self.color_image, self.image_format, VK_IMAGE_ASPECT_COLOR_BIT
        )

    def _create_depth_resources(self):
        ctx = self.ctx
        self.depth_image, self.depth_memory = memory.create_image(
            ctx,
            self.extent.width,
            self.extent.height,
            ctx.depth_format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            samples=self.samples,
        )
        self.depth_view = memory.create_image_view(
            ctx, self.depth_image, ctx.depth_format, VK_IMAGE_ASPECT_DEPTH_BIT
        )

    def _create_render_pass(self):
        ctx = self.ctx
        msaa = self.samples > VK_SAMPLE_COUNT_1_BIT

        # Attachment 0 = color (MSAA target, or the swapchain image when 1x).
        color = VkAttachmentDescription(
            format=self.image_format,
            samples=self.samples,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            if msaa else VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )
        # Attachment 1 = depth.
        depth = VkAttachmentDescription(
            format=ctx.depth_format,
            samples=self.samples,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )
        color_ref = VkAttachmentReference(
            attachment=0, layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )
        depth_ref = VkAttachmentReference(
            attachment=1, layout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )

        attachments = [color, depth]
        resolve_refs = None
        if msaa:
            # Attachment 2 = single-sample resolve target = the swapchain image.
            resolve = VkAttachmentDescription(
                format=self.image_format,
                samples=VK_SAMPLE_COUNT_1_BIT,
                loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                storeOp=VK_ATTACHMENT_STORE_OP_STORE,
                stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
                finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            )
            attachments.append(resolve)
            resolve_refs = [
                VkAttachmentReference(
                    attachment=2, layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                )
            ]

        subpass = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[color_ref],
            pResolveAttachments=resolve_refs,
            pDepthStencilAttachment=[depth_ref],
        )
        dependency = VkSubpassDependency(
            srcSubpass=VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            srcAccessMask=0,
            dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        )
        info = VkRenderPassCreateInfo(
            sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=len(attachments),
            pAttachments=attachments,
            subpassCount=1,
            pSubpasses=[subpass],
            dependencyCount=1,
            pDependencies=[dependency],
        )
        self.render_pass = vkCreateRenderPass(ctx.device, info, None)

    def _create_framebuffers(self):
        ctx = self.ctx
        msaa = self.samples > VK_SAMPLE_COUNT_1_BIT
        self.framebuffers = []
        for view in self.image_views:
            if msaa:
                attachments = [self.color_view, self.depth_view, view]
            else:
                attachments = [view, self.depth_view]
            info = VkFramebufferCreateInfo(
                sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=self.render_pass,
                attachmentCount=len(attachments),
                pAttachments=attachments,
                width=self.extent.width,
                height=self.extent.height,
                layers=1,
            )
            self.framebuffers.append(vkCreateFramebuffer(ctx.device, info, None))

    # --------------------------------------------------------------- teardown
    def destroy(self):
        ctx = self.ctx
        for fb in self.framebuffers:
            vkDestroyFramebuffer(ctx.device, fb, None)
        self.framebuffers = []
        if self.color_view:
            vkDestroyImageView(ctx.device, self.color_view, None)
            self.color_view = None
        if self.color_image:
            vkDestroyImage(ctx.device, self.color_image, None)
            self.color_image = None
        if self.color_memory:
            vkFreeMemory(ctx.device, self.color_memory, None)
            self.color_memory = None
        if self.depth_view:
            vkDestroyImageView(ctx.device, self.depth_view, None)
        if self.depth_image:
            vkDestroyImage(ctx.device, self.depth_image, None)
        if self.depth_memory:
            vkFreeMemory(ctx.device, self.depth_memory, None)
        if self.render_pass:
            vkDestroyRenderPass(ctx.device, self.render_pass, None)
        for view in self.image_views:
            vkDestroyImageView(ctx.device, view, None)
        self.image_views = []
        if self.swapchain:
            self._vkDestroySwapchainKHR(ctx.device, self.swapchain, None)
            self.swapchain = None
