"""Vulkan instance, debug messenger, surface, device and queues.

Encapsulated in :class:`VulkanContext`, which owns the long-lived Vulkan
objects that do not depend on the swapchain size.
"""

from __future__ import annotations

import ctypes
import sys

import glfw
from vulkan import *  # noqa: F401,F403  (Vulkan API: constants, structs, functions)
from vulkan import ffi

from .. import config

VALIDATION_LAYER = "VK_LAYER_KHRONOS_validation"
DEVICE_EXTENSIONS = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]


def _str(value) -> str:
    """Decode a Vulkan char[] field to a Python string (str/bytes/cdata)."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return ffi.string(value).decode("utf-8")


def _module_handle() -> int:
    """Return the Win32 HINSTANCE for this process (as an int)."""
    k32 = ctypes.windll.kernel32
    k32.GetModuleHandleW.restype = ctypes.c_void_p
    k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    return k32.GetModuleHandleW(None)


class VulkanContext:
    def __init__(self, window):
        self.window = window
        self.instance = None
        self.debug_callback = None          # keep python fn alive
        self._debug_messenger = None
        self.surface = None
        self.physical_device = None
        self.device = None
        self.graphics_family = None
        self.present_family = None
        self.graphics_queue = None
        self.present_queue = None
        self.depth_format = None

        self._validation = config.ENABLE_VALIDATION and self._validation_supported()

        self._create_instance()
        if self._validation:
            self._create_debug_callback()
        self._create_surface()
        self._pick_physical_device()
        self.msaa_samples = self._max_usable_sample_count()
        self._create_logical_device()
        self.depth_format = self._find_depth_format()

    # ------------------------------------------------------------------ init
    def _validation_supported(self) -> bool:
        layers = vkEnumerateInstanceLayerProperties()
        names = {_str(l.layerName) for l in layers}
        if VALIDATION_LAYER not in names:
            print("WARNING: validation layer not available; continuing without it.")
            return False
        return True

    def _create_instance(self):
        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=config.WINDOW_TITLE,
            applicationVersion=VK_MAKE_VERSION(0, 1, 0),
            pEngineName="OpenSubWay",
            engineVersion=VK_MAKE_VERSION(0, 1, 0),
            apiVersion=VK_API_VERSION_1_0,
        )

        extensions = list(glfw.get_required_instance_extensions())
        layers = []
        if self._validation:
            extensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)
            layers.append(VALIDATION_LAYER)

        create_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
        )
        self.instance = vkCreateInstance(create_info, None)

    def _create_debug_callback(self):
        def _cb(*args):
            # args: (flags, objType, obj, location, code, layerPrefix, message, userData)
            try:
                message = _str(args[6]) if isinstance(args[6], ffi.CData) else args[6]
                prefix = _str(args[5]) if isinstance(args[5], ffi.CData) else args[5]
            except Exception:
                prefix, message = args[5], args[6]
            print(f"[VK] {prefix}: {message}", file=sys.stderr)
            return 0

        self.debug_callback = _cb
        create_info = VkDebugReportCallbackCreateInfoEXT(
            sType=VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT
            | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
            pfnCallback=_cb,
        )
        func = vkGetInstanceProcAddr(self.instance, "vkCreateDebugReportCallbackEXT")
        if func:
            self._debug_messenger = func(self.instance, create_info, None)

    def _create_surface(self):
        hwnd = glfw.get_win32_window(self.window)
        create_info = VkWin32SurfaceCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            hinstance=ffi.cast("void *", _module_handle()),
            hwnd=ffi.cast("void *", hwnd),
        )
        func = vkGetInstanceProcAddr(self.instance, "vkCreateWin32SurfaceKHR")
        self.surface = func(self.instance, create_info, None)

    def _find_queue_families(self, device):
        graphics = present = None
        families = vkGetPhysicalDeviceQueueFamilyProperties(device)
        get_support = vkGetInstanceProcAddr(
            self.instance, "vkGetPhysicalDeviceSurfaceSupportKHR"
        )
        for i, fam in enumerate(families):
            if fam.queueCount > 0 and (fam.queueFlags & VK_QUEUE_GRAPHICS_BIT):
                if graphics is None:
                    graphics = i
            supported = get_support(device, i, self.surface)
            if supported and present is None:
                present = i
        return graphics, present

    def _device_supports_extensions(self, device) -> bool:
        props = vkEnumerateDeviceExtensionProperties(device, None)
        names = {_str(p.extensionName) for p in props}
        return all(ext in names for ext in DEVICE_EXTENSIONS)

    def _pick_physical_device(self):
        devices = vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError("No Vulkan-capable GPU found.")

        best = None
        for dev in devices:
            g, p = self._find_queue_families(dev)
            if g is None or p is None:
                continue
            if not self._device_supports_extensions(dev):
                continue
            props = vkGetPhysicalDeviceProperties(dev)
            score = 1000 if props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else 100
            if best is None or score > best[0]:
                best = (score, dev, g, p, props)

        if best is None:
            raise RuntimeError("No suitable Vulkan device (needs graphics + present + swapchain).")

        _, self.physical_device, self.graphics_family, self.present_family, props = best
        print(f"Using GPU: {_str(props.deviceName)}")

    def _create_logical_device(self):
        unique_families = {self.graphics_family, self.present_family}
        queue_infos = []
        for fam in unique_families:
            queue_infos.append(
                VkDeviceQueueCreateInfo(
                    sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex=fam,
                    queueCount=1,
                    pQueuePriorities=[1.0],
                )
            )

        features = VkPhysicalDeviceFeatures()

        # Device layers are deprecated (ignored since Vulkan 1.0); only instance
        # layers matter, so we don't pass any here.
        create_info = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=len(queue_infos),
            pQueueCreateInfos=queue_infos,
            enabledLayerCount=0,
            enabledExtensionCount=len(DEVICE_EXTENSIONS),
            ppEnabledExtensionNames=DEVICE_EXTENSIONS,
            pEnabledFeatures=[features],
        )
        self.device = vkCreateDevice(self.physical_device, create_info, None)
        self.graphics_queue = vkGetDeviceQueue(self.device, self.graphics_family, 0)
        self.present_queue = vkGetDeviceQueue(self.device, self.present_family, 0)

    def _max_usable_sample_count(self):
        """Clamp the desired MSAA level to what the device supports for
        both colour and depth attachments."""
        props = vkGetPhysicalDeviceProperties(self.physical_device)
        counts = (
            props.limits.framebufferColorSampleCounts
            & props.limits.framebufferDepthSampleCounts
        )
        for bit in (
            VK_SAMPLE_COUNT_8_BIT,
            VK_SAMPLE_COUNT_4_BIT,
            VK_SAMPLE_COUNT_2_BIT,
        ):
            if (counts & bit) and bit <= config.MSAA_SAMPLES:
                print(f"MSAA: {bit}x")
                return bit
        print("MSAA: disabled (1x)")
        return VK_SAMPLE_COUNT_1_BIT

    def _find_depth_format(self):
        candidates = [
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
        ]
        for fmt in candidates:
            props = vkGetPhysicalDeviceFormatProperties(self.physical_device, fmt)
            if props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT:
                return fmt
        raise RuntimeError("No supported depth format found.")

    # --------------------------------------------------------------- teardown
    def destroy(self):
        if self.device:
            vkDestroyDevice(self.device, None)
            self.device = None
        if self._debug_messenger:
            func = vkGetInstanceProcAddr(self.instance, "vkDestroyDebugReportCallbackEXT")
            if func:
                func(self.instance, self._debug_messenger, None)
            self._debug_messenger = None
        if self.surface:
            func = vkGetInstanceProcAddr(self.instance, "vkDestroySurfaceKHR")
            func(self.instance, self.surface, None)
            self.surface = None
        if self.instance:
            vkDestroyInstance(self.instance, None)
            self.instance = None
