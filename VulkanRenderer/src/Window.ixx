module;
// IMPORTANT: Without the define underneath, it will not link.
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <vector>
export module Window;

import Validation;

export class Window {
public:
	Window(uint32_t width, uint32_t height) {
		if (!glfwInitialized) InitGLFW();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindow = glfwCreateWindow(width, height, "Vulkan Test", nullptr, nullptr);
		++windowCount;
	}
	~Window() {
		glfwDestroyWindow(glfwWindow);
		--windowCount;
		if (windowCount == 0) glfwTerminate();
	}

private:
	inline static uint32_t windowCount = 0;
	inline static bool glfwInitialized = false;
	GLFWwindow* glfwWindow;

public:
	static inline void InitGLFW() {
		if (glfwInit() != GLFW_TRUE) {
			throw std::runtime_error("GLFW could not init!");
		}
		glfwInitialized = true;
	}
	static void Poll() {
		glfwPollEvents();
	}

	const bool CloseRequested() const noexcept {
		return glfwWindowShouldClose(glfwWindow);
	}

	const std::vector<const char*> GetRequiredVulkanExtensions() const {
		uint32_t extensionsCount = 0;

		const char** glfwExtensionNames;
		glfwExtensionNames = glfwGetRequiredInstanceExtensions(&extensionsCount);

		std::vector<const char*> extensions(glfwExtensionNames, glfwExtensionNames + extensionsCount);

		if (ENABLE_VALIDATION_LAYERS) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	const std::tuple<uint32_t, uint32_t> GetFramebufferSize() const noexcept {
		int bufWidth, bufHeight;
		glfwGetFramebufferSize(glfwWindow, &bufWidth, &bufHeight);
		return { static_cast<uint32_t>(bufWidth), static_cast<uint32_t>(bufWidth) };
	}

	vk::Result CreateSurface(const vk::Instance& instance, vk::SurfaceKHR& surface) {
		return (vk::Result)glfwCreateWindowSurface(instance, glfwWindow, nullptr, (VkSurfaceKHR*)&surface);
	}
};