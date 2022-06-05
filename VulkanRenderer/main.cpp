#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <exception>
#include <ranges>
#include <format>
#include <optional>
#include <string>
#include <set>
#include <limits>
#include <algorithm>
#include <fstream>
#include <span>
#include <map>

using namespace vk;

constexpr std::array<const char*, 1> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

constexpr std::array<const char*, 1> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool ENABLE_VALIDATION_LAYERS = false;
#else
constexpr bool ENABLE_VALIDATION_LAYERS = true;
#endif // NDEBUG


constexpr uint32_t WIDTH = 800, HEIGHT = 600;


GLFWwindow* window;

vk::Instance instance;
VkDebugUtilsMessengerEXT debugMessenger;
vk::SurfaceKHR surface;

vk::PhysicalDevice physicalDevice;
vk::Device device;

vk::Queue graphicsQueue;
vk::Queue presentQueue;

vk::SwapchainKHR swapchain;
std::vector<vk::Image> swapchainImages;
vk::Format swapchainImageFormat;
vk::Extent2D swapchainExtent;
std::vector<vk::ImageView> swapchainImageViews;
std::vector<vk::Framebuffer> swapchainFramebuffers;

vk::PipelineLayout pipelineLayout;
vk::RenderPass renderPass;
vk::Pipeline graphicsPipeline;

vk::CommandPool commandPool;
vk::CommandBuffer commandBuffer;

vk::Semaphore imageAvailableSemaphore;
vk::Semaphore renderFinishedSemaphore;
vk::Fence inFlightFence;

constexpr VkDebugUtilsMessageSeverityFlagBitsEXT VULKAN_LOG_LEVEL = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;

PFN_vkCreateDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT;
PFN_vkDestroyDebugUtilsMessengerEXT DestroyDebugUtilsMessengerEXT;

std::vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file!");
	}

	const auto len = static_cast<size_t>(file.tellg());
	std::vector<char> buf(len);

	file.seekg(0);
	file.read(buf.data(), len);
	return buf;
}

bool checkValidationLayerSupport() {
	auto availableLayers = vk::enumerateInstanceLayerProperties();

	for (auto xName : validationLayers)
	{
		bool found = false;
		for (auto& y : availableLayers)
		{
			auto yName = y.layerName;
			if (strcmp(xName, yName) == 0) {
				found = true;
				break;
			}
		}
		if (!found) return false;
	}
	return true;
}

std::vector<const char*> getRequiredExtensions() {
	uint32_t extensionsCount = 0;

	const char** glfwExtensionNames;
	glfwExtensionNames = glfwGetRequiredInstanceExtensions(&extensionsCount);

	std::vector<const char*> extensions(glfwExtensionNames, glfwExtensionNames + extensionsCount);

	if (ENABLE_VALIDATION_LAYERS) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData
) {
	if (messageSeverity < VULKAN_LOG_LEVEL) return VK_FALSE;
	std::cerr << "VALIDATION LAYER: " << pCallbackData->pMessage << std::endl;
	return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT createDebugMessengerCreateInfo() {
	VkDebugUtilsMessengerCreateInfoEXT createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo.messageSeverity =
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType =
		VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;
	createInfo.pUserData = nullptr;
	return createInfo;
}

void setupDebugMessenger() {
	if (!ENABLE_VALIDATION_LAYERS) return;

	CreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)instance.getProcAddr("vkCreateDebugUtilsMessengerEXT");
	DestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)instance.getProcAddr("vkDestroyDebugUtilsMessengerEXT");

	auto createInfo = createDebugMessengerCreateInfo();

	auto result = CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("Could not create vulkan debug messenger!");
	}
}

void init() {
	if (glfwInit() != GLFW_TRUE) {
		throw std::runtime_error("GLFW could not init!");
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Test", nullptr, nullptr);
}


void createInstance() {
	vk::ApplicationInfo appInfo{};
	appInfo.sType = vk::StructureType::eApplicationInfo;
	appInfo.pApplicationName = "Vulkan Test";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "No Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	vk::InstanceCreateInfo createInfo{};
	createInfo.sType = vk::StructureType::eInstanceCreateInfo;
	createInfo.pApplicationInfo = &appInfo;

	auto extensions = getRequiredExtensions();
	createInfo.enabledExtensionCount = extensions.size();
	createInfo.ppEnabledExtensionNames = extensions.data();

	auto debugCreateInfo = createDebugMessengerCreateInfo();
	if (ENABLE_VALIDATION_LAYERS) {
		if (!checkValidationLayerSupport()) {
			throw std::runtime_error("Valdation layers requested, but not available!");
		}
		createInfo.enabledLayerCount = validationLayers.size();
		createInfo.ppEnabledLayerNames = validationLayers.data();
		createInfo.pNext = &debugCreateInfo;
	}
	else {
		createInfo.enabledLayerCount = 0;
		createInfo.ppEnabledLayerNames = nullptr;
	}

	if (vk::createInstance(&createInfo, nullptr, &instance) != vk::Result::eSuccess) {
		std::cerr << "Could not create vulkan instance!";
		throw std::runtime_error("Could not create vulkan instance!");
	}
}


struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() const noexcept {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device) {
	QueueFamilyIndices indices;
	auto queueFamilies = device.getQueueFamilyProperties();
	int i = 0;
	for (const auto& family : queueFamilies)
	{
		if (family.queueFlags & vk::QueueFlagBits::eGraphics) {
			indices.graphicsFamily = i;
		}

		if (device.getSurfaceSupportKHR(i, surface)) {
			indices.presentFamily = i;
		}
		i++;
	}
	return indices;
}

bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device) {
	auto extensions = device.enumerateDeviceExtensionProperties();

	for (const auto requiedExtension : deviceExtensions)
	{
		bool containedCurrentReqExt = false;
		for (const auto& extension : extensions)
		{
			if (strcmp(extension.extensionName, requiedExtension) != 0)
				continue;
			containedCurrentReqExt = true;
			break;
		}
		if (!containedCurrentReqExt)
			return false;
	}
	return true;
}

struct SwapChainSupportDetails {
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;
};

SwapChainSupportDetails querySwapchainSupport(const vk::PhysicalDevice& device) {
	SwapChainSupportDetails details;
	details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
	details.formats = device.getSurfaceFormatsKHR(surface);
	details.presentModes = device.getSurfacePresentModesKHR(surface);
	return details;
}

using SortedDeviceMap = std::multimap<int, PhysicalDevice, std::greater<int>>;
const SortedDeviceMap scoreDevices(const std::span<PhysicalDevice> devices) {
	SortedDeviceMap sortedDevices;
	for (const auto& device : devices)
	{
		int points = 0;

		const auto swapChainSupport = querySwapchainSupport(device);
		const auto isSwapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		const auto indices = findQueueFamilies(device);
		const auto extensionsSupported = checkDeviceExtensionSupport(device);
		if (!indices.isComplete() || !extensionsSupported || !isSwapChainAdequate) continue;

		const auto deviceProps = device.getProperties();
		if (deviceProps.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
			points += 10000;

		const auto& deviceLimits = deviceProps.limits;
		points += deviceLimits.maxFramebufferHeight;
		points += deviceLimits.maxFramebufferWidth;

		sortedDevices.insert({ points, device });
	}
	return sortedDevices;
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
	for (const auto& availableFormat : availableFormats)
	{
		if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			return availableFormat;
		}
	}

	return availableFormats[0];
}

vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
	for (const auto& presentMode : availablePresentModes)
	{
		if (presentMode == vk::PresentModeKHR::eMailbox) {
			return presentMode;
		}
	}
	return vk::PresentModeKHR::eFifo;
}

vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilites) {
	if (capabilites.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		return capabilites.currentExtent;
	}

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	vk::Extent2D actualExtent = {
		static_cast<uint32_t>(width),
		static_cast<uint32_t>(height)
	};

	const auto minExtent = capabilites.minImageExtent;
	const auto maxExtent = capabilites.maxImageExtent;

	actualExtent.width = std::clamp(actualExtent.width, minExtent.width, maxExtent.width);
	actualExtent.height = std::clamp(actualExtent.height, minExtent.height, maxExtent.height);
	return actualExtent;
}

std::string getApiVersionString(uint32_t version) {
	auto variant = VK_API_VERSION_VARIANT(version);
	auto major = VK_API_VERSION_MAJOR(version);
	auto minor = VK_API_VERSION_MINOR(version);
	auto patch = VK_API_VERSION_PATCH(version);
	std::stringstream buf;
	buf << '(' << variant << ')';
	buf << major << '.' << minor << '.' << patch;
	return buf.str();
}

void pickPhysicalDevice() {
	vk::PhysicalDevice device = VK_NULL_HANDLE;

	auto devices = instance.enumeratePhysicalDevices();
	auto sortedDevices = scoreDevices(devices);
	auto sortedDevicePair = *sortedDevices.begin();
	device = sortedDevicePair.second;
	auto deviceProps = device.getProperties();
	std::cout << std::format("Picked suitable device '{}' with API version {}", deviceProps.deviceName, getApiVersionString(deviceProps.apiVersion)) << std::endl;
	physicalDevice = device;
}

void createLogicalDevice() {
	auto indices = findQueueFamilies(physicalDevice);

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };
	constexpr float queuePriority = 1.0f;
	for (auto& queueFamily : uniqueQueueFamilies)
	{
		vk::DeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = vk::StructureType::eDeviceQueueCreateInfo;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}

	vk::PhysicalDeviceFeatures deviceFeatures{};

	vk::DeviceCreateInfo createInfo{};
	createInfo.sType = vk::StructureType::eDeviceCreateInfo;
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	createInfo.queueCreateInfoCount = queueCreateInfos.size();
	createInfo.pEnabledFeatures = &deviceFeatures;
	createInfo.ppEnabledExtensionNames = deviceExtensions.data();
	createInfo.enabledExtensionCount = deviceExtensions.size();
	if (ENABLE_VALIDATION_LAYERS) {
		createInfo.enabledLayerCount = validationLayers.size();
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		createInfo.enabledLayerCount = 0;
	}

	if (physicalDevice.createDevice(&createInfo, nullptr, &device) != vk::Result::eSuccess) {
		throw std::runtime_error("Could not create logical device!");
	}

	device.getQueue(indices.graphicsFamily.value(), 0, &graphicsQueue);
	device.getQueue(indices.presentFamily.value(), 0, &presentQueue);
}

void createSurface() {
	if (glfwCreateWindowSurface(instance, window, nullptr, (VkSurfaceKHR*)&surface) != VK_SUCCESS) {
		throw std::runtime_error("Could not create window surface!");
	}
}

void createSwapchain() {
	const auto swapchainSupport = querySwapchainSupport(physicalDevice);

	const auto surfaceFormat = chooseSwapSurfaceFormat(swapchainSupport.formats);
	const auto presentMode = chooseSwapPresentMode(swapchainSupport.presentModes);
	const auto extent = chooseSwapExtent(swapchainSupport.capabilities);

	uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
	const auto maxImageCount = swapchainSupport.capabilities.maxImageCount;
	if (maxImageCount > 0 && imageCount > maxImageCount)
		imageCount = maxImageCount;

	vk::SwapchainCreateInfoKHR createInfo{};
	createInfo.sType = vk::StructureType::eSwapchainCreateInfoKHR;
	createInfo.surface = surface;
	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

	const auto indices = findQueueFamilies(physicalDevice);
	uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
	if (indices.graphicsFamily != indices.presentFamily) {
		createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else {
		createInfo.imageSharingMode = vk::SharingMode::eExclusive;
		createInfo.queueFamilyIndexCount = 0;
		createInfo.pQueueFamilyIndices = nullptr;
	}

	createInfo.preTransform = swapchainSupport.capabilities.currentTransform;
	createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;
	createInfo.oldSwapchain = nullptr;

	if (device.createSwapchainKHR(&createInfo, nullptr, &swapchain) != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to create swapchain!");
	}
	 
	swapchainImages = device.getSwapchainImagesKHR(swapchain);
	swapchainImageFormat = surfaceFormat.format;
	swapchainExtent = extent;
}

void createImageViews() {
	swapchainImageViews.resize(swapchainImages.size());
	for (size_t i = 0; i < swapchainImageViews.size(); i++)
	{
		vk::ImageViewCreateInfo createInfo{};
		createInfo.sType = vk::StructureType::eImageViewCreateInfo;
		createInfo.image = swapchainImages[i];
		createInfo.viewType = vk::ImageViewType::e2D;
		createInfo.format = swapchainImageFormat;

		createInfo.components.r = vk::ComponentSwizzle::eIdentity;
		createInfo.components.g = vk::ComponentSwizzle::eIdentity;
		createInfo.components.b = vk::ComponentSwizzle::eIdentity;
		createInfo.components.a = vk::ComponentSwizzle::eIdentity;

		createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		createInfo.subresourceRange.baseMipLevel = 0;
		createInfo.subresourceRange.levelCount = 1;
		createInfo.subresourceRange.baseArrayLayer = 0;
		createInfo.subresourceRange.layerCount = 1;

		if (device.createImageView(&createInfo, nullptr, &swapchainImageViews[i]) != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to create image views!");
		}
	}
}

vk::ShaderModule createShaderModule(const std::vector<char>& ir) {
	vk::ShaderModuleCreateInfo createInfo{};
	createInfo.sType = vk::StructureType::eShaderModuleCreateInfo;
	createInfo.codeSize = ir.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(ir.data());

	vk::ShaderModule shaderModule;
	if (device.createShaderModule(&createInfo, nullptr, &shaderModule) != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to create shader module!");
	}
	return shaderModule;
}

void createGraphicsPipeline() {
	const auto vertShaderIR = readFile("shaders/vert.spv");
	const auto fragShaderIR = readFile("shaders/frag.spv");

	const auto vertModule = createShaderModule(vertShaderIR);
	const auto fragModule = createShaderModule(fragShaderIR);

	vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
	vertShaderStageInfo.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
	vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
	vertShaderStageInfo.module = vertModule;
	vertShaderStageInfo.pName = "main";

	vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
	fragShaderStageInfo.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
	fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
	fragShaderStageInfo.module = fragModule;
	fragShaderStageInfo.pName = "main";

	vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = vk::StructureType::ePipelineVertexInputStateCreateInfo;
	vertexInputInfo.vertexBindingDescriptionCount = 0; 
	vertexInputInfo.vertexAttributeDescriptionCount = 0; 

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo;
	inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	vk::Viewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = swapchainExtent.width;
	viewport.height = swapchainExtent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	vk::Rect2D scissor{};
	scissor.offset = vk::Offset2D(0, 0);
	scissor.extent = swapchainExtent;

	vk::PipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = vk::StructureType::ePipelineViewportStateCreateInfo;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	vk::PipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = vk::StructureType::ePipelineRasterizationStateCreateInfo;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = vk::PolygonMode::eFill;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = vk::CullModeFlagBits::eBack;
	rasterizer.frontFace = vk::FrontFace::eClockwise;
	rasterizer.depthBiasEnable = VK_FALSE;

	vk::PipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = vk::StructureType::ePipelineMultisampleStateCreateInfo;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1; 

	vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	colorBlendAttachment.blendEnable = VK_FALSE;

	vk::PipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = vk::StructureType::ePipelineColorBlendStateCreateInfo;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = vk::LogicOp::eCopy;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment; 

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = vk::StructureType::ePipelineLayoutCreateInfo;
	pipelineLayoutInfo.setLayoutCount = 0;
	pipelineLayoutInfo.pushConstantRangeCount = 0;

	if (device.createPipelineLayout(&pipelineLayoutInfo, nullptr, &pipelineLayout) != vk::Result::eSuccess)
	{
		throw std::runtime_error("Failed to create pipeline layout!");
	}

	vk::GraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = vk::StructureType::eGraphicsPipelineCreateInfo;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling; 
	pipelineInfo.pColorBlendState = &colorBlending; 

	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = renderPass;
	pipelineInfo.subpass = 0;

	pipelineInfo.basePipelineHandle = nullptr; 

	auto result = device.createGraphicsPipeline(nullptr, pipelineInfo);
	if (result.result != vk::Result::eSuccess) {
		throw std::runtime_error("Could not create graphics pipeline!");
	}
	graphicsPipeline = result.value;

	device.destroyShaderModule(fragModule);
	device.destroyShaderModule(vertModule);
}

void createRenderPass() {
	vk::AttachmentDescription colorAttachment{};
	colorAttachment.format = swapchainImageFormat;
	colorAttachment.samples = vk::SampleCountFlagBits::e1;
	colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
	colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
	colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
	colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

	vk::AttachmentReference colorAttachmentRef{};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::SubpassDescription subpass{};
	subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	vk::SubpassDependency dependency{};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	dependency.srcAccessMask = vk::AccessFlagBits::eNone;
	dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

	vk::RenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = vk::StructureType::eRenderPassCreateInfo;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies = &dependency;
	if (device.createRenderPass(&renderPassInfo, nullptr, &renderPass) != vk::Result::eSuccess) {
		throw std::runtime_error("Could not create render pass!");
	}
}

void createFramebuffers() {
	swapchainFramebuffers.resize(swapchainImages.size());

	for (size_t i = 0; i < swapchainImageViews.size(); i++)
	{
		vk::ImageView attachments[] = {
			swapchainImageViews[i]
		};

		vk::FramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = StructureType::eFramebufferCreateInfo;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = swapchainExtent.width;
		framebufferInfo.height = swapchainExtent.height;
		framebufferInfo.layers = 1;
		if (device.createFramebuffer(&framebufferInfo, nullptr, &swapchainFramebuffers[i]) != vk::Result::eSuccess) {
			throw std::runtime_error("Could not create framebuffer!");
		}
	}
}

void createCommandPool() {
	auto queueFamilyIndices = findQueueFamilies(physicalDevice);

	vk::CommandPoolCreateInfo poolInfo{};
	poolInfo.sType = vk::StructureType::eCommandPoolCreateInfo;
	poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

	if (device.createCommandPool(&poolInfo, nullptr, &commandPool) != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to create command pool!");
	}

}

void createCommandBuffer() {
	vk::CommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = vk::StructureType::eCommandBufferAllocateInfo;
	allocInfo.commandPool = commandPool;
	allocInfo.level = vk::CommandBufferLevel::ePrimary;
	allocInfo.commandBufferCount = 1;

	if (device.allocateCommandBuffers(&allocInfo, &commandBuffer) != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to allocate command buffers!");
	}
}

void recordCommandBuffer(const vk::CommandBuffer& commandBuffer, uint32_t imageIndex) {
	vk::CommandBufferBeginInfo beginInfo{};
	beginInfo.sType = vk::StructureType::eCommandBufferBeginInfo;

	if (commandBuffer.begin(&beginInfo) != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to begin recording of command buffer!");
	}

	vk::RenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = vk::StructureType::eRenderPassBeginInfo;
	renderPassInfo.renderPass = renderPass;
	renderPassInfo.framebuffer = swapchainFramebuffers[imageIndex];
	renderPassInfo.renderArea.offset = vk::Offset2D(0, 0);
	renderPassInfo.renderArea.extent = swapchainExtent;

	vk::ClearValue clearColor = vk::ClearValue(vk::ClearColorValue(std::array{ 0.0f, 0.0f, 0.0f, 1.0f }));
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;

	commandBuffer.beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
	commandBuffer.draw(3, 1, 0, 0);
	commandBuffer.endRenderPass();
	commandBuffer.end();
}

void createSyncObjects() {
	vk::SemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = vk::StructureType::eSemaphoreCreateInfo;

	vk::FenceCreateInfo fenceInfo{};
	fenceInfo.sType = vk::StructureType::eFenceCreateInfo;
	fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

	if (device.createSemaphore(&semaphoreInfo, nullptr, &imageAvailableSemaphore) != vk::Result::eSuccess ||
		device.createSemaphore(&semaphoreInfo, nullptr, &renderFinishedSemaphore) != vk::Result::eSuccess ||
		device.createFence(&fenceInfo, nullptr, &inFlightFence) != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to create semaphore!");
	}
}

void drawFrame() {
	if (device.waitForFences(1, &inFlightFence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
		throw std::runtime_error("Failed at waiting for fence!");
	}
	if (device.resetFences(1, &inFlightFence) != vk::Result::eSuccess) {
		throw std::runtime_error("Could not reset fence!");
	}

	uint32_t imageIndex;
	auto acquireNextImgResult = device.acquireNextImageKHR(swapchain, UINT64_MAX, imageAvailableSemaphore, nullptr, &imageIndex);
	if (acquireNextImgResult != vk::Result::eSuccess) {
		throw std::runtime_error("Could not acquire next image!");
	}

	commandBuffer.reset();
	recordCommandBuffer(commandBuffer, imageIndex);

	vk::SubmitInfo submitInfo{};
	submitInfo.sType = vk::StructureType::eSubmitInfo;

	vk::Semaphore waitSemaphores[] = { imageAvailableSemaphore };
	vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vk::Semaphore signalSemaphores[] = { renderFinishedSemaphore };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;

	if (graphicsQueue.submit(1, &submitInfo, inFlightFence) != vk::Result::eSuccess) {
		throw std::runtime_error("Could not submit draw command buffer!");
	}

	vk::PresentInfoKHR presentInfo{};
	presentInfo.sType = vk::StructureType::ePresentInfoKHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	vk::SwapchainKHR swapchains[] = { swapchain };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapchains;
	presentInfo.pImageIndices = &imageIndex;
	presentInfo.pResults = nullptr;

	if (presentQueue.presentKHR(presentInfo) != vk::Result::eSuccess) {
		throw std::runtime_error("Could not present image!");
	}
}

void cleanup() {
	for (const auto& imageView : swapchainImageViews)
		device.destroyImageView(imageView);
	for (const auto& framebuffer : swapchainFramebuffers)
		device.destroyFramebuffer(framebuffer);
	device.destroySwapchainKHR(swapchain);
	device.destroyCommandPool(commandPool);
	device.destroyPipeline(graphicsPipeline);
	device.destroyPipelineLayout(pipelineLayout);
	device.destroyRenderPass(renderPass);
	device.destroySemaphore(imageAvailableSemaphore);
	device.destroySemaphore(renderFinishedSemaphore);
	device.destroyFence(inFlightFence);
	device.destroy();
	instance.destroySurfaceKHR(surface);
	if (ENABLE_VALIDATION_LAYERS) {
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
	}
	instance.destroy();

	glfwDestroyWindow(window);
	glfwTerminate();
}

int main() {
	init();
	createInstance();
	setupDebugMessenger();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createSwapchain();
	createImageViews();
	createRenderPass();
	createGraphicsPipeline();
	createFramebuffers();
	createCommandPool();
	createCommandBuffer();
	createSyncObjects();

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		drawFrame();
	}

	device.waitIdle();
	cleanup();
}