#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use anyhow::{anyhow, Result};
use log::*;
use std::{collections::HashSet, ffi::CStr, os::raw::c_void};
use thiserror::Error;
use vulkanalia::{
    bytecode::Bytecode,
    loader::{LibloadingLoader, LIBRARY},
    prelude::v1_0::*,
    vk::ExtDebugUtilsExtension,
    vk::KhrSurfaceExtension,
    vk::KhrSwapchainExtension,
    window as vk_window, Version,
};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

fn main() -> Result<()> {
    // Set rust log with below command for logging output.
    // RUST_LOG="debug"
    pretty_env_logger::init();

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Hi David!")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let mut app = unsafe { App::create(&window)? };
    event_loop.run(move |event, elwt| {
        match event {
            // Request a redraw when all events were processed.
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                // Render a frame if our Vulkan app is not being destroyed.
                WindowEvent::RedrawRequested if !elwt.exiting() => {
                    unsafe { app.render(&window) }.unwrap()
                }
                // Destroy our Vulkan app.
                WindowEvent::CloseRequested => {
                    elwt.exit();
                    unsafe {
                        app.destroy();
                    }
                }
                _ => {}
            },
            _ => {}
        }
    })?;

    Ok(())
}

#[derive(Clone, Debug)]
struct App {
    entry: Entry,
    vk_instance: Instance,
    data: AppData,
    device: Device,
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|e| anyhow!("{}", e))?;
        let mut data = AppData::default();

        let vk_instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&vk_instance, window, window)?;

        pick_physical_device(&vk_instance, &mut data)?;
        let device = create_logical_device(&entry, &vk_instance, &mut data)?;

        create_swapchain(window, &vk_instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;

        create_pipeline(&device, &mut data)?;

        Ok(Self {
            entry,
            vk_instance,
            data,
            device,
        })
    }

    /// Renders a frame for our Vulkan app.
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        Ok(())
    }

    /// Destroys our Vulkan app.
    unsafe fn destroy(&mut self) {
        if VALIDATION_ENABLED {
            self.vk_instance
                .destroy_debug_utils_messenger_ext(self.data.debug_messenger, None)
        }
        self.data
            .swapchain_image_views
            .iter()
            .for_each(|view| self.device.destroy_image_view(*view, None));

        self.device.destroy_swapchain_khr(self.data.swapchain, None);
        self.device.destroy_device(None);
        self.vk_instance
            .destroy_surface_khr(self.data.surface, None);
        self.vk_instance.destroy_instance(None);
    }
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct AppData {
    debug_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();
    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", msg_type, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", msg_type, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", msg_type, message);
    } else {
        trace!("({:?}) {}", msg_type, message);
    }
    vk::FALSE
}

//Pipe Line
unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    let vert = include_bytes!("../shaders/vert.spv");
    let frag = include_bytes!("../shaders/frag.spv");

    let vert_shader_module = create_shader_module(device, &vert[..])?;
    let frag_shader_module = create_shader_module(device, &frag[..])?;

    let vert_state = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(b"main\0");
    let frag_state = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");

    device.destroy_shader_module(vert_shader_module, None);
    device.destroy_shader_module(frag_shader_module, None);
    Ok(())
}

unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = match Bytecode::new(bytecode) {
        Ok(bytecode) => bytecode,
        Err(err) => {
            return Err(anyhow!(
                "Allocation of aligned buffer failed or byte slice is not multiple of 4."
            ))
        }
    };
    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(device.create_shader_module(&info, None)?)
}

// Queue
#[derive(Clone, Debug, Default)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;

        for (idx, properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                idx as u32,
                data.surface,
            )? {
                present = Some(idx as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
    }
}

// Swap chain
#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }

    fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        formats
            .iter()
            .cloned()
            .find(|val| {
                val.format == vk::Format::B8G8R8A8_SRGB
                    && val.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| formats[0])
    }

    fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        present_modes
            .iter()
            .cloned()
            .find(|mode| *mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }

    fn get_swapchain_extent(
        window: &Window,
        capabilities: vk::SurfaceCapabilitiesKHR,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D::builder()
                .width(window.inner_size().width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ))
                .height(window.inner_size().height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ))
                .build()
        }
    }
}

unsafe fn create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|img| {
            let components = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY);
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let info = vk::ImageViewCreateInfo::builder()
                .image(*img)
                .view_type(vk::ImageViewType::_2D)
                .format(data.swapchain_format)
                .components(components)
                .subresource_range(subresource_range);
            device.create_image_view(&info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    let surface_format = SwapchainSupport::get_swapchain_surface_format(&support.formats);
    let present_mode = SwapchainSupport::get_swapchain_present_mode(&support.present_modes);
    let extent = SwapchainSupport::get_swapchain_extent(window, support.capabilities);

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    data.swapchain = device.create_swapchain_khr(&info, None)?;
    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    Ok(())
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]

// Physical device
pub struct SuitabilityError(pub &'static str);

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|ext| ext.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|ext| extensions.contains(ext)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError(
            "Missing required device extensions."
        )))
    }
}

unsafe fn check_physical_device(
    instance: &Instance,
    data: &mut AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let properties = instance.get_physical_device_properties(physical_device);
    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if properties.device_type == vk::PhysicalDeviceType::OTHER {
        return Err(anyhow!(SuitabilityError("Unsupported Device Type.")));
    }

    if let Err(_) = QueueFamilyIndices::get(instance, &data, physical_device) {
        return Err(anyhow!(SuitabilityError(
            "Device does not support required commands."
        )));
    }

    if let Err(_) = check_physical_device_extensions(instance, physical_device) {
        return Err(anyhow!(SuitabilityError(
            "Device does not support required swapchain extensions."
        )));
    }

    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError(
            "Device does not support swap chain settings."
        )));
    }

    Ok(())
}

unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!(
                "Skipping device (' {} {}'): {}",
                properties.device_id, properties.device_name, error
            )
        } else {
            info!("Selected physical device (`{}`).", properties.device_name);
            data.physical_device = physical_device;
            return Ok(());
        }
    }
    Err(anyhow!("Could not find suitable physical device."))
}

unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let mut unique_indices = HashSet::new();
    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();

    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    // queue prioerity is between 1.0 and 0.0. Even with a single queue.
    let queue_priorities = &[1.0];

    // you can create all of the command buffers on multiple threads and then submit them
    // all at once on the main thread with a single low-overhead call.

    let queue_infos = unique_indices
        .iter()
        .map(|indc| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*indc)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    let features = vk::PhysicalDeviceFeatures::builder();

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(data.physical_device, &info, None)?;
    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.present_queue = device.get_device_queue(indices.present, 0);

    Ok(device)
}

// Instance
unsafe fn create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    let app_info = vk::ApplicationInfo::builder()
        .application_name(b"Hi David\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::with_capacity(25)
    };

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(
            vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                .name
                .as_ptr(),
        );
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    let info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    // Browse the Vulkan SDK and go to the Config directory.
    // There you will find a vk_layer_settings.txt file that explains how to configure the layers.

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;

    Ok(instance)
}
