#![allow(unused_imports, unused_variables)]
#![allow(dead_code)]

use std::any::{Any, TypeId};
use vulkanalia::prelude::v1_0::*;

pub trait RefineBounds {}
pub struct ColorAttachment;
impl RefineBounds for ColorAttachment {}

pub struct DepthStencilAttachment;
impl RefineBounds for DepthStencilAttachment {}

pub struct ColorResolveAttachment;
impl RefineBounds for ColorResolveAttachment {}

struct TriggerPanicTest;
impl RefineBounds for TriggerPanicTest {}

pub trait ExtendAttachmentDescription {
    fn typed_builder<T: 'static + Any + RefineBounds>(
        format: vk::Format,
        samples: Option<vk::SampleCountFlags>,
    ) -> vk::AttachmentDescriptionBuilder;
}

#[rustfmt::skip]
impl ExtendAttachmentDescription for vk::AttachmentDescriptionBuilder {
    fn typed_builder<T: 'static + Any + RefineBounds>(
        format: vk::Format,
        samples: Option<vk::SampleCountFlags>,
    ) -> vk::AttachmentDescriptionBuilder {
        match TypeId::of::<T>() {
            t @ _ if t == TypeId::of::<ColorAttachment>() => {
                vk::AttachmentDescription::builder()
                    .format(format)
                    .samples(samples.unwrap_or(vk::SampleCountFlags::_1))
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            },
            t @ _ if t == TypeId::of::<DepthStencilAttachment>() => {
                vk::AttachmentDescription::builder()
                    .format(format)
                    .samples(samples.unwrap_or(vk::SampleCountFlags::_1))
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            }
            t @ _ if t == TypeId::of::<ColorResolveAttachment>() => {
                vk::AttachmentDescription::builder()
                    .format(format)
                    .samples(samples.unwrap_or(vk::SampleCountFlags::_1))
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            }
            _ => panic!("Failed to build AttachmentDescriptionBu"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dyn_build() {
        let color_attachment = vk::AttachmentDescriptionBuilder::typed_builder::<ColorAttachment>(
            vk::Format::R8G8B8A8_SRGB,
            Some(vk::SampleCountFlags::_1),
        );
        let depth_stencil_attachment = vk::AttachmentDescriptionBuilder::typed_builder::<
            DepthStencilAttachment,
        >(
            vk::Format::R8G8B8A8_SRGB, Some(vk::SampleCountFlags::_1)
        );

        let color_resolve_attachment = vk::AttachmentDescriptionBuilder::typed_builder::<
            ColorResolveAttachment,
        >(vk::Format::R8G8B8A8_SRGB, None);

        assert_eq!(color_attachment.format, vk::Format::R8G8B8A8_SRGB);
        assert_eq!(color_attachment.samples, vk::SampleCountFlags::_1);
        assert_eq!(color_attachment.load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(color_attachment.store_op, vk::AttachmentStoreOp::STORE);
        assert_eq!(
            color_attachment.stencil_load_op,
            vk::AttachmentLoadOp::DONT_CARE
        );
        assert_eq!(
            color_attachment.stencil_store_op,
            vk::AttachmentStoreOp::DONT_CARE
        );
        assert_eq!(color_attachment.initial_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(
            color_attachment.final_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );

        // Break up the asserts so they are easier to read.
        assert_eq!(depth_stencil_attachment.format, vk::Format::R8G8B8A8_SRGB);
        assert_eq!(depth_stencil_attachment.samples, vk::SampleCountFlags::_1);
        assert_eq!(
            depth_stencil_attachment.load_op,
            vk::AttachmentLoadOp::CLEAR
        );
        assert_eq!(
            depth_stencil_attachment.store_op,
            vk::AttachmentStoreOp::DONT_CARE
        );
        assert_eq!(
            depth_stencil_attachment.stencil_load_op,
            vk::AttachmentLoadOp::DONT_CARE
        );
        assert_eq!(
            depth_stencil_attachment.stencil_store_op,
            vk::AttachmentStoreOp::DONT_CARE
        );
        assert_eq!(
            depth_stencil_attachment.initial_layout,
            vk::ImageLayout::UNDEFINED
        );
        assert_eq!(
            depth_stencil_attachment.final_layout,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );

        // Wow thats a lot of word soup.
        assert_eq!(color_resolve_attachment.format, vk::Format::R8G8B8A8_SRGB);
        assert_eq!(color_resolve_attachment.samples, vk::SampleCountFlags::_1);
        assert_eq!(
            color_resolve_attachment.load_op,
            vk::AttachmentLoadOp::DONT_CARE
        );
        assert_eq!(
            color_resolve_attachment.store_op,
            vk::AttachmentStoreOp::STORE
        );
        assert_eq!(
            color_resolve_attachment.stencil_load_op,
            vk::AttachmentLoadOp::DONT_CARE
        );
        assert_eq!(
            color_resolve_attachment.stencil_store_op,
            vk::AttachmentStoreOp::DONT_CARE
        );
        assert_eq!(
            color_resolve_attachment.initial_layout,
            vk::ImageLayout::UNDEFINED
        );
        assert_eq!(
            color_resolve_attachment.final_layout,
            vk::ImageLayout::PRESENT_SRC_KHR
        );
    }

    #[test]
    #[should_panic]
    fn unimplemented_type() {
        let color_attachment = vk::AttachmentDescriptionBuilder::typed_builder::<TriggerPanicTest>(
            vk::Format::R8G8B8A8_SRGB,
            Some(vk::SampleCountFlags::_1),
        );
    }
}
