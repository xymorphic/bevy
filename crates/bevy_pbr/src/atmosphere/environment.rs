use crate::{
    generate::{EnvironmentMapUnchanged, GeneratorPipelines},
    resources::{
        AtmosphereLutPipelines, AtmosphereSampler, AtmosphereTextures, AtmosphereTransform,
        AtmosphereTransforms, AtmosphereTransformsOffset, GpuAtmosphere,
    },
    ExtractedAtmosphere, ExtractedDirectionalLight, GpuAtmosphereSettings, GpuLights,
    GpuScatteringMedium, LightMeta, ViewLightsUniformOffset,
};
use bevy_asset::{load_embedded_asset, AssetId, AssetServer, Assets, Handle, RenderAssetUsages};
use bevy_color::ColorToComponents;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Has, With, Without},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
    template::FromTemplate,
};
use bevy_image::Image;
use bevy_light::{
    atmosphere::ScatteringMedium, AtmosphereEnvironmentMapLight, GeneratedEnvironmentMapLight,
};
use bevy_math::{Mat4, Quat, UVec2, Vec3};
use bevy_render::{
    extract_component::{ComponentUniforms, DynamicUniformIndex, ExtractComponent},
    render_asset::{ExtractedAssets, RenderAssets},
    render_resource::{binding_types::*, *},
    renderer::{RenderContext, RenderDevice, ViewQuery},
    texture::{CachedTexture, GpuImage},
    view::{ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms},
};
use bevy_utils::default;
use tracing::warn;

// Render world representation of an environment map light for the atmosphere
#[derive(Component, ExtractComponent, Clone, FromTemplate)]
pub struct AtmosphereEnvironmentMap {
    pub environment_map: Handle<Image>,
    pub size: UVec2,
}

#[derive(Component)]
pub struct AtmosphereProbeTextures {
    pub environment: TextureView,
    pub transmittance_lut: CachedTexture,
    pub multiscattering_lut: CachedTexture,
    pub sky_view_lut: CachedTexture,
    pub aerial_view_lut: CachedTexture,
}

#[derive(Component)]
pub(crate) struct AtmosphereProbeBindGroups {
    pub environment: BindGroup,
}

#[derive(Resource)]
pub struct AtmosphereProbeLayouts {
    pub environment: BindGroupLayoutDescriptor,
}

#[derive(Resource)]
pub struct AtmosphereProbePipeline {
    pub environment: CachedComputePipelineId,
}

pub fn init_atmosphere_probe_layout(mut commands: Commands) {
    let environment = BindGroupLayoutDescriptor::new(
        "environment_bind_group_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                // uniforms
                (0, uniform_buffer::<GpuAtmosphere>(true)),
                (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                (2, uniform_buffer::<AtmosphereTransform>(true)),
                (3, uniform_buffer::<ViewUniform>(true)),
                (4, uniform_buffer::<GpuLights>(true)),
                // atmosphere luts and sampler
                (8, texture_2d(TextureSampleType::default())), // transmittance
                (9, texture_2d(TextureSampleType::default())), // multiscattering
                (10, texture_2d(TextureSampleType::default())), // sky view
                (11, texture_3d(TextureSampleType::default())), // aerial view
                (12, sampler(SamplerBindingType::Filtering)),
                // output 2D array texture
                (
                    13,
                    texture_storage_2d_array(
                        TextureFormat::Rgba16Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                ),
            ),
        ),
    );

    commands.insert_resource(AtmosphereProbeLayouts { environment });
}

pub(super) fn prepare_atmosphere_probe_bind_groups(
    probes: Query<(Entity, &AtmosphereProbeTextures), With<AtmosphereEnvironmentMap>>,
    render_device: Res<RenderDevice>,
    layouts: Res<AtmosphereProbeLayouts>,
    atmosphere_sampler: Res<AtmosphereSampler>,
    view_uniforms: Res<ViewUniforms>,
    lights_uniforms: Res<LightMeta>,
    atmosphere_transforms: Res<AtmosphereTransforms>,
    atmosphere_uniforms: Res<ComponentUniforms<GpuAtmosphere>>,
    settings_uniforms: Res<ComponentUniforms<GpuAtmosphereSettings>>,
    pipeline_cache: Res<PipelineCache>,
    mut commands: Commands,
) {
    // The transforms exist only for active views, so a probe on an inactive camera waits.
    let (Some(atmosphere), Some(settings), Some(transforms), Some(view), Some(lights)) = (
        atmosphere_uniforms.binding(),
        settings_uniforms.binding(),
        atmosphere_transforms.uniforms().binding(),
        view_uniforms.uniforms.binding(),
        lights_uniforms.view_gpu_lights.binding(),
    ) else {
        return;
    };
    for (entity, textures) in &probes {
        let environment = render_device.create_bind_group(
            "environment_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.environment),
            &BindGroupEntries::with_indices((
                // uniforms
                (0, atmosphere.clone()),
                (1, settings.clone()),
                (2, transforms.clone()),
                (3, view.clone()),
                (4, lights.clone()),
                // atmosphere luts and sampler
                (8, &textures.transmittance_lut.default_view),
                (9, &textures.multiscattering_lut.default_view),
                (10, &textures.sky_view_lut.default_view),
                (11, &textures.aerial_view_lut.default_view),
                (12, &**atmosphere_sampler),
                // output 2D array texture
                (13, &textures.environment),
            )),
        );

        commands
            .entity(entity)
            .insert(AtmosphereProbeBindGroups { environment });
    }
}

pub(super) fn prepare_probe_textures(
    view_textures: Query<&AtmosphereTextures, With<ExtractedAtmosphere>>,
    probes: Query<
        (
            Entity,
            &AtmosphereEnvironmentMap,
            Option<&AtmosphereTextures>,
        ),
        (
            With<AtmosphereEnvironmentMap>,
            Without<AtmosphereProbeTextures>,
        ),
    >,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut commands: Commands,
) {
    for (probe, render_env_map, own_textures) in &probes {
        let environment = gpu_images.get(&render_env_map.environment_map).unwrap();
        // create a cube view
        let environment_view = environment.texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        // A probe on a camera uses that camera's own sky. Other probes borrow the first view's.
        if let Some(view_textures) = own_textures.or_else(|| view_textures.iter().next()) {
            commands.entity(probe).insert(AtmosphereProbeTextures {
                environment: environment_view,
                transmittance_lut: view_textures.transmittance_lut.clone(),
                multiscattering_lut: view_textures.multiscattering_lut.clone(),
                sky_view_lut: view_textures.sky_view_lut.clone(),
                aerial_view_lut: view_textures.aerial_view_lut.clone(),
            });
        }
    }
}

/// How many frames a view draws its sky, and filters its probe into light, after its sky
/// changed. A changed medium may reach the GPU a frame late, so a few frames make sure that
/// the view holds the new sky.
const SKY_SETTLE_FRAMES: u8 = 3;
/// How far the camera moves up or down, in meters, before its view draws the sky again.
const SKY_HEIGHT_STEP: f32 = 10.0;
/// How far the camera moves along the ground, in meters, before its view draws the sky again.
/// The surface of an Earth-sized planet turns by about 0.005° over that distance.
const SKY_GROUND_STEP: f32 = 500.0;
/// How far a light turns, in radians, before a view draws the sky again: 0.01°.
const SKY_TURN: f32 = 1.75e-4;
/// How much the light of a light changes, as a share, before a view draws the sky again.
const SKY_LIGHT_SHARE: f32 = 1e-3;

/// What the sky of a view was drawn from: its transmittance, multiscattering, and sky-view
/// LUTs, and the environment map of a probe on its camera. None of them follows the direction
/// that the camera looks in.
#[derive(Clone)]
struct DrawnSky {
    atmosphere: (f32, f32, Vec3, AssetId<ScatteringMedium>, Mat4),
    settings: GpuAtmosphereSettings,
    environment: Option<AssetId<Image>>,
    /// The camera in atmosphere space, from the center of the planet, in meters.
    camera: Vec3,
    /// Each directional light: the direction toward it, its light, and the size and the
    /// intensity of its sun disk.
    lights: Vec<(Vec3, Vec3, f32, f32)>,
}

impl DrawnSky {
    /// Whether this sky differs from `old` by enough to show.
    fn differs_from(&self, old: &Self) -> bool {
        let (height, old_height) = (self.camera.length(), old.camera.length());
        let along =
            (self.camera.normalize_or_zero() - old.camera.normalize_or_zero()).length() * height;
        self.atmosphere != old.atmosphere
            || self.settings != old.settings
            || self.environment != old.environment
            || (height - old_height).abs() > SKY_HEIGHT_STEP
            || along > SKY_GROUND_STEP
            || self.lights.len() != old.lights.len()
            || self.lights.iter().zip(&old.lights).any(|(light, old)| {
                (light.0 - old.0).length() > SKY_TURN
                    || (light.1 - old.1).length() > old.1.length() * SKY_LIGHT_SHARE
                    || light.2 != old.2
                    || light.3 != old.3
            })
    }
}

/// When a view draws its sky: its transmittance, multiscattering, and sky-view LUTs, and the
/// environment map of a probe on its camera; and when the light probe filters that map into
/// light. They are drawn only for a few frames after the sky changed (see [`DrawnSky`]), so a
/// still sky costs nothing. Filtering runs before the views draw in a frame, so it goes on one
/// frame longer than drawing.
#[derive(Component)]
pub struct AtmosphereSkyRefresh {
    sky: DrawnSky,
    draws: u8,
    filters: u8,
    /// Whether the view draws its sky this frame.
    pub draw: bool,
}

/// Decides for each view with an atmosphere whether it draws its sky this frame, and whether
/// the probe on its camera filters its light. Frames count down only once every pipeline that
/// the sky needs is ready.
pub(super) fn refresh_atmosphere_skies(
    mut commands: Commands,
    mut views: Query<(
        Entity,
        Option<&AtmosphereEnvironmentMap>,
        &ExtractedView,
        &ExtractedAtmosphere,
        &GpuAtmosphereSettings,
        Option<&mut AtmosphereSkyRefresh>,
    )>,
    lights: Query<(Entity, &ExtractedDirectionalLight)>,
    media: Option<Res<ExtractedAssets<GpuScatteringMedium>>>,
    pipeline_cache: Res<PipelineCache>,
    probe_pipeline: Res<AtmosphereProbePipeline>,
    lut_pipelines: Res<AtmosphereLutPipelines>,
    generator: Option<Res<GeneratorPipelines>>,
) {
    let mut suns: Vec<_> = lights.iter().collect();
    suns.sort_by_key(|(entity, _)| *entity);
    let suns: Vec<_> = suns
        .into_iter()
        .map(|(_, light)| {
            (
                Vec3::from(light.transform.back()),
                light.color.to_vec3() * light.illuminance,
                light.sun_disk_angular_size,
                light.sun_disk_intensity,
            )
        })
        .collect();
    let ready = [
        probe_pipeline.environment,
        lut_pipelines.transmittance_lut,
        lut_pipelines.multiscattering_lut,
        lut_pipelines.sky_view_lut,
        lut_pipelines.aerial_view_lut,
    ]
    .into_iter()
    .chain(generator.iter().flat_map(|generator| {
        [
            generator.downsample_first,
            generator.downsample_second,
            generator.copy,
            generator.radiance,
            generator.irradiance,
        ]
    }))
    .all(|id| pipeline_cache.get_compute_pipeline(id).is_some());
    for (entity, map, view, atmosphere, settings, refresh) in &mut views {
        let sky = DrawnSky {
            atmosphere: (
                atmosphere.inner_radius,
                atmosphere.outer_radius,
                atmosphere.ground_albedo,
                atmosphere.medium,
                atmosphere.world_to_atmosphere,
            ),
            settings: settings.clone(),
            environment: map.map(|map| map.environment_map.id()),
            camera: atmosphere
                .world_to_atmosphere
                .transform_point3(view.world_from_view.translation()),
            lights: suns.clone(),
        };
        let medium_changed = media.as_ref().is_some_and(|media| {
            media.added.contains(&atmosphere.medium) || media.modified.contains(&atmosphere.medium)
        });
        let Some(mut refresh) = refresh else {
            commands
                .entity(entity)
                .remove::<EnvironmentMapUnchanged>()
                .insert(AtmosphereSkyRefresh {
                    sky,
                    draws: SKY_SETTLE_FRAMES,
                    filters: SKY_SETTLE_FRAMES + 1,
                    draw: true,
                });
            continue;
        };
        if medium_changed || sky.differs_from(&refresh.sky) {
            refresh.sky = sky;
            refresh.draws = SKY_SETTLE_FRAMES;
            refresh.filters = SKY_SETTLE_FRAMES + 1;
        }
        refresh.draw = refresh.draws > 0;
        let filter = refresh.filters > 0;
        if ready {
            refresh.draws = refresh.draws.saturating_sub(1);
            refresh.filters = refresh.filters.saturating_sub(1);
        }
        if filter || map.is_none() {
            commands.entity(entity).remove::<EnvironmentMapUnchanged>();
        } else {
            commands.entity(entity).insert(EnvironmentMapUnchanged);
        }
    }
}

pub fn init_atmosphere_probe_pipeline(
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<AtmosphereProbeLayouts>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
) {
    let environment = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("environment_pipeline".into()),
        layout: vec![layouts.environment.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "environment.wgsl"),
        ..default()
    });
    commands.insert_resource(AtmosphereProbePipeline { environment });
}

// Ensure power-of-two dimensions to avoid edge update issues on cubemap faces
pub fn validate_environment_map_size(size: UVec2) -> UVec2 {
    let new_size = UVec2::new(
        size.x.max(1).next_power_of_two(),
        size.y.max(1).next_power_of_two(),
    );
    if new_size != size {
        warn!(
            "Non-power-of-two AtmosphereEnvironmentMapLight size {}, correcting to {new_size}",
            size
        );
    }
    new_size
}

pub fn prepare_atmosphere_probe_components(
    probes: Query<(Entity, &AtmosphereEnvironmentMapLight), (Without<AtmosphereEnvironmentMap>,)>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    for (entity, env_map_light) in &probes {
        // Create a cubemap image in the main world that we can reference
        let new_size = validate_environment_map_size(env_map_light.size);
        let mut environment_image = Image::new_fill(
            Extent3d {
                width: new_size.x,
                height: new_size.y,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            &[0; 8],
            TextureFormat::Rgba16Float,
            RenderAssetUsages::all(),
        );

        environment_image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });

        environment_image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_SRC;

        // Add the image to assets to get a handle
        let environment_handle = images.add(environment_image);

        commands.entity(entity).insert(AtmosphereEnvironmentMap {
            environment_map: environment_handle.clone(),
            size: new_size,
        });

        commands
            .entity(entity)
            .insert(GeneratedEnvironmentMapLight {
                environment_map: environment_handle,
                intensity: env_map_light.intensity,
                rotation: Quat::IDENTITY,
                affects_lightmapped_mesh_diffuse: env_map_light.affects_lightmapped_mesh_diffuse,
            });
    }
}
pub fn atmosphere_environment(
    view: ViewQuery<(
        &DynamicUniformIndex<GpuAtmosphere>,
        &DynamicUniformIndex<GpuAtmosphereSettings>,
        &AtmosphereTransformsOffset,
        &ViewUniformOffset,
        &ViewLightsUniformOffset,
    )>,
    probe_query: Query<(
        Entity,
        &AtmosphereProbeBindGroups,
        &AtmosphereEnvironmentMap,
        Has<ExtractedView>,
        Option<&AtmosphereSkyRefresh>,
    )>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<AtmosphereProbePipeline>,
    mut ctx: RenderContext,
) {
    let Some(environment_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.environment)
    else {
        return;
    };

    let view_entity = view.entity();
    let (
        atmosphere_uniforms_offset,
        settings_uniforms_offset,
        atmosphere_transforms_offset,
        view_uniforms_offset,
        lights_uniforms_offset,
    ) = view.into_inner();

    // A probe on a camera is filled once, by its own view, and only while its sky changes.
    // Other probes go with every view.
    for (_, bind_groups, env_map_light, ..) in
        probe_query
            .iter()
            .filter(|(probe, _, _, is_view, refresh)| {
                (!is_view || *probe == view_entity) && refresh.is_none_or(|refresh| refresh.draw)
            })
    {
        let command_encoder = ctx.command_encoder();
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("environment_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(environment_pipeline);
        pass.set_bind_group(
            0,
            &bind_groups.environment,
            &[
                atmosphere_uniforms_offset.index(),
                settings_uniforms_offset.index(),
                atmosphere_transforms_offset.index(),
                view_uniforms_offset.offset,
                lights_uniforms_offset.offset,
            ],
        );

        pass.dispatch_workgroups(
            env_map_light.size.x / 8,
            env_map_light.size.y / 8,
            6, // 6 cubemap faces
        );
    }
}
