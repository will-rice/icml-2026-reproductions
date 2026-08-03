# VLM-RobustBench

49 augmentation types organized in 10 categories:
- **Blur** (5): gaussian_blur, motion_blur, defocus_blur, glass_blur, zoom_blur
- **Noise** (4): gaussian_noise, shot_noise, speckle_noise, salt_pepper
- **Weather** (5): fog, frost, snow, rain, spatter
- **Digital** (2): jpeg_compression, pixelate
- **Geometric** (5): rotate, shear, affine, perspective_transform, elastic_transform
- **Occlusion** (3): center_occlusion, random_occlusion, grid_mask
- **Color** (10): brightness, brightness_up, contrast, contrast_up, saturation, saturation_up, gamma, gamma_up, hue_shift, color_jitter
- **Resolution** (5): downsample, upsample, sharpen, posterize, solarize
- **VLM-specific** (3): text_overlay, watermark, add_border
- **Binary** (7): flip_h, flip_v, grayscale, invert, channel_swap, equalize, autocontrast

Severity levels: 1 (low), 3 (mid), 5 (high) for non-binary augmentations.
