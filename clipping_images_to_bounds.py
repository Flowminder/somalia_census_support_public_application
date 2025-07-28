import os
import geopandas as gpd
import rioxarray
from shapely.geometry import box

# --- USER CONFIG ---
mode = "validation"  # "training" or "validation"

normal_image_path = r"C:\Users\Emily Webb\somalia_census_support_public_application_og\data\imagery\SkySatCollect\20230320_104520_ssc11_u0001_pansharpened_clip.tif"
background_image_path = r"C:\Users\Emily Webb\OneDrive - Lancaster University\Data Science Masters\Placement\backgeround_large.tif"
geojson_dir = r"C:\Users\Emily Webb\OneDrive - Lancaster University\Data Science Masters\Placement\validation_masks"

if mode == "training":
    output_image_dir = r"C:\Users\Emily Webb\somalia_census_support_public_application\data\training\training_data\img"
    output_mask_dir = r"C:\Users\Emily Webb\somalia_census_support_public_application\data\training\training_data\mask"
else:  # validation
    output_image_dir = r"C:\Users\Emily Webb\somalia_census_support_public_application\data\validation\validation_data\img"
    output_mask_dir = r"C:\Users\Emily Webb\somalia_census_support_public_application\data\validation\validation_data\mask"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

target_size_pixels = 384  # Output tile size in pixels

# Load images
normal_image = rioxarray.open_rasterio(normal_image_path, masked=True)

if mode == "training":
    background_image = rioxarray.open_rasterio(background_image_path, masked=True)
    background_bounds = box(*background_image.rio.bounds())

transform = normal_image.rio.transform()
res_x, res_y = abs(transform.a), abs(transform.e)
tile_width_m = res_x * target_size_pixels
tile_height_m = res_y * target_size_pixels

normal_bounds = box(*normal_image.rio.bounds())

normal_idx = 1
background_idx = 1

# Precompute 4 tile centers from background image extent (only for training)
if mode == "training":
    xmin, ymin, xmax, ymax = background_image.rio.bounds()
    mid_x = (xmin + xmax) / 2
    mid_y = (ymin + ymax) / 2

    background_tile_centers = [
        (xmin + tile_width_m / 2, mid_y + tile_height_m / 2),  # top-left
        (mid_x + tile_width_m / 2, mid_y + tile_height_m / 2),  # top-right
        (xmin + tile_width_m / 2, ymin + tile_height_m / 2),  # bottom-left
        (mid_x + tile_width_m / 2, ymin + tile_height_m / 2),  # bottom-right
    ]

# Loop through GeoJSON masks
for filename in sorted(os.listdir(geojson_dir)):
    if not filename.endswith(".geojson"):
        continue

    mask_path = os.path.join(geojson_dir, filename)
    print(f"🔄 Processing {filename}")

    try:
        gdf = gpd.read_file(mask_path)
    except (OSError, ValueError) as e:
        print(f"❌ Failed to read {filename}: {e}")
        continue

    # Detect background only if training mode
    is_background = mode == "training" and "background" in filename.lower()

    if is_background:
        if background_idx > len(background_tile_centers):
            print(f"🚫 Skipping {filename}: not enough background tiles defined.")
            continue

        center_x, center_y = background_tile_centers[background_idx - 1]
    else:
        if gdf.empty:
            print(f"❌ Skipping {filename}: empty GeoJSON.")
            continue

        if not gdf.is_valid.all():
            print(f"⚠️ Fixing invalid geometries in {filename}")
            gdf["geometry"] = gdf["geometry"].buffer(0)

        xmin, ymin, xmax, ymax = gdf.total_bounds
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

    half_w = tile_width_m / 2
    half_h = tile_height_m / 2
    square_bounds = (
        center_x - half_w, center_y - half_h,
        center_x + half_w, center_y + half_h
    )
    square_geom = box(*square_bounds)
    if not square_geom.is_valid:
        square_geom = square_geom.buffer(0)

    image = background_image if is_background else normal_image
    image_bounds = background_bounds if is_background else normal_bounds

    if not square_geom.intersects(image_bounds):
        print(f"🚫 Skipping {filename}: tile outside image bounds.")
        continue

    print(f"📐 Image bounds: {image_bounds.bounds}")
    print(f"📦 Tile bounds: {square_geom.bounds}")

    tile_id = f"{background_idx:02d}_background" if is_background else f"{normal_idx:02d}"
    area = "baidoa"
    prefix = "training_data" if mode == "training" else "validation_data"
    out_image_name = f"{prefix}_{area}_{tile_id}.tif"
    out_mask_name = f"{prefix}_{area}_{tile_id}.geojson"

    try:
        clipped_image = image.rio.clip([square_geom], normal_image.rio.crs, drop=True)
        output_image_path = os.path.join(output_image_dir, out_image_name)
        clipped_image.rio.to_raster(output_image_path)
    except (ValueError, RuntimeError) as e:
        print(f"🚫 Failed to clip image for {filename}: {e}")
        continue

    try:
        output_mask_path = os.path.join(output_mask_dir, out_mask_name)
        if not is_background:
            clipped_mask = gpd.clip(gdf, square_geom)
            clipped_mask.to_file(output_mask_path, driver="GeoJSON")
        else:
            # Create empty mask file for background
            gpd.GeoDataFrame(geometry=[], crs=normal_image.rio.crs).to_file(output_mask_path, driver="GeoJSON")
    except (ValueError, RuntimeError) as e:
        print(f"🚫 Failed to save mask for {filename}: {e}")
        continue

    if is_background:
        background_idx += 1
    else:
        normal_idx += 1

print(f"✅ Done: Created {mode} image/mask tiles{', including background.' if mode == 'training' else '.'}")
