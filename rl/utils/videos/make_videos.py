import os
import glob
import imageio
import imageio.v3 as iio

def png_to_video_imageio(
    image_root_folder: str = "./output/image", fps: int = 10, folder_name_save:str = "test_videos_folder"
):
    # Output directory for videos
    output_video_folder = os.path.join(image_root_folder, folder_name_save)
    os.makedirs(output_video_folder, exist_ok=True)

    # Find subfolders ending with "test"
    subfolders = [
        os.path.join(image_root_folder, d)
        for d in os.listdir(image_root_folder)
        if os.path.isdir(os.path.join(image_root_folder, d)) and "test" in d
    ]

    if not subfolders:
        print("❌ No subfolders ending with 'test' found.")
        return

    for subfolder in subfolders:
        images = sorted(glob.glob(os.path.join(subfolder, "*.png")))
        if not images:
            print("⚠️ No images in:", subfolder)
            continue

        print(f"🖼️ Found {len(images)} images in {subfolder}. First image: {images[0]}")
        frame = iio.imread(images[0])
        height, width, _ = frame.shape
        print(f"📏 Image size: {width}x{height}")

        video_name = os.path.basename(subfolder) + ".mp4"
        output_video_path = os.path.join(output_video_folder, video_name)

        writer = imageio.get_writer(
            output_video_path, fps=fps, codec="libx264", format="FFMPEG", pixelformat="yuv420p"
        )

        for img in images:
            frame = iio.imread(img)
            writer.append_data(frame)

        writer.close()
        print(f"✅ Video saved as {output_video_path}")

if __name__=="__main__":
    png_to_video_imageio(image_root_folder="/home/alex/Physi/PhysiCell/runs/physigym/ModelPhysiCellEnv-v0__sac_corporate-manu-sureli_1746537184/image",
                         fps=10,
                         folder_name_save="tests")
