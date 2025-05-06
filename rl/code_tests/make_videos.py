import imageio.v2 as imageio
from pathlib import Path

def _make_video(target_filename:str, output_path:str):
    image_paths = sorted(Path(".").rglob(target_filename))
    images = [imageio.imread(str(path)) for path in image_paths]
    imageio.mimsave(output_path, images, fps=1, format='ffmpeg')  # key fix: format='ffmpeg'
    print(f"Video saved to {output_path}")


def main():
    _make_video("Quantiles of Cumulative Reward at Each Step.png","reward_quantiles_video.mp4")
    _make_video("Step vs Number of Cancer Cells for Each Episode.png", "cancer_cells_quantiles_video.mp4")
if __name__=="__main__":
    main()