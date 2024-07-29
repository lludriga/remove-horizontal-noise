import matplotlib.pyplot as plt
import numpy as np

from ica import (ComponentSelectorGUI, GaussianSpatialFilter, ICAAnalysis,
                 PCAAnalysis, VideoAnalyzer, VideoData, save_frame_as_image,
                 save_is_component_selected_json)

# Initialize VideoProcessor with preprocessors and analysis methods
video_path = "0.avi"
output_path = "output_video.mp4"
video = VideoData(
    video_path,
)

# Load video
video.load_video()

n_components = 122

analyzer = VideoAnalyzer(
    video,
    [GaussianSpatialFilter()],
    PCAAnalysis(n_components),
    ICAAnalysis(),
)

# Choose how many components to use
# analyzer.choose_n_pca_components_interactive()

# Decompose video
analyzer.decompose_video()
images = analyzer.get_component_maps()
images = abs(images)

for i, image in enumerate(images):
    save_frame_as_image(image, f"component_map_{i}.png")


# Select ICA components
# Two options: component maps and variance images (more expensive)
# is_component_selected = ComponentSelectorGUI().select_components(
#    analyzer.get_component_maps()
# )
# save_is_component_selected_json(is_component_selected, video_path)
#
# video.save_frames_as_mp4(
#    analyzer.compose_video(is_component_selected=is_component_selected), output_path
# )
#
# non_selected_components = [not selected for selected in is_component_selected]
# video.save_frames_as_mp4(
#    analyzer.compose_video(is_component_selected=non_selected_components),
#    "non_selected.mp4",
# )
