from ica import *

# Ensure the intermediate frames directory exists
os.makedirs("intermediate_frames", exist_ok=True)

video_path = "0.avi"
n_components = 50


## Select frames with potential noise
# selected_frames: List[bool] = analyze_and_select_frames(video_data.flat_frames)
#
# print(
# """The ica anlyisis will be applied to the following frames
# which are most probably noisy, based on the assumption that
# they have black lines of noise (the mean birghtness is lower):"""
# )
# print(np.array(range(len(selected_frames)))[selected_frames])
# selected_video_data = VideoData(
# video_data.flat_frames[selected_frames], video_data.frame_shape, video_data.fps
# )
#
# pca_result = apply_pca(selected_video_data.flat_frames, selected_video_data.flat_frames.shape[0])
# plot_cumulative_variance(pca_result.model)

video_data: VideoData = process_video(video_path)
video_data.flat_frames = video_data.flat_frames[0:200,:]

# Select frames with potential noise
selected_frames: List[bool] = analyze_and_select_frames(video_data.flat_frames)

print(
    """The ica anlyisis will be applied to the following frames
    which are most probably noisy, based on the assumption that
    they have black lines of noise (the mean birghtness is lower):"""
)
print(np.array(range(len(selected_frames)))[selected_frames])
selected_video_data = VideoData(
    video_data.flat_frames[selected_frames], video_data.frame_shape, video_data.fps
)

# Apply PCA and ICA
pca_result, ica_result = apply_pca_and_ica(
    selected_video_data.flat_frames, n_components
)

for i in range(n_components):
    save_frame_as_image(
        variance_image_from_video(
            video_from_nth_pca(pca_result, i), video_data.frame_shape
        ),
        f"variance_pca_component_{i}.png",
    )


for i in range(n_components):
    save_frame_as_image(
        variance_image_from_video(
            video_from_nth_ica(pca_result.model, ica_result, i), video_data.frame_shape
        ),
        f"variance_component_{i}.png",
    )
