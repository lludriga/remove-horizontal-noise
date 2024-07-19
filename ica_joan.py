"""Apply ICA to denoise images."""

import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Callable, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA  # type: ignore


@dataclass
class VideoData:
    """Information of the loaded video."""

    flat_frames: np.ndarray
    frame_shape: Tuple[int, int]
    fps: int


@dataclass
class DecompositionResult:
    """Structure for the ica and pca results."""

    model: PCA | FastICA
    components: np.ndarray
    n_components: int


def save_frame_as_image(frame: np.ndarray, filename: str) -> None:
    """Save a single frame as an image file."""
    cv2.imwrite(f"intermediate_frames/{filename}", frame)


def process_not_flat_video(video_path: str) -> np.ndarray:
    """Load the video and return a not flat grayscale version."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_shape = None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame_gray)

        # Save a few original frames
        if frame_count < 5:
            save_frame_as_image(frame_gray, f"original_frame_{frame_count}.png")

        if frame_shape is None:
            frame_shape = frame_gray.shape

        frame_count += 1

    cap.release()

    return np.array(frames)


def process_video(video_path: str) -> VideoData:
    """Load the video and return a flattened grayscale version."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_shape = None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame_gray.flatten())

        # Save a few original frames
        if frame_count < 5:
            save_frame_as_image(frame_gray, f"original_frame_{frame_count}.png")

        if frame_shape is None:
            frame_shape = frame_gray.shape

        frame_count += 1

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if isinstance(frame_shape, tuple) and len(frame_shape) == 2:
        return VideoData(np.array(frames), frame_shape, fps)
    else:
        raise Exception("Error getting the frame shape. Incorrect dimensions")


def apply_pca(frames: np.ndarray, n_components: int) -> DecompositionResult:
    """Apply pca to the input array with n components."""
    print("Applying PCA")
    pca = PCA(n_components=n_components, random_state=0)
    components_pca = pca.fit_transform(frames)
    print(f"Finished PCA. Shape {components_pca.shape}")

    return DecompositionResult(pca, components_pca, n_components)


def apply_ica(components: np.ndarray, n_components) -> DecompositionResult:
    """Apply ICA to the input array with n components."""
    print("Applying ICA")
    ica = FastICA(n_components=n_components, random_state=0, max_iter=1000)
    components_ica = ica.fit_transform(components)
    print(f"Finished ICA. Shape {components_ica.shape}")

    return DecompositionResult(ica, components_ica, n_components)


def reconstruct_pca_from_ica(
    ica_result: DecompositionResult,
    selected_components: List[bool] = [],
) -> np.ndarray:
    """Reconstruct the PCA data from the selected components from the ICA."""
    if not selected_components:
        selected_components = [True for i in range(ica_result.n_components)]
    if len(selected_components) != ica_result.n_components:
        raise Exception("Incorrect length of list of selected components.")

    inverted_selection = np.invert(selected_components)
    inverted_indices = [i for i, include in enumerate(inverted_selection) if include]
    selected_ica_components = ica_result.components.copy()
    selected_ica_components[:, inverted_indices] = 0
    reconstructed_components_pca = ica_result.model.inverse_transform(
        selected_ica_components
    )
    print("Reconstructed pca components shape: ", reconstructed_components_pca.shape)
    return reconstructed_components_pca


def video_from_pca(
    pca_result: DecompositionResult,
    selected_components: List[bool] = [],
) -> np.ndarray:
    """Reconstruct the video frames from the PCA data."""
    if not selected_components:
        selected_components = [True for i in range(pca_result.n_components)]
    if len(selected_components) != pca_result.n_components:
        raise Exception("Incorrect length of list of selected components.")

    inverted_selection = np.invert(selected_components)
    inverted_indices = [i for i, include in enumerate(inverted_selection) if include]
    selected_pca_components = pca_result.components.copy()
    selected_pca_components[:, inverted_indices] = 0
    # Reconstruct, reshape and normalize frames
    frames: np.ndarray = pca_result.model.inverse_transform(selected_pca_components)

    return frames


def video_from_nth_pca(
    pca_result: DecompositionResult,
    component: int,
) -> np.ndarray:
    """Create the array of frames for the nth pca component."""
    selected_components = [False for i in range(pca_result.n_components)]
    selected_components[component] = True
    video = video_from_pca(pca_result, selected_components)
    return video


def video_from_ica(
    pca: PCA,
    ica_result: DecompositionResult,
    selected_components: List[bool] = [],
) -> np.ndarray:
    """Reconstruct frames from selected ICA components."""
    if not selected_components:
        selected_components = [True for i in range(ica_result.n_components)]
    if len(selected_components) != ica_result.n_components:
        raise Exception("Incorrect length of list of selected components.")

    reconstructed_components_pca = reconstruct_pca_from_ica(
        ica_result, selected_components
    )
    reconstructed_frames = video_from_pca(
        DecompositionResult(
            pca, reconstructed_components_pca, reconstructed_components_pca.shape[1]
        ),
    )

    return reconstructed_frames


def video_from_nth_ica(
    pca: PCA,
    ica_result: DecompositionResult,
    component: int,
) -> np.ndarray:
    """Create the array of frames for the nth ica component."""
    selected_components = [False for i in range(ica_result.n_components)]
    selected_components[component] = True
    video = video_from_ica(pca, ica_result, selected_components)
    return video


def save_video_as_mp4(
    flat_frames: np.ndarray,
    frame_shape: Tuple[int, int],
    fps: int,
    output_path: str = "output_video.mp4",
) -> None:
    """Save frames as a video file."""
    frames: np.ndarray = flat_frames.reshape(-1, *frame_shape)

    if np.ndim(frames) == 3:
        frames = cv2.normalize(frames, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # type: ignore
    else:
        raise Exception("Invalid dimensions for video when reshaping.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (frame_shape[1], frame_shape[0]), isColor=False
    )

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved as {output_path}")


# TODO (first frame is not sufficient)
def plot_ica_components(
    ica_result: DecompositionResult, pca: PCA, frame_shape: Tuple[int, int]
) -> None:
    """Plot the ICA components for visualization."""
    n_components = ica_result.components.shape[1]
    fig, axes = plt.subplots(1, n_components, figsize=(15, 5))

    if isinstance(axes, Iterable):
        for i, ax in enumerate(axes):
            selected_components = [
                True if j == i else False for j in range(0, n_components)
            ]
            reconstructed_frames = video_from_ica(pca, ica_result, selected_components)
            component_image = reconstructed_frames[0]
            ax.imshow(component_image, cmap="gray")
            ax.set_title(f"Component {i+1}")
            ax.axis("off")

    else:
        raise Exception(
            """For some reason, axes is not iterable.
            Maybe there isnt't any component in the ica? Or only 1"""
        )

    plt.show()


def select_ica_components_interactive(n_components: int) -> List[bool]:
    """Use a GUI to select which ICA components to include."""
    root = tk.Tk()
    root.title("Select ICA Components")

    selected_components = [tk.BooleanVar() for _ in range(n_components)]

    def on_submit():
        root.quit()

    for i in range(n_components):
        ttk.Checkbutton(
            root, text=f"Component {i+1}", variable=selected_components[i]
        ).pack(anchor="w")

    submit_button = ttk.Button(root, text="Submit", command=on_submit)
    submit_button.pack(anchor="s")

    root.mainloop()
    return [var.get() for var in selected_components]


def save_pca_components_videos(pca_result, video_data):
    """Save each pca component as an mp4 to visualize them independently."""
    for i in range(0, pca_result.n_components):
        video = video_from_nth_pca(pca_result, i)
        save_video_as_mp4(
            video,
            video_data.frame_shape,
            video_data.fps,
            f"output_component_{i}.mp4",
        )


def save_ica_components_videos(ica_result, pca_result, video_data):
    """Save each component as an mp4 to visualize them independently."""
    n_final_components = ica_result.components.shape[1]
    for i in range(0, n_final_components):
        video = video_from_nth_ica(
            pca_result.model,
            ica_result,
            i,
        )
        save_video_as_mp4(
            video,
            video_data.frame_shape,
            video_data.fps,
            f"output_component_{i}.mp4",
        )


def save_ica_components_frames(ica_result, pca_result, video_data):
    """Save each component frames as png to visualize them independently."""
    n_final_components = ica_result.components.shape[1]
    for i in range(0, n_final_components):
        selected_components = [False for i in range(n_final_components)]
        selected_components[i] = True
        processed_frames_array = video_from_ica(
            pca_result.model,
            ica_result,
            selected_components,
            video_data.frame_shape,
        )
        for j, frame in enumerate(processed_frames_array):
            save_frame_as_image(frame, f"component_{i}_frame_{j}.png")


def save_first_frames_filtered(
    video: np.ndarray,
    filter: Callable[[np.ndarray], np.ndarray],
    output_path: str = "filtered",
    n_frames: int = 5,
) -> None:
    """Apply the filter function (from image to image) to the first frames of the videos to test it."""
    for i in range(0, min(n_frames, video.shape[0])):
        filtered = filter(video[i])
        save_frame_as_image(filtered, f"{output_path}_frame_{i}.png")


def plot_flat_frames(frames: np.ndarray, n: int = 5) -> None:
    """
    Plot the first n frames flattened.

    Parameters:
    frames (np.ndarray): The array of frames.
    n (int): The number of frames to plot. Defaults to 5.
    """
    # Ensure n does not exceed the number of frames available
    n = min(n, frames.shape[0])

    fig, axes = plt.subplots(1, n, figsize=(15, 5))

    if isinstance(axes, Iterable):
        for i in range(n):
            axes[i].plot(frames[i])
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def analyze_and_select_frames(flat_frames: np.ndarray) -> List[bool]:
    """
    Analyze the frames statistically to try to remove the ones without noise.

    We apply a statistic, the mean in these case, 3 by 3 so we ensure the background
    image stays mostly the same, and we can detect almost surely the variation caused
    by the black lines, making the mean lower.
    We return a list of bools containg true in the ones chosen, the ones with noise
    supposedly. We choose 2 out of 3 every time, since the noise doesn't happen (apparently)
    3 frames.
    """
    batch = 3
    remaining_frames = flat_frames.shape[0] % batch
    selected_frames: List[bool] = []
    for j in range(flat_frames.shape[0] // batch):
        frame_mean: np.ndarray = np.zeros(batch)
        for i in range(batch):
            frame_mean[i] = np.mean(flat_frames[j * batch + i])

        max_accepted_mean = np.quantile(frame_mean, 0.66)

        for mean in frame_mean:
            if mean <= max_accepted_mean:
                selected_frames.append(True)
            else:
                selected_frames.append(False)

    # Since the remaining batch is smaller than what we analyze
    # we do the easy thing and just include it all
    for i in range(remaining_frames):
        selected_frames.append(True)

    return selected_frames


def review_ica_components(
    ica_result: DecompositionResult, pca: PCA, frame_shape: Tuple[int, int], fps: int
) -> List[bool]:
    """Interactively review each ICA component and decide whether to keep it."""
    selected_components = [False for _ in range(ica_result.n_components)]

    # Function to show video and ask for decision
    def show_component_video(component_index):
        # Create video for the given component
        video = video_from_nth_ica(pca, ica_result, component_index)
        video_path = f"temp_component_{component_index}.mp4"
        save_video_as_mp4(video, frame_shape, fps, video_path)

        # Play the video using the default video player
        os.system(f"start {video_path}" if os.name == "nt" else f"mpv {video_path}")

        # Ask if the component should be kept
        result = messagebox.askyesno(
            "Keep Component", f"Do you want to keep component {component_index + 1}?"
        )
        return result

    # Iterate over each component and get user input
    for i in range(ica_result.n_components):
        keep = show_component_video(i)
        selected_components[i] = keep

    # Clean up temporary video files
    for i in range(ica_result.n_components):
        video_path = f"temp_component_{i}.mp4"
        if os.path.exists(video_path):
            os.remove(video_path)

    return selected_components


def reconstruct_full_video(
    original_video_data: VideoData,
    selected_frames: List[bool],
    reconstructed_frames: np.ndarray,
) -> np.ndarray:
    """Reconstruct the full video including non-noisy frames in their correct order."""
    full_video = np.zeros_like(original_video_data.flat_frames)

    # Insert reconstructed frames into the full video array
    j = 0
    for i in range(len(selected_frames)):
        if selected_frames[i]:
            full_video[i] = reconstructed_frames[j]
            j += 1
        else:
            full_video[i] = original_video_data.flat_frames[i]

    return full_video



# Ensure the intermediate frames directory exists
os.makedirs("intermediate_frames", exist_ok=True)

video_path = "0.avi"
video_data = process_video(video_path)

# Print shape of frames array
print("Shape of flat frames array:", video_data.flat_frames.shape)

selected_frames = analyze_and_select_frames(video_data.flat_frames)

# Check the selected frames
for i in range(len(selected_frames)):
    if selected_frames[i]:
        print(i)

selected_video_data = VideoData(
    video_data.flat_frames[selected_frames], video_data.frame_shape, video_data.fps
)


n_components_pca = 3
pca_result = apply_pca(selected_video_data.flat_frames, n_components_pca)

n_components_ica = n_components_pca
ica_result = apply_ica(pca_result.components, n_components_ica)

selected_ica_components = review_ica_components(
    ica_result, pca_result.model, video_data.frame_shape, video_data.fps
)

# Reconstruct the selected frames with the selected ICA components
reconstructed_selected_frames = video_from_ica(
    pca_result.model, ica_result, selected_ica_components
)

# Reconstruct the full video including non-noisy frames
full_reconstructed_video = reconstruct_full_video(
    video_data, selected_frames, reconstructed_selected_frames
)

# Save the full reconstructed video
save_video_as_mp4(
    full_reconstructed_video,
    video_data.frame_shape,
    video_data.fps,
    "full_reconstructed_video.mp4",
)
