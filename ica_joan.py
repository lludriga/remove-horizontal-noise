"""Apply ICA to denoise images."""

import cv2
import numpy as np
from sklearn.decomposition import PCA, FastICA
from dataclasses import dataclass
import os
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk


@dataclass
class VideoData:
    """Information of the loaded video."""

    flat_frames: np.ndarray
    frame_shape: Tuple[int, int]
    fps: int


@dataclass
class DecompositionResult:
    """Structure for the ica and pca results."""

    model: any
    components: np.ndarray
    n_components: int


def save_frame_as_image(frame: np.ndarray, filename: str) -> None:
    """Save a single frame as an image file."""
    cv2.imwrite(f"intermediate_frames/{filename}", frame)


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
    return VideoData(np.array(frames), frame_shape, fps)


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
    ica = FastICA(n_components=n_components, random_state=0)
    components_ica = ica.fit_transform(components)
    print(f"Finished ICA. Shape {components_ica.shape}")

    return DecompositionResult(ica, components_ica, n_components)


def reconstruct_pca_from_ica(
    ica_result: DecompositionResult, selected_components: List[bool]
) -> np.ndarray:
    """Reconstruct the PCA data from the selected components from the ICA."""
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
    pca, pca_components: np.ndarray, frame_shape: Tuple[int, int]
) -> np.ndarray:
    """Reconstruct the video frames from the PCA data."""
    frames = pca.inverse_transform(pca_components)

    # Reshape and normalize reconstructed frames
    frames = frames.reshape(-1, *frame_shape)
    frames = cv2.normalize(frames, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return frames


def video_from_ica(
    pca: PCA,
    ica_result: DecompositionResult,
    selected_components: List[bool],
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """Reconstruct frames from selected ICA components."""
    reconstructed_components_pca = reconstruct_pca_from_ica(
        ica_result, selected_components
    )
    reconstructed_frames = video_from_pca(
        pca, reconstructed_components_pca, frame_shape
    )

    # Save a few reconstructed frames as images
    for i in range(5):
        save_frame_as_image(reconstructed_frames[i], f"reconstructed_frame_{i}.png")

    return reconstructed_frames


def video_from_nth_ica(
    pca: PCA,
    ica_result: DecompositionResult,
    component: int,
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """Create the array of frames for the nth ica component."""
    n_final_components = ica_result.components.shape[1]
    selected_components = [False for i in range(n_final_components)]
    selected_components[component] = True
    processed_frames_array = video_from_ica(
        pca_result.model, ica_result, selected_components, video_data.frame_shape
    )
    return processed_frames_array


def save_video_as_mp4(
    video: np.ndarray,
    frame_shape: Tuple[int, int],
    fps: int,
    output_path: str = "output_video.mp4",
) -> None:
    """Save frames as a video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (frame_shape[1], frame_shape[0]), isColor=False
    )

    for frame in video:
        out.write(frame)

    out.release()
    print(f"Video saved as {output_path}")


# TODO (first frame is not sufficient)
def plot_ica_components(
    ica_result: DecompositionResult, pca: PCA, frame_shape: Tuple[int, int]
):
    """Plot the ICA components for visualization."""
    n_components = ica_result.components.shape[1]
    fig, axes = plt.subplots(1, n_components, figsize=(15, 5))

    for i, ax in enumerate(axes):
        selected_components = [
            True if j == i else False for j in range(0, n_components)
        ]
        reconstructed_frames = video_from_ica(
            pca, ica_result, selected_components, frame_shape
        )
        component_image = reconstructed_frames[0]
        ax.imshow(component_image, cmap="gray")
        ax.set_title(f"Component {i+1}")
        ax.axis("off")

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
        inverted_indices = [j for j in range(pca_result.n_components) if j != i]
        selected_pca = pca_result.components.copy()
        selected_pca[:, inverted_indices] = 0
        processed_frames_array = video_from_pca(
            pca_result.model,
            selected_pca,
            video_data.frame_shape,
        )
        save_video_as_mp4(
            processed_frames_array,
            video_data.frame_shape,
            video_data.fps,
            f"output_component_{i}.mp4",
        )


def save_ica_components_videos(ica_result, pca_result, video_data):
    """Save each component as an mp4 to visualize them independently."""
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
        save_video_as_mp4(
            processed_frames_array,
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


def detect_horizontal_noise(
    pca: PCA, ica_result: DecompositionResult, frame_shape: Tuple[int, int]
) -> List[float]:
    """Compute a score for each component to try to detect the horizontal noise."""
    scores = []
    for i in range(ica_result.components.shape[1]):
        component = video_from_nth_ica(pca, ica_result, i, frame_shape)

        # Apply Sobel filter to detect horizontal edges
        sobel_horizontal = cv2.Sobel(component[0], cv2.CV_64F, 1, 0, ksize=5)
        if i < 5:
            save_frame_as_image(sobel_horizontal, f"sobel_frame_{i}.png")
        score = np.sum(np.abs(sobel_horizontal))
        scores.append(score)

    return scores


# Realment no detecta res, son linies massa poc concretes
def detect_and_remove_horizontal_lines(image: np.ndarray) -> np.ndarray:
    """Detect and remove horizontal lines from an image using Hough Line Transform."""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )
    print(f"Detected {lines} lines")

    mask = (
        np.ones_like(image, dtype=np.uint8) * 255
    )  # Initialize mask with white background

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # Horizontal line condition
                cv2.line(mask, (x1, y1), (x2, y2), 0, thickness=2)

    cleaned_image = cv2.bitwise_and(image, mask)
    return cleaned_image


def test_filter_first_frames(
    video: np.ndarray,
    filter: Callable[[np.ndarray], np.ndarray],
    output_path: str = "filtered",
) -> np.ndarray:
    """Apply the filter function (from image to image) to the first frames of the videos to test it."""
    for i in range(0, 5):
        filtered = detect_and_remove_horizontal_lines(video[i])
        save_frame_as_image(filtered, f"{output_path}_frame_{i}.png")


# Ensure the intermediate frames directory exists
os.makedirs("intermediate_frames", exist_ok=True)

# Load and process video frames
video_path = "0.mp4"
video_data = process_video(video_path)

# Print shape of frames array
print("Shape of flat frames array:", video_data.flat_frames.shape)

# Batch apply
n_components_pca = 60  # Use the same number of components for both PCA and ICA
pca_result = apply_pca(video_data.flat_frames[0:60, :], n_components_pca)

save_pca_components_videos(pca_result, video_data)

# Apply ICA on PCA-transformed data
#n_components_ica = 20
#ica_result = apply_ica(pca_result.components, n_components_ica)
#save_ica_components_videos(ica_result, pca_result, video_data)

# Apply PCA for dimensionality reduction
# n_components_pca = 100  # Use the same number of components for both PCA and ICA
# pca_result = apply_pca(video_data_flat.frames, n_components_pca)
#
## Apply ICA on PCA-transformed data
# n_components_ica = 10
# ica_result = apply_ica(pca_result.components, n_components_ica)

# Plot ICA components for visualization
# plot_ica_components(ica_result, pca_result.model, video_data_flat.frame_shape)

# Select components to keep
# selected_components = select_ica_components(n_components_ica)
# print(
#    detect_horizontal_noise(pca_result.model, ica_result, video_data_flat.frame_shape)
# )

# Reconstruct frames from selected ICA components
# processed_frames_array = frame_array_from_ica(
#    pca_result.model, ica_result, selected_components, video_data_flat.frame_shape
# )
# print("Shape of reconstructed frames:", processed_frames_array.shape)



# for i in range(ica_result.n_components):
#    test_filter_first_frames(
#        video_from_nth_ica(
#            pca_result.model, ica_result, i, video_data_flat.frame_shape
#        ),
#        detect_and_remove_horizontal_lines,
#        f"filtered_component_{i}",
#    )
