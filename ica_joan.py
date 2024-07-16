"""Apply ICA to denoise images."""

import cv2
import numpy as np
from sklearn.decomposition import PCA, FastICA
from dataclasses import dataclass
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk


@dataclass
class VideoData:
    """Information of the loaded video."""

    frames: np.ndarray
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
    cv2.imwrite(filename, frame)


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
            save_frame_as_image(
                frame_gray, f"intermediate_frames/original_frame_{frame_count}.png"
            )

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


def frame_array_from_pca(
    pca, pca_components: np.ndarray, frame_shape: Tuple[int, int]
) -> np.ndarray:
    """Reconstruct the video frames from the PCA data."""
    frames = pca.inverse_transform(pca_components)

    # Reshape and normalize reconstructed frames
    frames = frames.reshape(-1, *frame_shape)
    frames = cv2.normalize(frames, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return frames


def frame_array_from_ica(
    pca: PCA,
    ica_result: DecompositionResult,
    selected_components: List[bool],
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """Reconstruct frames from selected ICA components."""
    reconstructed_components_pca = reconstruct_pca_from_ica(
        ica_result, selected_components
    )
    reconstructed_frames = frame_array_from_pca(
        pca, reconstructed_components_pca, frame_shape
    )

    # Save a few reconstructed frames as images
    for i in range(5):
        save_frame_as_image(
            reconstructed_frames[i], f"intermediate_frames/reconstructed_frame_{i}.png"
        )

    return reconstructed_frames


def frame_array_from_nth_ica(
    pca: PCA,
    ica_result: DecompositionResult,
    component: int,
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    n_final_components = ica_result.components.shape[1]
    selected_components = [False for i in range(n_final_components)]
    selected_components[component] = True
    processed_frames_array = frame_array_from_ica(
        pca_result.model, ica_result, selected_components, video_data_flat.frame_shape
    )
    return processed_frames_array


def save_frames_as_video(
    frames: np.ndarray,
    frame_shape: Tuple[int, int],
    fps: int,
    output_path: str = "output_video.mp4",
) -> None:
    """Save frames as a video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (frame_shape[1], frame_shape[0]), isColor=False
    )

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved as {output_path}")

    ## TODO (first frame is not sufficient)


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
        reconstructed_frames = frame_array_from_ica(
            pca, ica_result, selected_components, frame_shape
        )
        component_image = reconstructed_frames[0]
        ax.imshow(component_image, cmap="gray")
        ax.set_title(f"Component {i+1}")
        ax.axis("off")

    plt.show()


def select_ica_components(n_components: int) -> List[bool]:
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


def save_ica_components_videos(ica_result, pca_result, video_data_flat):
    """Save each component as an mp4 to visualize them independently."""
    n_final_components = ica_result.components.shape[1]
    for i in range(0, n_final_components):
        selected_components = [False for i in range(n_final_components)]
        selected_components[i] = True
        processed_frames_array = frame_array_from_ica(
            pca_result.model,
            ica_result,
            selected_components,
            video_data_flat.frame_shape,
        )
        save_frames_as_video(
            processed_frames_array,
            video_data_flat.frame_shape,
            video_data_flat.fps,
            f"output_component_{i}.mp4",
        )


def detect_horizontal_noise(
    pca: PCA, ica_result: DecompositionResult, frame_shape: Tuple[int, int]
) -> List[float]:
    scores = []
    for i in range(ica_result.components.shape[1]):
        component = frame_array_from_nth_ica(pca, ica_result, i, frame_shape)

        # Apply Sobel filter to detect horizontal edges
        sobel_horizontal = cv2.Sobel(component[0], cv2.CV_64F, 1, 0, ksize=5)
        score = np.sum(np.abs(sobel_horizontal))
        scores.append(score)

    return scores


# Ensure the intermediate frames directory exists
os.makedirs("intermediate_frames", exist_ok=True)

# Load and process video frames
video_path = "0.mp4"
video_data_flat = process_video(video_path)

# Print shape of frames array
print("Shape of flat frames array:", video_data_flat.frames.shape)

# Apply PCA for dimensionality reduction
n_components_pca = 100  # Use the same number of components for both PCA and ICA
pca_result = apply_pca(video_data_flat.frames, n_components_pca)

# Apply ICA on PCA-transformed data
n_components_ica = 10
ica_result = apply_ica(pca_result.components, n_components_ica)

# Plot ICA components for visualization
#plot_ica_components(ica_result, pca_result.model, video_data_flat.frame_shape)

# Select components to keep
# selected_components = select_ica_components(n_components_ica)
print(detect_horizontal_noise(pca_result.model, ica_result, video_data_flat.frame_shape))

# Reconstruct frames from selected ICA components
# processed_frames_array = frame_array_from_ica(
#    pca_result.model, ica_result, selected_components, video_data_flat.frame_shape
# )
# print("Shape of reconstructed frames:", processed_frames_array.shape)

save_ica_components_videos(ica_result, pca_result, video_data_flat)
