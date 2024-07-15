"""Apply ICA to denoise images."""

import cv2
import numpy as np
from sklearn.decomposition import PCA, FastICA
from dataclasses import dataclass
import os
from typing import List, Tuple


@dataclass
class VideoData:
    frames: np.ndarray
    frame_shape: Tuple[int, int]
    fps: int


@dataclass
class DecompositionResult:
    model: any
    components: np.ndarray


def save_frame_as_image(frame: np.ndarray, filename: str) -> None:
    """Save a single frame as an image file."""
    cv2.imwrite(filename, frame)


def process_video(video_path: str) -> VideoData:
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
            save_frame_as_image(frame_gray, f'intermediate_frames/original_frame_{frame_count}.png')

        if frame_shape is None:
            frame_shape = frame_gray.shape

        frame_count += 1
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return VideoData(np.array(frames), frame_shape, fps)

def apply_pca(frames: np.ndarray, n_components: int) -> DecompositionResult:
    print("Applying PCA")
    pca = PCA(n_components=n_components, random_state=0)
    components_pca = pca.fit_transform(frames)
    print("Finished PCA")
    
    # Save a few PCA components as images
    for i in range(5):
        pca_frame = pca.inverse_transform(components_pca[i]).reshape(video_data.frame_shape)
        save_frame_as_image(pca_frame, f'intermediate_frames/pca_frame_{i}.png')
    
    return DecompositionResult(pca, components_pca)

def apply_ica(components: np.ndarray, n_components: int) -> DecompositionResult:
    print("Applying ICA")
    ica = FastICA(n_components=n_components, random_state=0)
    components_ica = ica.fit_transform(components[:, :n_components])
    print("Finished ICA")
    
    return DecompositionResult(ica, components_ica)

def frame_array_from_ica_components(
        pca: PCA,
        ica_result: DecompositionResult,
        selected_components: List[bool],
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
    """Reconstruct frames from selected ICA components."""
    selected_indices = [i for i, include in enumerate(selected_components) if include]
    selected_ica_components = ica_result.components[:, selected_indices]
    selected_mixing_matrix = ica_result.model.mixing_[:, selected_indices].T
    reconstructed_components_pca = np.dot(selected_ica_components, selected_mixing_matrix)
    reconstructed_frames = pca.inverse_transform(reconstructed_components_pca)
    
    # Reshape and normalize reconstructed frames
    reconstructed_frames = reconstructed_frames.reshape(-1, *frame_shape)
    reconstructed_frames = cv2.normalize(reconstructed_frames, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Save a few reconstructed frames as images
    for i in range(5):
        save_frame_as_image(reconstructed_frames[i], f'intermediate_frames/reconstructed_frame_{i}.png')
    
    return reconstructed_frames

def save_frames_as_video(frames: np.ndarray, frame_shape: Tuple[int, int], fps: int, output_path: str = 'output_video.mp4') -> None:
    """Save frames as a video file."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_shape[1], frame_shape[0]), isColor=False)
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved as {output_path}")


# Ensure the intermediate frames directory exists
os.makedirs('intermediate_frames', exist_ok=True)

# Load and process video frames
video_path = '0.mp4'
video_data_flat = process_video(video_path)

# Print shape of frames array
print("Shape of flat frames array:", video_data_flat.frames.shape)

# Apply PCA for dimensionality reduction
n_components = 10  # Use the same number of components for both PCA and ICA
pca_result = apply_pca(video_data_flat.frames, n_components)

# Apply ICA on PCA-transformed data
ica_result = apply_ica(pca_result.components, n_components)

# Select components to keep
selected_components = [True, False, False, False, False, False, False, False, False, False]

# Reconstruct frames from selected ICA components
processed_frames_array = frame_array_from_ica_components(pca_result.model, ica_result, selected_components, video_data_flat.frame_shape)
print("Shape of reconstructed frames:", processed_frames_array.shape)

# Save the reconstructed frames as a video
save_frames_as_video(processed_frames_array, video_data.frame_shape, video_data.fps)
