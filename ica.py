"""Apply ICA to denoise images."""

import os
import tkinter as tk
from abc import ABC, abstractmethod
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


class ComponentAnalysis(ABC):
    def __init__(self, n_components: int | None = None):
        self.n_components: int | None = None

    @abstractmethod
    def decompose(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def compose(
        self, components: np.ndarray | None = None, selected_components: List[bool] = []
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compose_nth_component(self, component: int) -> np.ndarray:
        pass


class Preprocessor(ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reverse_process(
        self, data: np.ndarray, original_data: np.ndarray
    ) -> np.ndarray:
        pass


class VideoProcessor:
    def __init__(
        self,
        video_path: str,
        preprocessors: list[Preprocessor] = [],
        analysis: list[ComponentAnalysis] = [],
    ):
        self.video_path = video_path
        self.flat_frames: np.ndarray
        self.frame_shape: tuple[int, int]
        self.fps: int

        self.preprocessors: list[Preprocessor] = preprocessors
        self.analysis: list[ComponentAnalysis] = analysis

    def load_video(self) -> VideoData:
        """Load the video and return a flattened grayscale version."""
        cap = cv2.VideoCapture(self.video_path)
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
            self.flat_frames = np.array(frames)
            self.frame_shape = frame_shape
            self.fps = fps
            return VideoData(np.array(frames), frame_shape, fps)
        else:
            raise Exception("Error getting the frame shape. Incorrect dimensions")

    def save_flat_frames_as_mp4(
        self,
        flat_frames: np.ndarray,
        output_path: str = "output_video.mp4",
    ) -> None:
        """Normalize and save array flattened of frames as a video file."""
        frames: np.ndarray = flat_frames.reshape(-1, *self.frame_shape)

        if np.ndim(frames) == 3:
            frames = cv2.normalize(frames, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # type: ignore
        else:
            raise Exception("Invalid dimensions for video when reshaping.")

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.frame_shape[1], self.frame_shape[0]),
            isColor=False,
        )

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"Video saved as {output_path}")

    def variance_image_from_flat_frames(
        self,
        flat_frames: np.ndarray,
    ) -> np.ndarray:
        """Create an image whose pixels represent each pixel variance through the video, normalized."""
        flat_image = np.var(flat_frames, axis=0)
        flat_image = (flat_image / np.max(flat_image)) * 255
        return flat_image.reshape(*self.frame_shape)

    def preprocess_video(self) -> np.ndarray:
        data = self.flat_frames
        if self.preprocessors:
            for preprocessor in self.preprocessors:
                data = preprocessor.process(data)
        return data

    def reverse_preprocess_video(self, data) -> np.ndarray:
        if self.preprocessors:
            for preprocessor in reversed(self.preprocessors):
                data = preprocessor.reverse_process(data, self.flat_frames)
        return data

    def decompose_video(self):
        if not self.analysis:
            raise Exception("Please add the component analysis you want to make.")
        data = self.preprocess_video()
        for analysis in self.analysis:
            data = analysis.decompose(data)

    def compose_video(self, selected_components: list[bool] = []) -> np.ndarray:
        if not self.analysis:
            raise Exception("Please add the component analysis you want to make.")
        data = self.analysis[-1].compose(selected_components=selected_components)
        for analysis in reversed(self.analysis[:-1]):
            data = analysis.compose(data)
        data = self.reverse_preprocess_video(data)
        return data

    def compose_nth_component(
        self, component: int, reverse_preprocessing: bool = False
    ) -> np.ndarray:
        if not self.analysis:
            raise Exception("Please add the component analysis you want to make.")
        data = self.analysis[-1].compose_nth_component(component)
        for analysis in reversed(self.analysis[:-1]):
            data = analysis.compose(data)
        if reverse_preprocessing:
            data = self.reverse_preprocess_video(data)
        return data

    def save_components_videos(self) -> None:
        last_analysis = self.analysis[-1]
        if last_analysis.n_components is None:
            raise Exception(
                "Please decompose the video before trying to compose it again."
            )
        for i in range(last_analysis.n_components):
            self.save_flat_frames_as_mp4(
                self.compose_nth_component(i), f"component_{i}.mp4"
            )


class PCAAnalysis(ComponentAnalysis):
    def __init__(
        self, n_components: int | None = None, variance_threshold: float = 0.8
    ):
        super().__init__(n_components)
        self.model: PCA = (
            PCA(n_components=n_components, random_state=0)
            if n_components is not None
            else PCA(random_state=0)
        )
        self.n_components: int | None = n_components
        self.variance_threshold = variance_threshold

    def decompose(self, data: np.ndarray) -> np.ndarray:
        print("Applying PCA")
        self.components = self.model.fit_transform(data)

        if isinstance(self.components, np.ndarray):
            self.n_components = self.components.shape[1]
            print("Finished PCA")
            return self.components
        else:
            raise Exception("Failed decomposing to an array")

    def compose(
        self, components: np.ndarray | None = None, selected_components: List[bool] = []
    ) -> np.ndarray:
        if self.components is None or self.n_components is None:
            raise Exception(
                "Before recomposing the PCA components you have to run decompose."
            )
        if not selected_components:
            selected_components = [True for i in range(self.n_components)]
        if len(selected_components) != self.n_components:
            raise Exception("Incorrect length of list of selected components.")

        if components is None:
            components = self.components

        print("Reverting PCA")
        inverted_selection = np.invert(selected_components)
        inverted_indices = [
            i for i, include in enumerate(inverted_selection) if include
        ]
        remaining_components = components.copy()
        remaining_components[:, inverted_indices] = 0
        # Reconstruct, reshape and normalize frames
        data: np.ndarray = self.model.inverse_transform(remaining_components)

        print("PCA reverted")
        return data

    def n_components_from_variance(self, variance_threshold: float) -> int:
        """Get the number of components needed to conserve the specified acumulated variance (between 0 and 1)."""
        cumulative_variance = np.cumsum(self.model.explained_variance_ratio_)
        n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
        return int(n_components)

    def plot_cumulative_variance(self) -> None:
        """
        Plot the cumulative explained variance against the number of PCA components.

        Parameters:
        pca (PCA): The PCA object after fitting the data.
        """
        cumulative_variance = np.cumsum(self.model.explained_variance_ratio_)
        plt.figure(figsize=(8, 6))
        plt.plot(
            np.arange(1, len(cumulative_variance) + 1),
            cumulative_variance,
            marker="o",
            linestyle="-",
        )
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Cumulative Explained Variance by Number of PCA Components")
        plt.grid()
        plt.show()

    def compose_nth_component(self, component: int) -> np.ndarray:
        if self.n_components is None:
            raise Exception(
                "Decompose first the data in it's components to be able to compose them."
            )
        selected_components = [False for i in range(self.n_components)]
        selected_components[component] = True
        video = self.compose(selected_components=selected_components)
        return video


class ICAAnalysis(ComponentAnalysis):
    def __init__(self, n_components: int | None = None, max_iter: int = 1000):
        super().__init__(n_components)
        self.model: FastICA = (
            FastICA(n_components=n_components, random_state=0, max_iter=max_iter)
            if n_components is not None
            else FastICA(random_state=0, max_iter=max_iter)
        )
        self.n_components: int | None = n_components

    def decompose(self, data: np.ndarray) -> np.ndarray:
        print("Applying ICA")
        self.components = self.model.fit_transform(data)

        if isinstance(self.components, np.ndarray):
            self.n_components = self.components.shape[1]
            print("Finished ICA")
            return self.components
        else:
            raise Exception("Failed decomposing to an array")

    def compose(
        self, components: np.ndarray | None = None, selected_components: List[bool] = []
    ) -> np.ndarray:
        if self.components is None or self.n_components is None:
            raise Exception(
                "Before recomposing the ICA components you have to run decompose."
            )
        if not selected_components:
            selected_components = [True for i in range(self.n_components)]
        if len(selected_components) != self.n_components:
            raise Exception("Incorrect length of list of selected components.")

        if components is None:
            components = self.components

        print("Reverting ICA")
        inverted_selection = np.invert(selected_components)
        inverted_indices = [
            i for i, include in enumerate(inverted_selection) if include
        ]
        remaining_components = components.copy()
        remaining_components[:, inverted_indices] = 0
        # Reconstruct, reshape and normalize frames
        data: np.ndarray = self.model.inverse_transform(remaining_components)

        print("ICA reverted")
        return data

    def compose_nth_component(self, component: int) -> np.ndarray:
        if self.n_components is None:
            raise Exception(
                "Decompose first the data in it's components to be able to compose them."
            )
        selected_components = [False for i in range(self.n_components)]
        selected_components[component] = True
        video = self.compose(selected_components=selected_components)
        return video


class MeanFramesSelector(Preprocessor):
    def __init__(self, batch_size: int = 3):
        self.batch_size = batch_size
        self.selected_frames: List[bool] = []

    def process(self, flat_frames: np.ndarray) -> np.ndarray:
        """
        Analyze the frames statistically to try to remove the ones without noise.

        We apply a statistic, the mean in these case, 3 by 3 so we ensure the background
        image stays mostly the same, and we can detect almost surely the variation caused
        by the black lines, making the mean lower.
        We return a list of bools containg true in the ones chosen, the ones with noise
        supposedly. We choose 2 out of 3 every time, since the noise doesn't happen (apparently)
        3 frames.
        """
        n_frames = flat_frames.shape[0]
        remaining_frames = flat_frames.shape[0] % self.batch_size
        self.selected_frames = []

        for i in range(n_frames // self.batch_size):
            batch_mean = np.array(
                [
                    np.mean(flat_frames[i * self.batch_size + j])
                    for j in range(self.batch_size)
                ]
            )
            max_accepted_mean = np.quantile(batch_mean, 0.66)
            self.selected_frames.extend(
                [mean <= max_accepted_mean for mean in batch_mean]
            )

        # Since the remaining batch is smaller than what we analyze
        # we do the easy thing and just include it all
        self.selected_frames.extend([True] * remaining_frames)

        return flat_frames[self.selected_frames]

    def reverse_process(
        self, data: np.ndarray, original_data: np.ndarray
    ) -> np.ndarray:
        """Reconstruct the full video including non-noisy frames in their correct order."""
        full_video = np.zeros_like(original_data)

        # Insert reconstructed frames into the full video array
        j = 0
        for i in range(len(self.selected_frames)):
            if self.selected_frames[i]:
                full_video[i] = data[j]
                j += 1
            else:
                full_video[i] = original_data[i]

        return full_video


# Helper function
def save_frame_as_image(frame: np.ndarray, filename: str) -> None:
    """Save a single frame as an image file."""
    cv2.imwrite(f"intermediate_frames/{filename}", frame)


# Helper function
def load_not_flat_video(video_path: str) -> np.ndarray:
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


# Object Done
def load_video(video_path: str) -> VideoData:
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


# Object done
def apply_pca(frames: np.ndarray, n_components: int) -> DecompositionResult:
    """Apply pca to the input array with n components."""
    print("Applying PCA")
    pca = PCA(n_components=n_components, random_state=0)
    components_pca = pca.fit_transform(frames)
    print(f"Finished PCA. Shape {components_pca.shape}")

    return DecompositionResult(pca, components_pca, n_components)


# Object done
def n_pca_components_from_variance(pca: PCA, variance_threshold: float) -> int:
    """Get the number of components needed to conserve the specified acumulated variance (between 0 and 1)."""
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
    return int(n_components)


# Object done
def plot_cumulative_variance(pca: PCA) -> None:
    """
    Plot the cumulative explained variance against the number of PCA components.

    Parameters:
    pca (PCA): The PCA object after fitting the data.
    """
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 6))
    plt.plot(
        np.arange(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o",
        linestyle="-",
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance by Number of PCA Components")
    plt.grid()
    plt.show()


# Object done
def apply_ica(components: np.ndarray, n_components: int) -> DecompositionResult:
    """Apply ICA to the input array with n components."""
    print("Applying ICA")
    ica = FastICA(n_components=n_components, random_state=0, max_iter=1000)
    components_ica = ica.fit_transform(components)
    print(f"Finished ICA. Shape {components_ica.shape}")

    return DecompositionResult(ica, components_ica, n_components)


# TODO use the covariance to get the pca components used
# Helper function
def apply_pca_and_ica(
    flat_frames: np.ndarray, n_components: int
) -> Tuple[DecompositionResult, DecompositionResult]:
    """Apply PCA and ICA to the selected video data."""
    n_components_pca = n_components
    pca_result = apply_pca(flat_frames, n_components_pca)

    n_components_ica = n_components_pca
    ica_result = apply_ica(pca_result.components, n_components_ica)

    return pca_result, ica_result


# Object done
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


# Object done
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


# Object done
def video_from_nth_pca(
    pca_result: DecompositionResult,
    component: int,
) -> np.ndarray:
    """Create the array of frames for the nth pca component."""
    selected_components = [False for i in range(pca_result.n_components)]
    selected_components[component] = True
    video = video_from_pca(pca_result, selected_components)
    return video


# Object done
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


# Object Done
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


# Object Done
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

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (frame_shape[1], frame_shape[0]), isColor=False
    )

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved as {output_path}")


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


# Object Done
def save_pca_components_videos(
    pca_result: DecompositionResult, video_data: VideoData
) -> None:
    """Save each pca component as an mp4 to visualize them independently."""
    for i in range(pca_result.n_components):
        video = video_from_nth_pca(pca_result, i)
        save_video_as_mp4(
            video,
            video_data.frame_shape,
            video_data.fps,
            f"output_component_{i}.mp4",
        )


# Object Done
def save_ica_components_videos(
    ica_result: DecompositionResult,
    pca_result: DecompositionResult,
    video_data: VideoData,
) -> None:
    """Save each component as an mp4 to visualize them independently."""
    n_final_components = ica_result.components.shape[1]
    for i in range(n_final_components):
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


# Object Done
def analyze_and_select_frames(
    flat_frames: np.ndarray, batch_size: int = 3
) -> List[bool]:
    """
    Analyze the frames statistically to try to remove the ones without noise.

    We apply a statistic, the mean in these case, 3 by 3 so we ensure the background
    image stays mostly the same, and we can detect almost surely the variation caused
    by the black lines, making the mean lower.
    We return a list of bools containg true in the ones chosen, the ones with noise
    supposedly. We choose 2 out of 3 every time, since the noise doesn't happen (apparently)
    3 frames.
    """
    n_frames = flat_frames.shape[0]
    remaining_frames = flat_frames.shape[0] % batch_size
    selected_frames: List[bool] = []

    for i in range(n_frames // batch_size):
        batch_mean = np.array(
            [np.mean(flat_frames[i * batch_size + j]) for j in range(batch_size)]
        )
        max_accepted_mean = np.quantile(batch_mean, 0.66)
        selected_frames.extend([mean <= max_accepted_mean for mean in batch_mean])

    # Since the remaining batch is smaller than what we analyze
    # we do the easy thing and just include it all
    selected_frames.extend([True] * remaining_frames)

    return selected_frames


# Object Done
def variance_image_from_video(
    video: np.ndarray,
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """Create an image whose pixels represent each pixel variance through the video, normalized."""
    flat_image = np.var(video, axis=0)
    flat_image = (flat_image / np.max(flat_image)) * 255
    return flat_image.reshape(*frame_shape)


# Object Done
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


# TODO Choose pca components based on covariance (at least 80%)
def filter_black_lines_video_ica_interactively(
    video_path: str, n_components: int, output_path: str = "output_video.mp4"
) -> None:
    """Full pipeline for filtering a video from it's ica's and saving it."""
    # Load the video into a flat array (a 2D matrix)
    video_data: VideoData = load_video(video_path)

    # Select frames with potential noise
    selected_frames: List[bool] = analyze_and_select_frames(video_data.flat_frames)

    print(
        """The ica anlyisis will be applied to the following frames
    which are most probably noisy, based on the assumption that
    they have black lines of noise (the mean birghtness is lower):"""
    )
    # print(np.array(range(len(selected_frames)))[selected_frames])
    selected_video_data = VideoData(
        video_data.flat_frames[selected_frames], video_data.frame_shape, video_data.fps
    )

    # Apply PCA and ICA
    pca_result, ica_result = apply_pca_and_ica(
        selected_video_data.flat_frames, n_components
    )

    # Review ICA component manually
    selected_ica_components = [True] * n_components

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
        output_path,
    )


video = VideoProcessor("0.avi", [MeanFramesSelector()], [PCAAnalysis(3), ICAAnalysis()])
video.load_video()

video.decompose_video()
video.save_flat_frames_as_mp4(video.compose_video(), "output_video_object.mp4")

save_frame_as_image(
    video.variance_image_from_flat_frames(video.compose_nth_component(0)),
    "component_0.png",
)
save_frame_as_image(
    video.variance_image_from_flat_frames(video.compose_nth_component(1)),
    "component_1.png",
)
video.save_components_videos()
