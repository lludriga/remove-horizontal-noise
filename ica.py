"""Apply ICA to denoise images."""

import os
import tkinter as tk
from abc import ABC, abstractmethod
from tkinter import messagebox, ttk
from typing import Callable, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA, FastICA  # type: ignore


# Define the abstract base class for Component Selector
class ComponentSelector(ABC):
    @abstractmethod
    def select_components(
        self, component_images: np.ndarray, frame_shape: tuple[int, int]
    ) -> List[bool]:
        pass


class IndividualComponentSelectorGUI(ComponentSelector):
    def __init__(self):
        self.selected_components = []

    def select_components(
        self, component_images: np.ndarray, frame_shape: tuple[int, int]
    ) -> List[bool]:
        self.root = tk.Tk()
        self.root.title("ICA Component Selector")

        self.components = component_images
        self.frame_shape = frame_shape
        self.selected_components = [
            tk.BooleanVar(value=True) for _ in range(self.components.shape[1])
        ]

        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.buttons_frame = ttk.Frame(self.root)
        self.buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.next_button = tk.Button(
            self.buttons_frame, text="Next", command=self.next_component
        )
        self.next_button.pack(side=tk.RIGHT)

        self.prev_button = tk.Button(
            self.buttons_frame, text="Previous", command=self.prev_component
        )
        self.prev_button.pack(side=tk.RIGHT)

        self.save_button = tk.Button(
            self.buttons_frame, text="Save", command=self.save_selection
        )
        self.save_button.pack(side=tk.RIGHT)

        self.finish_button = tk.Button(
            self.buttons_frame, text="Finish", command=self.finish_selection
        )
        self.finish_button.pack(side=tk.RIGHT)

        self.component_index = 0
        self.plot_component()

        self.root.mainloop()
        return [var.get() for var in self.selected_components]

    def plot_component(self):
        # Close the previous figure if it exists
        if hasattr(self, "canvas") and self.canvas:
            plt.close(self.canvas.figure)  # Close the current figure
        if hasattr(self, "canvas"):
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots()
        component_image = self.components[:, self.component_index].reshape(
            self.frame_shape
        )
        ax.imshow(component_image, cmap="gray")
        ax.set_title(f"Component {self.component_index + 1}")

        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_button_state()
        # Update the save button color based on selection state
        if self.selected_components[self.component_index].get():
            self.save_button.config(bg="lightgreen", text="Deselect")
        else:
            self.save_button.config(bg="lightcoral", text="Select")

    def update_button_state(self):
        self.prev_button.config(
            state=tk.NORMAL if self.component_index > 0 else tk.DISABLED
        )
        self.next_button.config(
            state=(
                tk.NORMAL
                if self.component_index < self.components.shape[1] - 1
                else tk.DISABLED
            )
        )

    def next_component(self):
        self.component_index += 1
        self.plot_component()

    def prev_component(self):
        self.component_index -= 1
        self.plot_component()

    def save_selection(self):
        self.selected_components[self.component_index].set(
            not self.selected_components[self.component_index].get()
        )
        self.plot_component()

    def finish_selection(self):
        self.root.quit()
        self.root.destroy()


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


class Preprocessor(ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reverse_process(
        self, data: np.ndarray, original_data: np.ndarray
    ) -> np.ndarray:
        pass


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


class VideoProcessor:
    def __init__(
        self,
        video_path: str,
        preprocessors: list[Preprocessor],
        analysis: list[ComponentAnalysis],
        selector: ComponentSelector,
    ):
        self.video_path = video_path
        self.flat_frames: np.ndarray
        self.frame_shape: tuple[int, int]
        self.fps: int

        self.preprocessors: list[Preprocessor] = preprocessors
        self.analysis: list[ComponentAnalysis] = analysis
        self.selector = selector

    def load_video(self) -> np.ndarray:
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
            return np.array(frames)
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

    def variance_flat_image_from_flat_frames(
        self,
        flat_frames: np.ndarray,
    ) -> np.ndarray:
        """Create an image whose pixels represent each pixel variance through the video, normalized."""
        flat_image = np.var(flat_frames, axis=0)
        flat_image = (flat_image / np.max(flat_image)) * 255
        return flat_image

    def variance_images_array(self) -> np.ndarray:
        last_analysis = self.analysis[-1]
        if last_analysis.n_components is None:
            raise Exception("Run the analysis before plotting it.")
        array = np.zeros((self.flat_frames.shape[1], last_analysis.n_components))
        for i in range(last_analysis.n_components):
            array[:, i] = self.variance_flat_image_from_flat_frames(
                self.compose_nth_component(i)
            )
        return array

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

    def select_components_interactive(self) -> List[bool]:
        if not self.analysis:
            raise Exception("Please add an analysis method.")
        selected_components = self.selector.select_components(
            self.variance_images_array(), self.frame_shape
        )
        return selected_components


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


# TODO Choose pca components based on covariance (at least 80%)
def filter_video_ica_black_lines_interactively(
    video_path: str,
    n_components: int,
    variance_threshold: float,
    output_path: str = "output_video.mp4",
) -> None:
    """Full pipeline for filtering a video from its ICA components and saving it."""
    # Initialize VideoProcessor with preprocessors and analysis methods
    video_processor = VideoProcessor(
        video_path,
        preprocessors=[MeanFramesSelector()],
        analysis=[
            PCAAnalysis(
                n_components=n_components, variance_threshold=variance_threshold
            ),
            ICAAnalysis(),
        ],
        selector=IndividualComponentSelectorGUI(),
    )

    # Load video
    video_processor.load_video()

    # Decompose video
    video_processor.decompose_video()

    # Reconstruct the selected frames with the selected ICA components
    reconstructed_selected_frames = video_processor.compose_video()

    # Save the full reconstructed video
    video_processor.save_flat_frames_as_mp4(reconstructed_selected_frames, output_path)


video = VideoProcessor(
    "0.avi",
    [MeanFramesSelector()],
    [PCAAnalysis(3), ICAAnalysis()],
    IndividualComponentSelectorGUI(),
)
video.load_video()

video.decompose_video()

selected_components = video.select_components_interactive()

print(selected_components)

# video.save_flat_frames_as_mp4(video.compose_video(), "output_video_object.mp4")
