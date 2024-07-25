"""Apply PCA and ICA to denoise videos."""

import os
import tkinter as tk
from abc import ABC, abstractmethod
from tkinter import messagebox, ttk
from typing import Callable, Iterable, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import gaussian_filter  # type: ignore
from sklearn.decomposition import PCA, FastICA  # type: ignore
from TensorPCA.tensorpca import TensorPCA  # type: ignore


class ComponentSelector(ABC):
    """Abstract base class for selecting components from ICA/PCA analysis."""

    @abstractmethod
    def select_components(self, component_images: np.ndarray) -> List[bool]:
        """Select components from the provided component images.

        Args:
            component_images (np.ndarray): Array of component images.

        Returns:
            List[bool]: List indicating which components are selected.
        """
        pass


class IndividualComponentSelectorGUI(ComponentSelector):
    """GUI for selecting components one by one."""

    def __init__(self):
        """Initialize the component selector GUI."""
        self.selected_components = []

    def select_components(self, component_images: np.ndarray) -> List[bool]:
        """Launch the GUI to select components from the provided images.

        Args:
            component_images (np.ndarray): Array of component images.

        Returns:
            List[bool]: List indicating which components are selected.
        """
        # Initialize GUI components
        self.root = tk.Tk()
        self.root.title("ICA Component Selector")

        self.components = component_images
        self.selected_components = [
            tk.BooleanVar(value=True) for _ in range(self.components.shape[0])
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
        """Plot the current component image in the GUI."""
        # Close the previous figure if it exists
        if hasattr(self, "canvas") and self.canvas:
            plt.close(self.canvas.figure)  # Close the current figure
        if hasattr(self, "canvas"):
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots()
        component_image = self.components[self.component_index]
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
        """Update the state of navigation buttons."""
        self.prev_button.config(
            state=tk.NORMAL if self.component_index > 0 else tk.DISABLED
        )
        self.next_button.config(
            state=(
                tk.NORMAL
                if self.component_index < self.components.shape[0] - 1
                else tk.DISABLED
            )
        )

    def next_component(self):
        """Display the next component."""
        self.component_index += 1
        self.plot_component()

    def prev_component(self):
        """Display the previous component."""
        self.component_index -= 1
        self.plot_component()

    def save_selection(self):
        """Toggle the selection state of the current component."""
        self.selected_components[self.component_index].set(
            not self.selected_components[self.component_index].get()
        )
        self.plot_component()

    def finish_selection(self):
        """Finalize the component selection and close the GUI."""
        if hasattr(self, "canvas") and self.canvas:
            plt.close(self.canvas.figure)  # Close the last figure
        self.root.quit()
        self.root.destroy()


class ComponentAnalysis(ABC):
    """Abstract base class for component analysis (PCA/ICA)."""

    n_components: Optional[int]

    @abstractmethod
    def decompose(self, data: np.ndarray) -> np.ndarray:
        """Decompose the data into components.

        Args:
            data (np.ndarray): Data to decompose.

        Returns:
            np.ndarray: Decomposed components.
        """
        pass

    @abstractmethod
    def compose(
        self,
        components: np.ndarray | None = None,
        selected_components: Optional[list[bool]] = None,
    ) -> np.ndarray:
        """Recompose data from the selected components.

        Args:
            components (np.ndarray, optional): Components to recompose. Defaults to None.
                           If none, recompose from the decomposed components saved in the class.
            selected_components (Optional[list[bool]], optional): List of selected components. Defaults to None.
                           If none, use all of them

        Returns:
            np.ndarray: Reconstructed data.
        """
        pass

    @abstractmethod
    def compose_nth_component(self, component: int) -> np.ndarray:
        """Recompose data from a specific component.

        Args:
            component (int): Component index.

        Returns:
            np.ndarray: Reconstructed data from the specified component.
        """
        pass


class PCAAnalysis(ComponentAnalysis):
    """Perform PCA (Principal Component Analysis) on data."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: Optional[float] = None,
    ):
        """Initialize PCAAnalysis with number of components or variance threshold.

        Args:
            n_components (Optional[int], optional): Number of components to retain. Defaults to None.
                                                    If None, use the variance threshold
            variance_threshold (Optional[float], optional): Variance threshold to retain. Defaults to None.
                                                    If None and n_components None, use all of them.
        """
        self.n_components = n_components
        self.model: PCA = PCA(
            n_components=(
                n_components if variance_threshold is None else variance_threshold
            ),
            random_state=0,
        )

        self.n_components: Optional[int] = n_components
        self.variance_threshold: Optional[float] = variance_threshold

    def decompose(self, data: np.ndarray) -> np.ndarray:
        """Decompose data into principal components using PCA.

        Args:
            data (np.ndarray): Data to decompose.

        Returns:
            np.ndarray: Decomposed components.
        """
        print("Applying PCA")
        self.components = self.model.fit_transform(data)
        print(f"PCA shape {self.components.shape}")

        if isinstance(self.components, np.ndarray):
            self.n_components = self.components.shape[1]
            print(f"Finished PCA, {self.n_components} components extracted.")
            return self.components
        else:
            raise Exception("Failed decomposing to an array")

    def compose(
        self,
        components: np.ndarray | None = None,
        selected_components: Optional[list[bool]] = None,
    ) -> np.ndarray:
        """Recompose data from the selected PCA components.

        Args:
            components (np.ndarray, optional): Matrix components to recompose. Defaults to None.
                           If none, recompose from the decomposed components saved in the class.
            selected_components (Optional[list[bool]], optional): List of selected components. Defaults to None.
                           If none, use all of them.

        Returns:
            np.ndarray: Reconstructed data.
        """
        if self.components is None or self.n_components is None:
            raise ValueError(
                "Before recomposing the PCA components you have to run decompose."
            )
        if selected_components is None:
            selected_components = [True for i in range(self.n_components)]
        if len(selected_components) != self.n_components:
            raise ValueError("Incorrect length of list of selected components.")

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
        """Get the number of components needed to conserve the specified accumulated variance.

        Args:
            variance_threshold (float): Variance threshold to retain.

        Returns:
            int: Number of components needed to conserve the specified variance.
        """
        cumulative_variance = np.cumsum(self.model.explained_variance_ratio_)
        n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
        return int(n_components)

    def plot_cumulative_variance(self) -> None:
        """Plot the cumulative explained variance against the number of PCA components."""
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
        """Recompose data from the nth principal component.

        Args:
            component (int): Component index.

        Returns:
            np.ndarray: Reconstructed data from the specified component.
        """
        if self.n_components is None:
            raise ValueError(
                "Decompose first the data in it's components to be able to compose them."
            )
        selected_components = [False for i in range(self.n_components)]
        selected_components[component] = True
        video = self.compose(selected_components=selected_components)
        return video


class ICAAnalysis(ComponentAnalysis):
    """Perform ICA (Independent Component Analysis) on data."""

    def __init__(self, n_components: Optional[int] = None, max_iter: int = 1000):
        """Initialize ICAAnalysis with the number of components.

        Args:
            n_components (int): Number of components to use. If None use them all.
        """
        self.n_components: Optional[int] = n_components
        self.model: FastICA = FastICA(
            n_components=n_components, random_state=0, max_iter=max_iter
        )

    def decompose(self, data: np.ndarray) -> np.ndarray:
        """Decompose data into independent components using ICA.

        Args:
            data (np.ndarray): Data to decompose.

        Returns:
            np.ndarray: Decomposed components.
        """
        print("Applying ICA")
        self.components = self.model.fit_transform(data)

        if isinstance(self.components, np.ndarray):
            self.n_components = self.components.shape[1]
            print("Finished ICA")
            return self.components
        else:
            raise Exception("Failed decomposing to an array")

    def compose(
        self,
        components: Optional[np.ndarray] = None,
        selected_components: Optional[list[bool]] = None,
    ) -> np.ndarray:
        """Recompose data from the selected ICA components.

        Args:
            components (np.ndarray, optional): Matrix components to recompose. Defaults to None.
                           If none, recompose from the decomposed components saved in the class.
            selected_components (Optional[list[bool]], optional): List of selected components. Defaults to None.
                           If none, use all of them.

        Returns:
            np.ndarray: Reconstructed data.
        """
        if self.components is None or self.n_components is None:
            raise ValueError(
                "Before recomposing the ICA components you have to run decompose."
            )
        if selected_components is None:
            selected_components = [True for i in range(self.n_components)]
        if len(selected_components) != self.n_components:
            raise ValueError("Incorrect length of list of selected components.")

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
        """Recompose data from the nth independent component.

        Args:
            component (int): Component index.

        Returns:
            np.ndarray: Reconstructed data from the specified component.
        """
        if self.n_components is None:
            raise ValueError(
                "Decompose first the data in it's components to be able to compose them."
            )
        selected_components = [False for i in range(self.n_components)]
        selected_components[component] = True
        video = self.compose(selected_components=selected_components)
        return video


# Preprocessors operate on non-flat data
class Preprocessor(ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reverse_process(self, data: np.ndarray) -> np.ndarray:
        pass


class MeanFramesSelector(Preprocessor):
    def __init__(self, batch_size: int = 3):
        self.batch_size = batch_size
        self.selected_frames: List[bool] = []
        self.original_not_used_frames: np.ndarray

    def process(self, frames: np.ndarray) -> np.ndarray:
        """
        Analyze the frames statistically to try to remove the ones without noise.

        We apply a statistic, the mean in these case, 3 by 3 so we ensure the background
        image stays mostly the same, and we can detect almost surely the variation caused
        by the black lines, making the mean lower.
        We return a list of bools containg true in the ones chosen, the ones with noise
        supposedly. We choose 2 out of 3 every time, since the noise doesn't happen (apparently)
        3 frames.
        """
        print("Selecting possible noisy frames from the brightness mean.")
        n_frames = frames.shape[0]
        remaining_frames = frames.shape[0] % self.batch_size
        self.selected_frames = []

        for i in range(n_frames // self.batch_size):
            batch_mean = np.array(
                [
                    np.mean(frames[i * self.batch_size + j])
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

        self.original_not_used_frames = frames[
            [not selected for selected in self.selected_frames]
        ]

        print(
            f"Finished selection. Selected {sum(map(lambda x: 1 if x else 0, self.selected_frames))} frames"
        )
        return frames[self.selected_frames]

    def reverse_process(self, data: np.ndarray) -> np.ndarray:
        """Reconstruct the full video including non-noisy frames in their correct order."""
        print("Joining the selected frames with the original non-processed frames.")
        full_video = np.zeros((len(self.selected_frames), data.shape[1], data.shape[2]))

        # Insert reconstructed frames into the full video array
        j = 0
        k = 0
        for i in range(len(self.selected_frames)):
            if self.selected_frames[i]:
                full_video[i] = data[j]
                j += 1
            else:
                full_video[i] = self.original_not_used_frames[k]
                k += 1

        return full_video


class GaussianSpatialFilter(Preprocessor):
    def __init__(self, sigma: float = 1):
        self.sigma = sigma

    def process(self, data: np.ndarray) -> np.ndarray:
        print("Applying the spatial gaussian mean filter to each frame.")
        return gaussian_filter(data, sigma=self.sigma, axes=(1, 2))

    def reverse_process(self, data: np.ndarray) -> np.ndarray:
        return data


class VideoData:
    """Class to handle loading and saving of video data."""

    def __init__(
        self,
        video_path: str,
    ):
        """Initialize VideoData with the provided video path.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path
        self.frames: np.ndarray
        self.frame_shape: tuple[int, int]
        self.fps: int

    def load_video(self) -> np.ndarray:
        """Load the video and return a flattened grayscale version.

        Returns:
            np.ndarray: Array of grayscale frames.

        Raises:
            ValueError: If the video cannot be loaded or frame shape is incorrect.
        """
        cap = cv2.VideoCapture(self.video_path)
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

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        if isinstance(frame_shape, tuple) and len(frame_shape) == 2:
            self.frames = np.array(frames)
            self.frame_shape = frame_shape
            self.fps = fps
            return np.array(frames)
        else:
            raise ValueError("Error getting the frame shape. Incorrect dimensions")

    def save_flat_frames_as_mp4(
        self,
        flat_frames: np.ndarray,
        output_path: str = "output_video.mp4",
    ) -> None:
        """Save an array of flat frames as a video file.

        Args:
            flat_frames (np.ndarray): Array of video flat frames.
            output_path (str): Path to save the output video.

        Raises:
            ValueError: If the frames reshaped array is not 3-dimensional.
        """
        frames: np.ndarray = flat_frames.reshape(-1, *self.frame_shape)
        self.save_frames_as_mp4(frames, output_path)

    def save_frames_as_mp4(
        self,
        frames: np.ndarray,
        output_path: str = "output_video.mp4",
    ) -> None:
        """Save an array of frames as a video file.

        Args:
            frames (np.ndarray): Array of video frames.
            output_path (str): Path to save the output video.

        Raises:
            ValueError: If the frames array is not 3-dimensional.
        """
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


class VideoAnalyzer:
    def __init__(
        self,
        video_data: VideoData,
        preprocessors: List[Preprocessor],
        pca: PCAAnalysis,
        ica: ICAAnalysis,
        selector: ComponentSelector = IndividualComponentSelectorGUI(),
    ):
        self.video_data = video_data
        self.preprocessors = preprocessors
        self.pca = pca
        self.ica = ica
        self.selector = selector

    def variance_images_array(self) -> np.ndarray:
        if self.ica.n_components is None:
            raise ValueError(
                "Run the component analysis before operating or composing on it."
            )
        array = np.zeros((self.ica.n_components, *self.video_data.frame_shape))
        for i in range(self.ica.n_components):
            array[i] = variance_image_from_frames(self.compose_nth_component_video(i))
        return array

    def preprocess_video(self) -> np.ndarray:
        data = self.video_data.frames
        if self.preprocessors:
            for preprocessor in self.preprocessors:
                data = preprocessor.process(data)
        return data

    def reverse_preprocess_video(self, data) -> np.ndarray:
        if self.preprocessors:
            for preprocessor in reversed(self.preprocessors):
                data = preprocessor.reverse_process(data)
        return data

    def decompose_video(self):
        data = self.preprocess_video()
        data = flatten(data)
        data = self.pca.decompose(data)
        data = self.ica.decompose(data)

    def decompose_video_pca_only(self):
        data = self.preprocess_video()
        data = flatten(data)
        data = self.pca.decompose(data)

    def compose_video(
        self, selected_components: Optional[list[bool]] = None
    ) -> np.ndarray:
        data = self.ica.compose(selected_components=selected_components)
        data = self.pca.compose(data)
        data = reverse_flatten(data, self.video_data.frame_shape)
        data = self.reverse_preprocess_video(data)
        return data

    def compose_nth_component_video(
        self, component: int, reverse_preprocessing: bool = False
    ) -> np.ndarray:
        data = self.ica.compose_nth_component(component)
        data = self.pca.compose(data)
        data = reverse_flatten(data, self.video_data.frame_shape)
        if reverse_preprocessing:
            data = self.reverse_preprocess_video(data)
        return data

    def save_components_videos(self) -> None:
        if self.ica.n_components is None:
            raise ValueError(
                "Please decompose the video before trying to compose it again."
            )
        for i in range(self.ica.n_components):
            self.video_data.save_frames_as_mp4(
                self.compose_nth_component_video(i), f"component_{i}.mp4"
            )

    def select_components(self) -> List[bool]:
        selected_components = self.selector.select_components(
            self.variance_images_array()
        )
        return selected_components

    def choose_n_pca_components_interactive(self):
        self.decompose_video_pca_only()
        self.pca.plot_cumulative_variance()

        chosen: bool = False
        while not chosen:
            variance_threshold = float(
                input("How much variance would you like to keep? ")
            )
            n_components = self.pca.n_components_from_variance(variance_threshold)
            print(f"That would need {n_components} number of components.")
            print("Do you want to get that many number of components?")
            chosen = ask_y_or_n()

        self.pca = PCAAnalysis(n_components)


def save_frame_as_image(frame: np.ndarray, filename: str) -> None:
    """Save a single frame as an image file."""
    os.makedirs("intermediate_frames/", exist_ok=True)
    cv2.imwrite(f"intermediate_frames/{filename}", frame)


def variance_image_from_frames(
    frames: np.ndarray,
) -> np.ndarray:
    """Create an image whose pixels represent each pixel variance through the video, normalized."""
    image = np.var(frames, axis=0)
    image = (image / np.max(image)) * 255
    return image


def flatten(data: np.ndarray) -> np.ndarray:
    """Reshape video frames (3 dimensions) in a matrix for PCA and ICA.

    Args:
        frames (np.ndarray): Array of video frames.

    Returns:
        np.ndarray: Reshaped frames matrix.
    """
    return data.reshape(len(data), -1)


def reverse_flatten(data: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """Reshape flat frames matrix back to original frame shape.

    Args:
        components (np.ndarray): Matrix of flat frames.
        frame_shape (tuple[int, int]): Original frame shape.

    Returns:
        np.ndarray: Reshaped frames, into video (3 dimensions: 1 for time 2 for spatial (pixels)).
    """
    return data.reshape(-1, *frame_shape)


def ask_y_or_n() -> bool:
    """Ask interactively for yes or no, and return True or False respectively."""
    while True:
        answer = input("Enter y or n (yes or no): ")
        if answer == "y":
            return True
        elif answer == "n":
            return False
        else:
            print("Please enter y or n.")


# Example usage
def main(
    video_path: str,
    output_path: str = "output_video.mp4",
) -> None:
    """Full pipeline for filtering a video from its ICA components and saving it."""
    # Initialize VideoProcessor with preprocessors and analysis methods
    video = VideoData(
        video_path,
    )

    # Load video
    video.load_video()

    analyzer = VideoAnalyzer(
        video,
        [GaussianSpatialFilter()],
        PCAAnalysis(),
        ICAAnalysis(),
    )

    # Choose how many components to use
    analyzer.choose_n_pca_components_interactive()

    # Decompose video
    analyzer.decompose_video()

    # Select ICA components
    selected_components = analyzer.select_components()
    video.save_frames_as_mp4(
        analyzer.compose_video(selected_components=selected_components), output_path
    )

    non_selected_components = [not selected for selected in selected_components]
    video.save_frames_as_mp4(
        analyzer.compose_video(selected_components=non_selected_components),
        "non_selected.mp4",
    )


if __name__ == "__main__":
    # 0.93 de variança 122 components
    # 0.95 de variança 288 comopoents
    main("0.avi")
