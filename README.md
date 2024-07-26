# Video Processing Pipeline

## Overview

This project provides a comprehensive pipeline for video denoising using various techniques, including Fourier Transform, Principal Component Analysis (PCA), and Independent Component Analysis (ICA). The primary focus is on the VideoAnalyzer class, which coordinates the main video processing pipeline.

## File Descriptions

### fourier_transform.py

This file implements the denoising of videos by removing horizontal line noise using the Fourier Transform. The primary function remove_horizontal_noise processes each frame of a video to filter out the noise.
Key Functions:

* remove_horizontal_noise(image): Removes horizontal line noise from an image using the Fourier Transform.
* Video Processing Script: Reads frames from an input video, applies the remove_horizontal_noise function to each frame, and writes the processed frames to an output video.

### ica.py

This file contains the implementation of PCA and ICA for video denoising, including GUI components for selecting components to retain in the final reconstruction.
Key Classes and Functions:

* ComponentSelector (Abstract Base Class): Defines the interface for selecting components from ICA/PCA analysis.
* ComponentSelectorGUI: A GUI for manually selecting components one by one.
* ComponentAnalysis (Abstract Base Class): Defines the interface for component analysis.
* PCAAnalysis: Implements PCA, including decomposition and recomposition of data.
* ICAAnalysis: Implements ICA, including decomposition and recomposition of data.
* Preprocessor (Abstract Base Class): Defines the interface for preprocessing data.
* MeanFramesSelector: Selects frames based on brightness mean to filter out non-noisy frames.

#### VideoAnalyzer Class

The VideoAnalyzer class, which is assumed to be the primary orchestrator of the video processing pipeline, leverages the functionalities provided in fourier_transform.py and ica.py.
Key Responsibilities:

1. Reading and Writing Videos: Handles video input and output using OpenCV.
2. Preprocessing: Applies preprocessing steps like converting to grayscale and selecting frames based on brightness mean.
3. Denoising: Applies Fourier Transform, PCA, and ICA for denoising the video frames.
4. Component Selection: Utilizes the GUI for selecting components to retain in the final reconstruction.

#### Example Usage:

Below is an example of how the VideoAnalyzer class can be used.

``` python
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
    # Two options: component maps and variance images (more expensive)
    is_component_selected = ComponentSelectorGUI().select_components(
        analyzer.get_component_maps()
    )
    save_is_component_selected_json(is_component_selected, video_path)

    video.save_frames_as_mp4(
        analyzer.compose_video(is_component_selected=is_component_selected), output_path
    )

    non_selected_components = [not selected for selected in is_component_selected]
    video.save_frames_as_mp4(
        analyzer.compose_video(is_component_selected=non_selected_components),
        "non_selected.mp4",
    )

```


## Example Usage

video_analyzer = VideoAnalyzer("input_video.mp4", "output_video.mp4")
video_analyzer.process_video()

Main Pipeline

1. Initialization: Create an instance of VideoAnalyzer with the input and output video paths.
2. Reading Frames: Read frames from the input video.
3. Preprocessing: Preprocess the frames to select the noisy ones.
4. Fourier Transform: Apply Fourier Transform to remove horizontal noise.
5. PCA: Apply PCA to decompose and recompose the frames.
6. ICA: Apply ICA to decompose and recompose the frames.
7. Writing Frames: Write the processed frames to the output video.

Conclusion

This pipeline provides a robust framework for denoising video data using a combination of Fourier Transform, PCA, and ICA. The VideoAnalyzer class coordinates the entire process, ensuring seamless integration and processing of video frames.
