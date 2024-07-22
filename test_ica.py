import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock
from ica import (
    IndividualComponentSelectorGUI,
    VideoProcessor,
    PCAAnalysis,
    ICAAnalysis,
    MeanFramesSelector,
)


@pytest.fixture
def sample_video_data():
    """Fixture to provide sample video data."""
    video_path = "test_video.avi"
    frame_shape = (100, 100)
    frames = np.random.randint(0, 256, (10, 10000), dtype=np.uint8)
    video_processor = VideoProcessor(
        video_path,
        preprocessors=[],
        analysis=[],
        selector=MagicMock()
    )
    video_processor.flat_frames = frames
    video_processor.frame_shape = frame_shape
    video_processor.fps = 30
    return video_processor


def test_mean_frames_selector_process(sample_video_data):
    """Test the MeanFramesSelector process method."""
    preprocessor = MeanFramesSelector(batch_size=3)
    processed_frames = preprocessor.process(sample_video_data.flat_frames)
    assert processed_frames.shape[0] == (sample_video_data.flat_frames.shape[0] // 3) * 2 + 1


def test_pca_analysis(sample_video_data):
    """Test PCA analysis methods."""
    pca = PCAAnalysis(n_components=3)
    decomposed_data = pca.decompose(sample_video_data.flat_frames)
    assert decomposed_data.shape[1] == 3
    composed_data = pca.compose(decomposed_data)
    assert composed_data.shape == sample_video_data.flat_frames.shape


def test_ica_analysis(sample_video_data):
    """Test ICA analysis methods."""
    ica = ICAAnalysis(n_components=3)
    decomposed_data = ica.decompose(sample_video_data.flat_frames)
    assert decomposed_data.shape[1] == 3
    composed_data = ica.compose(decomposed_data)
    assert composed_data.shape == sample_video_data.flat_frames.shape


def test_video_processor_load_video(sample_video_data):
    """Test the video loading functionality."""
    video_processor = sample_video_data
    assert video_processor.flat_frames.shape == (10, 10000)
    assert video_processor.frame_shape == (100, 100)
    assert video_processor.fps == 30


def test_video_processor_save_flat_frames_as_mp4(sample_video_data, tmpdir):
    """Test saving the video frames as an MP4 file."""
    video_processor = sample_video_data
    output_path = tmpdir.join("test_output_video.mp4")
    video_processor.save_flat_frames_as_mp4(video_processor.flat_frames, str(output_path))
    assert output_path.check()


def test_individual_component_selector_gui():
    """Test the GUI component selector (limited test)."""
    selector = IndividualComponentSelectorGUI()
    component_images = np.random.rand(10000, 10)  # Mock component images
    frame_shape = (100, 100)
    selected_components = selector.select_components(component_images, frame_shape)
    assert len(selected_components) == 10  # Assuming 20 components
