"""Hilbert huang trasnform"""

import emd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from ica_joan import *

video_path = "0.mp4"
video_data = process_video(video_path)
video_data.flat_frames = video_data.flat_frames[0:10, :]

print(emd.sift.mask_sift(video_data.flat_frames[0], max_imfs=30).shape)

print(video_data.flat_frames.shape)

reconstructed_video = np.array(
    [np.sum(emd.sift.mask_sift(x, max_imfs=30), axis = 1) for x in video_data.flat_frames]
).reshape(-1, *video_data.frame_shape)
print(reconstructed_video.shape)

reconstructed_video = cv2.normalize(reconstructed_video, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # type: ignore

save_video_as_mp4(reconstructed_video, video_data.frame_shape, video_data.fps)
