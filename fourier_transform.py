"""Llevar bandes horitzontals de videos amb la transformada de fourier."""

import numpy as np
import cv2

def remove_horizontal_noise(image):
    # Convert image to float32 for DFT
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Shift zero-frequency component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)

    # Create a mask to filter out horizontal line noise
    rows, cols = image.shape
    mask = np.ones((rows, cols, 2), np.uint8)

    # Remove horizontal lines by zeroing out specific frequencies
    center_column = cols // 2
    center_row = rows // 2
    radius_not_removed = 4
    mask[0:center_row - radius_not_removed, center_column-2:center_column+2] = 0
    mask[center_row + radius_not_removed:rows-1, center_column-2:center_column+2] = 0

    # Apply mask and perform the inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    
    # Get the magnitude
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize the image to the range [0, 255]
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    
    return img_back

# Process video
cap = cv2.VideoCapture('0.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height), isColor=False)


print("Started processing video")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Process each frame
    processed_frame = remove_horizontal_noise(gray_frame)

    # Write the frame to the output video
    out.write(processed_frame)


cap.release()
out.release()

print("Finished processing video")
