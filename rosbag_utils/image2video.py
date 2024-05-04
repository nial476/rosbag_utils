import cv2
import numpy as np

fc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("filter.mp4", fc, 0.25, (500, 500))

for idx in range(10):
    color = np.random.randint(0, 255, size=3)
    if idx in [0, 2, 3]:  # only 3 frames will be in the final video
        image = np.full((500, 500, 3), fill_value=color, dtype=np.uint8)
    else:
        # slighly different size
        image = np.full((400, 500, 3), fill_value=color, dtype=np.uint8)

    video.write(image)