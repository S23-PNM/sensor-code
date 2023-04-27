import sys
import cv2
import numpy as np
from collections import deque
from typing import List, Optional
import time
import ArducamDepthCamera as ac

print(dir(ac))

MAX_DISTANCE = 4

FRAME_BUFFER_SIZE = 30
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

first_frame_mean_depths = None
running_mean_depths = deque(maxlen=FRAME_BUFFER_SIZE)
frame_height = 180
frame_width = 240


def process_frame(depth_buf: np.ndarray, amplitude_buf: np.ndarray) -> np.ndarray:
    depth_buf = np.nan_to_num(depth_buf)

    amplitude_buf[amplitude_buf <= 7] = 0
    amplitude_buf[amplitude_buf > 7] = 255

    depth_buf = (1 - (depth_buf / MAX_DISTANCE)) * 255
    depth_buf = np.clip(depth_buf, 0, 255)
    result_frame = depth_buf.astype(np.uint8) & amplitude_buf.astype(np.uint8)
    return result_frame


def get_mean_depth(frame: np.ndarray) -> List[float]:
    region_width = frame.shape[1] // 3
    mean_depths = [
        round(np.mean(frame[:, region_width * i : region_width * (i + 1)]), 2)
        for i in range(3)
    ]
    return mean_depths


def analyze_movement(depth_threshold: float = 10) -> Optional[str]:
    global first_frame_mean_depths, running_mean_depths, frame_buffer

    if len(frame_buffer) < 30:
        return None

    if first_frame_mean_depths is None:
        first_frame_mean_depths = get_mean_depth(frame_buffer[-1])

    current_frame = frame_buffer[-1]
    current_mean_depths = get_mean_depth(current_frame)
    running_mean_depths.append(current_mean_depths)

    max_mean_depths = np.max(running_mean_depths, axis=0)
    depth_diffs = (max_mean_depths - first_frame_mean_depths).tolist()
    # direction = np.sign(depth_diffs)

    # for i, depth_diff in enumerate(depth_diffs):

    for diff in depth_diffs:
        if diff < depth_threshold:
            return None

    # Calculate the overall trend of movement
    # print(depth_diffs)
    # print(running_mean_depths)
    arr = np.array(running_mean_depths)
    arr = np.rot90(arr)
    arr = np.flip(arr, axis=1)
    # print(arr)
    arr = np.argmax(arr, axis=1).tolist()
    for i in arr:
        if i == 0:
            return None
    first = arr[0]
    last = arr[-1]
    frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
    running_mean_depths = deque(maxlen=FRAME_BUFFER_SIZE)
    if last > first:
        return "right"
    else:
        return "left"

    # print(first_frame_mean_depths)
    # print(max_mean_depths)


def create_colored_regions(mean_depths: List[float]) -> np.ndarray:
    region_width = frame_width // 3
    colored_regions = np.zeros((frame_height, 240), dtype=np.uint8)
    for i, mean_depth in enumerate(mean_depths):
        colored_regions[:, region_width * i : region_width * (i + 1)] = mean_depth
    return colored_regions


if __name__ == "__main__":
    cam = ac.ArducamCamera()
    if cam.init(ac.TOFConnect.CSI, 0) != 0:
        print("initialization failed")
    if cam.start(ac.TOFOutput.DEPTH) != 0:
        print("Failed to start camera")
    cam.setControl(ac.TOFControl.RANG, MAX_DISTANCE)
    cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("colored_regions", cv2.WINDOW_AUTOSIZE)
    while True:
        frame = cam.requestFrame(200)
        if frame != None:
            depth_buf = frame.getDepthData()
            amplitude_buf = frame.getAmplitudeData()
            cam.releaseFrame(frame)
            amplitude_buf *= 255 / 1024
            amplitude_buf = np.clip(amplitude_buf, 0, 255)
            cv2.imshow("preview_amplitude", amplitude_buf.astype(np.uint8))
            result_image = process_frame(depth_buf, amplitude_buf)
            frame_buffer.append(result_image)
            line_1_x = frame_width // 3
            line_2_x = (frame_width * 2) // 3
            color = (0, 255, 0)  # Green color in BGR format
            thickness = 1
            cv2.line(
                result_image, (line_1_x, 0), (line_1_x, frame_height), color, thickness
            )
            cv2.line(
                result_image, (line_2_x, 0), (line_2_x, frame_height), color, thickness
            )
            result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_JET)
            cv2.imshow("preview", result_image)
            mean_depths = get_mean_depth(frame_buffer[-1])
            colored_regions = create_colored_regions(mean_depths)
            colored_regions = cv2.applyColorMap(colored_regions, cv2.COLORMAP_JET)
            cv2.imshow("colored_regions", colored_regions)
            movement_direction = analyze_movement()
            if movement_direction:
                print({"direction": movement_direction, "time": time.time()})

            key = cv2.waitKey(1)
            if key == ord("q"):
                exit_ = True
                cam.stop()
                sys.exit(0)
