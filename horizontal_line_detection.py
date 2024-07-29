"""Detect horizontal lines automagically."""

import json
import math

import cv2
import numpy as np


def show(img):
    cv2.imshow("image", img)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


def cannyPipeline(path: str) -> bool:
    img = cv2.imread(path)
    img = cv2.Canny(img, 80, 120)
    lines = cv2.HoughLinesP(img, 1, math.pi / 2, 2, None, 30, 1)
    return lines is None


def thresholdPipeline(path: str) -> bool:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ret, img = cv2.threshold(
        img, np.quantile(img, 0.95), 255, cv2.THRESH_BINARY
    )  # type: ignore
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY, 1001, 0.1)
    lines = cv2.HoughLinesP(
        img,
        rho=1,  # resolution of the parameter rho (1 pixel)
        theta=math.pi / 2,  # resolution of the theta parameter (90 degree)
        threshold=200,
        lines=None,
        minLineLength=150,
        maxLineGap=5,
    )
    return lines is None


with open("0.avi_is_component_selected_122_manual.json", "r") as f:
    is_component_selected_manual = json.load(f)


is_component_selected: list[bool] = []
for i in range(122):
    path = f"intermediate_frames/component_map_{i}.png"
    is_component_selected.append(thresholdPipeline(path))

difference = [i for i in range(122) if is_component_selected[i] != is_component_selected_manual[i]]

for i in difference:
    print(f"Element {i}: {is_component_selected_manual[i]} in manual, {is_component_selected[i]} in auto.")
    
