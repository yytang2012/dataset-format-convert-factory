from math import ceil

import cv2
import numpy as np

from visualize.draw_settings import GREEN


def apply_box(image, bbox, color=GREEN, thickness=2, ratio=1):
    thickness *= ceil(ratio)
    y1, x1, y2, x2 = [int(i) for i in bbox]

    # rgb_color = tuple([int(c * 255) for c in color])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def apply_polygon(image, pts_list, color=GREEN, thickness=2):
    pts = np.array(pts_list, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], True, color, thickness)
    return image


def apply_line(image, start_point, end_point, color, thickness=1, ratio=1):
    thickness *= ceil(ratio)
    cv2.line(image, start_point, end_point, color=color, thickness=thickness)


def apply_text_left(image, bbox, color, text, font_scale=0.5, thickness=1, ratio=1):
    rgb_color = tuple([int(c * 255) for c in color])
    font_scale *= ratio
    thickness *= ceil(ratio)
    if len(bbox) == 4:
        y1, x1, y2, x2 = [int(i) for i in bbox]

        cv2.putText(image, '{}'.format(text), (x1 + 5, (y1 + y2) // 2),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, rgb_color, thickness)
    if len(bbox) == 2:
        x, y = bbox
        cv2.putText(image, '{}'.format(text), (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
    return image


def apply_text_upper_left(image, bbox, color, text, font_scale=1, thickness=1, ratio=1):
    font_scale *= ratio
    thickness *= ceil(ratio)
    # text_wind_size = lambda tex: cv2.getTextSize(tex, cv2.FONT_HERSHEY_DUPLEX, fontScale, thickness)[0]
    text_length, text_height = text_size(text)
    ys, xs, _, _ = bbox
    ys = ys - text_height - 5
    ye, xe = ys + text_height * ceil(ratio), xs + text_length * ceil(ratio)
    cv2.rectangle(image, (xs, ys), (xe, ye + 5), color=color, thickness=-1)
    cv2.putText(image, text, (xs, ye), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness)
    return image


def apply_text_center(image, y, x, color, text, font_scale=0.5, thickness=1, ratio=1):
    font_scale *= ratio
    thickness *= ceil(ratio)
    w, h = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, thickness=thickness)[0]
    ys = y
    _, width_img = image.shape[:2]
    xs = width_img // 2 - w // 2
    cv2.putText(image, "{}".format(text), (xs, ys), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
    return image


def apply_contour(image, mask, bbox, color, thickness=3, ratio=1):
    thickness *= ceil(ratio)
    # new_mask = np.where(mask, 255, 0).astype(np.uint8)
    y1, x1, y2, x2 = bbox
    new_mask = -mask[y1:y2, x1:x2].astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(new_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    cv2.drawContours(image[y1:y2, x1:x2], contours, -1, color, thickness)


def text_size(text, font_scale=0.5, thickness=1, ratio=1):
    font_scale *= ratio
    thickness *= ceil(ratio)
    width, height = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, thickness=thickness)[0]
    return width, height
