import numpy as np
import cv2


def adjust_gamma(image, gamma=1.0):
    image = image.astype(np.uint8)
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    return cv2.LUT(image, table)


def apply_mask(image, bg, mask, fg_gamma=1):
    mask = np.expand_dims(mask, axis=2)

    hand = image * mask
    hand = adjust_gamma(hand, fg_gamma)

    return hand + bg * (1 - mask)
