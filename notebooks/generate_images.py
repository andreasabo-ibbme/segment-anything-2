import glob, os
from icecream import ic
import numpy as np
import cv2


IMAGE_PATH = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants_SAM2/EDS001_EDS100__FifthFinger-liam-2023-07-04_SAM2_auto/labeled-data/fifth_finger_r__EDS098__fifth_finger_r/img112.png"
MASK_PATH = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants_SAM2/EDS001_EDS100__FifthFinger-liam-2023-07-04_SAM2_auto/labeled-data/fifth_finger_r__EDS098__fifth_finger_r/mask_data/img112.npy"
BACKGOUND_IMG = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants_SAM2/backgrounds/IMG_2110.jpg"
BLUR_KERNEL_SIZE = 7
ERODE_KERNEL_SIZE = 3

def apply_mask(image, bg, mask):
    mask = np.expand_dims(mask, axis=2)
    return image * mask + bg * (1 - mask)


def change_background(img_path, mask_path, background):
    image = cv2.imread(img_path)
    bg = cv2.imread(background)
    bg = cv2.resize(bg, (image.shape[1], image.shape[0]))
    
    
    mask = np.load(mask_path)
    mask_orig = mask.squeeze()
    cv2.imwrite("mask.png", mask_orig*255)
    
    erode_kernel = np.ones((ERODE_KERNEL_SIZE,ERODE_KERNEL_SIZE),np.uint8)
    mask_eroded = cv2.erode(mask_orig,erode_kernel,iterations = 1)
    cv2.imwrite("mask_eroded.png", mask*255)

    blur_kernel = np.ones((BLUR_KERNEL_SIZE,BLUR_KERNEL_SIZE),np.float32)/(BLUR_KERNEL_SIZE**2)
    mask_eroded_blurred = cv2.filter2D(mask_eroded,-1,blur_kernel)
    
    cv2.imwrite("mask_blurred.png", mask_eroded_blurred*255)

    ic(mask_orig.shape, mask_eroded.shape, mask_eroded_blurred.shape, image.shape, bg.shape)
    mask_orig_out = apply_mask(image, bg, mask_orig)
    mask_eroded_out = apply_mask(image, bg, mask_eroded)
    mask_eroded_blurred_out = apply_mask(image, bg, mask_eroded_blurred)

    cv2.imwrite("mask_orig_out.png", mask_orig_out)
    cv2.imwrite("mask_eroded_out.png", mask_eroded_out)
    cv2.imwrite("mask_eroded_blurred_out.png", mask_eroded_blurred_out)
    



if __name__ == "__main__":
    change_background(IMAGE_PATH, MASK_PATH, BACKGOUND_IMG)