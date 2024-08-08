import os, glob
import random
import numpy as np
import cv2

from icecream import ic
import generate_image_utils

INPUT_FOLDER = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants_SAM2/EDS001_EDS100__FifthFinger-liam-2023-07-04_SAM2_auto/labeled-data/"
OUTPUT_FOLDER = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants_SAM2/EDS001_EDS100__FifthFinger-liam-2023-07-04_SAM2_auto/labeled-data-new-bg/"
BACKGROUND_IMAGES = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants_SAM2/backgrounds_edited"

NUM_IMAGES_TO_GENERATE = 5
GAMMA_RANGE = [0.4, 1.25]

BLUR_KERNEL_SIZE = 5
ERODE_KERNEL_SIZE = 5

random.seed(27)

background_images = glob.glob(os.path.join(BACKGROUND_IMAGES, "*"))

def add_new_background(image, mask, background_name, output_folder):
    bg = cv2.imread(background_name)
    bg = cv2.resize(bg, (image.shape[1], image.shape[0]))

    bg_image_base = os.path.splitext(os.path.split(background_name)[-1])[0]
    gamma = random.uniform(*GAMMA_RANGE)
    combined_image = generate_image_utils.apply_mask(image, bg, mask, gamma)
    output_image_name = os.path.join(output_folder, f"{bg_image_base}_{gamma:03f}.jpg")
    cv2.imwrite(output_image_name, combined_image)


if __name__ == "__main__":
    all_folders = glob.glob(os.path.join(INPUT_FOLDER, "*"))
    
    ic(len(all_folders))
    for folder_i, folder in enumerate(all_folders):
        ic(folder_i)
        images = glob.glob(os.path.join(folder, "*.png"))
        video_folder = os.path.split(folder)[-1]
        for image_path in images:
            image_name = os.path.split(image_path)[-1]
            image_name_base = os.path.splitext(image_name)[0]
        
            output_folder_cur = os.path.join(OUTPUT_FOLDER, video_folder, image_name_base)
            os.makedirs(output_folder_cur, exist_ok=True)
            
            
            mask_path = os.path.join(folder, "mask_data", f"{image_name_base}.npy")
            # quit()
            # The original image and mask don't change for each image, load once only
            raw_image = cv2.imread(image_path)
            raw_mask = np.load(mask_path).squeeze()
            
            erode_kernel = np.ones((ERODE_KERNEL_SIZE,ERODE_KERNEL_SIZE),np.uint8)
            mask_eroded = cv2.erode(raw_mask,erode_kernel,iterations = 1)
            
            blur_kernel = np.ones((BLUR_KERNEL_SIZE,BLUR_KERNEL_SIZE),np.float32)/(BLUR_KERNEL_SIZE**2)
            mask_eroded_blurred = cv2.filter2D(mask_eroded,-1,blur_kernel)
            
            # Use random.sample for selection without replacement
            random_backgrounds = random.sample(background_images, NUM_IMAGES_TO_GENERATE)
            for bg in random_backgrounds:
                add_new_background(raw_image, mask_eroded_blurred, bg, output_folder_cur)
            