import cv2
import os, glob

from icecream import ic

BACKGROUND_IN = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants_SAM2/background_thumb"
BACKGROUND_OUT = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants_SAM2/background_thumb_edited"
# IMAGES_TO_EDIT = ["IMG_6351.jpg", "IMG_6352.jpg", "IMG_6354.jpg", "IMG_6358.jpg", "IMG_6356.jpg"]
IMAGES_TO_EDIT = ["IMG_6350.jpg"] #, "IMG_6352.jpg", "IMG_6354.jpg", "IMG_6358.jpg", "IMG_6356.jpg"]

IMAGES_TO_EDIT = None


OUTPUT_SIZE = (1920, 1080)
scale_factor = 1.275
def center_bottom_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[height-crop_height:height, mid_x-cw2:mid_x+cw2]
	return crop_img


if __name__ == "__main__":
	os.makedirs(BACKGROUND_OUT, exist_ok=True)
	
	if IMAGES_TO_EDIT is None:
		full_image_names = glob.glob(os.path.join(BACKGROUND_IN, "*"))
		IMAGES_TO_EDIT = [os.path.basename(f) for f in full_image_names]
		# ic(IMAGES_TO_EDIT)
	# quit()
 
	for image_name in IMAGES_TO_EDIT:
		full_file = os.path.join(BACKGROUND_IN, image_name)
		image = cv2.imread(full_file)
		
		# Resize by the scale factor, then crop the middle
		image = cv2.resize(image, (int(OUTPUT_SIZE[0]*scale_factor), int(OUTPUT_SIZE[1] *scale_factor)))
		
		ccrop_img = center_bottom_crop(image, OUTPUT_SIZE)
		output_name = os.path.join(BACKGROUND_OUT, image_name)
		cv2.imwrite(output_name, ccrop_img)

	pass