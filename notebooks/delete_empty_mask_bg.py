import torch
import glob, os

from icecream import ic
import clean_up_mask_utils


INPUT_IMAGE_FOLDER = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants/EDS001_EDS100__FifthFinger-liam-2023-07-04_SAM2/labeled-data/"
OUTPUT_FOLDER = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants_SAM2/EDS001_EDS100__FifthFinger-liam-2023-07-04_SAM2_auto/labeled-data-v2/"
FOLDER_PREFIX = "andrea_new-"
BG_FOLDER = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants_SAM2/EDS001_EDS100__FifthFinger-liam-2023-07-04_SAM2_auto/labeled-data-new-bg/"


def prerun_settings():
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def list_all_folders(INPUT_IMAGE_FOLDER):
    all_folders = glob.glob(os.path.join(INPUT_IMAGE_FOLDER, f"{FOLDER_PREFIX}*"))
    return all_folders
    

if __name__ == "__main__":
    prerun_settings()
    folders_to_process = list_all_folders(INPUT_IMAGE_FOLDER)
    clean_up_mask_utils.clean_up_all_folders(folders_to_process, BG_FOLDER, FOLDER_PREFIX)
