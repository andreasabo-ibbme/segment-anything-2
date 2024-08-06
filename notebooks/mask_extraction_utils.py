import numpy as np
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from icecream import ic

def read_labels(input_file):
    df = pd.read_csv(input_file)
    df.rename(
        columns={"Unnamed: 2": "image_name", "combined_scorer": "combined_scorer.0"},
        inplace=True,
    )
    return df

def read_points(df, image_name):
    row = df[df["image_name"] == image_name]

    cols = [col for col in df.columns.to_list() if "combined_scorer" in col]
    points = []
    for col in cols:

        col_num = int(col.split(".")[-1])
        if col_num % 2 == 1:
            continue

        # Take this point and the one after it
        points.append(
            
                [
                    float(row[f"combined_scorer.{col_num}"].values[0]),
                    float(row[f"combined_scorer.{col_num+ 1}"].values[0]),
                ]
        )

    points = np.array(points)
    return points


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        # plt.show()
        plt.savefig(f"image-{score}.png")


def process_image():
    IMAGE_PATH = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants/EDS001_EDS100__FifthFinger-liam-2023-07-04_SAM2/labeled-data/andrea_new-fifth_finger_r__EDS093__fifth_finger_r/img141.png"
    LABELS_PATH = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants/EDS001_EDS100__FifthFinger-liam-2023-07-04_SAM2/labeled-data/andrea_new-fifth_finger_r__EDS093__fifth_finger_r/CollectedData_combined_scorer.csv"
    # IMAGE_PATH = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants/EDS001_EDS100__FifthFinger-liam-2023-07-04_SAM2/labeled-data/andrea_new-fifth_finger_l__EDS047__fifth_finger_l/img100.png"
    # LABELS_PATH = r"/home/saboa/mnt/n_drive/AMBIENT/Andrea_S/EDS/DLC_working_dir/dlc_projects_participants/EDS001_EDS100__FifthFinger-liam-2023-07-04_SAM2/labeled-data/andrea_new-fifth_finger_l__EDS047__fifth_finger_l/CollectedData_combined_scorer.csv"
    # IMAGE_PATH = "images/truck.jpg"
    image_name = os.path.split(IMAGE_PATH)[-1]
    ic(image_name)
    points = read_points(LABELS_PATH, image_name)
    ic(points)

    # quit()

    image = Image.open(IMAGE_PATH)
    image = np.array(image.convert("RGB"))

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("on")
    # plt.show()
    plt.savefig("image.png")

    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)

    input_point = np.array([[500, 375]])
    input_point = np.array(points)
    input_label = np.array([1] * len(points))

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis("on")
    plt.savefig("image_with_point.png")

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    ic(masks[2])

    show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
    

def process_folder(folder, label_csv, predictor):
    labels_df = read_labels(label_csv)
    
    # Iterate through all images
    all_images = glob.glob(os.path.join(folder, "*.png"))
    
    for full_image_path in all_images:
        image_name = os.path.split(full_image_path)[-1]
        points = read_points(labels_df, image_name)
        
        image = Image.open(full_image_path)
        image = np.array(image.convert("RGB"))
        
        predictor.set_image(image)
        ic(points)
        input_point = np.array(points)
        input_label = np.array([1] * len(points))

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)


    
def process_all_folders(folders, output_folder, folder_prefix):
    os.makedirs(output_folder, exist_ok=True)

    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    predictor = SAM2ImagePredictor(sam2_model)
    
    for folder in folders:
        label_csv = glob.glob(os.path.join(folder, "*scorer.csv"))[0]
        process_folder(folder, label_csv, predictor)
        quit()
        # ic(label_csv)
        
        
        