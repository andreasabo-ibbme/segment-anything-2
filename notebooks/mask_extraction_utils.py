import numpy as np
import pandas as pd
import os, glob, shutil
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
    borders=False,
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


def plot_single_img_mask(
    image,
    mask,
    score,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
    image_name=None,
):

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca(), borders=borders)
    plt.axis("off")
    # plt.show()
    plt.savefig(f"image-{score}.png")

    if image_name is None:
        image_name = f"image-{mask}.png"
    plt.savefig(image_name)


def show_masks_combined(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
    image_name=None,
):
    # plt.figure(figsize=(10, 30))
    fig, axs = plt.subplots(len(scores), figsize=(15, 30))
    for i, (mask, score) in enumerate(zip(masks, scores)):
        axs[i].imshow(image)
        show_mask(mask, axs[i], borders=borders)
        axs[i].axis("off")
        axs[i].set_title(
            f"Mask {i}, Score: {score:.3f}, Num Pxls: {np.sum(mask):.0f}", fontsize=30
        )

    if image_name is None:
        image_name = f"image-{score}.png"
    plt.savefig(image_name)
    plt.show()


def process_folder(folder, label_csv, predictor, output_folder):
    labels_df = read_labels(label_csv)

    # Iterate through all images
    all_images = glob.glob(os.path.join(folder, "*.png"))
    mask_pixels = None

    for full_image_path in all_images:
        image_name = os.path.split(full_image_path)[-1]
        image_name_base = os.path.splitext(image_name)[0]

        if os.path.exists(os.path.join(output_folder, image_name)):
            continue

        points = read_points(labels_df, image_name)

        image = Image.open(full_image_path)
        image = np.array(image.convert("RGB"))

        predictor.set_image(image)
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

        output_image = os.path.join(
            output_folder, "mask_images", f"{image_name_base}-all_masks.png"
        )
        os.makedirs(os.path.dirname(output_image), exist_ok=True)

        # if no previous mask, make the figures and manually select which one to use
        delta = image.size

        if not mask_pixels:
            show_masks_combined(
                image,
                masks,
                scores,
                point_coords=input_point,
                input_labels=input_label,
                borders=True,
                image_name=output_image,
            )

            mask_id = int(input("Which mask to select? 0-indexed\n"))
            plt.close()

        else:
            for i, mask in enumerate(masks):
                if delta > abs(mask_pixels - mask.sum()):
                    mask_id = i
                    delta = abs(mask_pixels - mask.sum())

        # Run through one more iteration
        mask_input = logits[mask_id, :, :]  # Choose the model's best mask
        mask, score, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )

        if not mask_pixels:
            mask_pixels = mask.sum()


        # Save the mask for future use
        output_npy = os.path.join(output_folder, "mask_data", f"{image_name_base}")
        output_image_name = os.path.join(
            output_folder, "mask_images", f"{image_name_base}-mask_{mask_id}.png"
        )

        os.makedirs(os.path.dirname(output_npy), exist_ok=True)
        os.makedirs(os.path.dirname(output_image_name), exist_ok=True)

        np.save(output_npy, mask)
        plot_single_img_mask(
            image,
            mask,
            score,
            borders=False,
            image_name=output_image_name,
        )

        # Copy the original images to the output folder to use with the new masks
        shutil.copy2(full_image_path, output_folder)


def process_all_folders(folders, output_folder, folder_prefix):

    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    predictor = SAM2ImagePredictor(sam2_model)

    for folder in folders:
        label_csv = glob.glob(os.path.join(folder, "*scorer.csv"))[0]
        outer_folder = os.path.split(folder)[-1]
        cur_output_folder = os.path.join(output_folder, outer_folder)
        cur_output_folder = cur_output_folder.replace(folder_prefix, "")

        os.makedirs(cur_output_folder, exist_ok=True)
        process_folder(folder, label_csv, predictor, cur_output_folder)
