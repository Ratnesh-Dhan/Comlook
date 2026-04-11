
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from supports.TextWork import PanelWork
from supports.Ml import Ml
import cv2

if __name__ == "__main__":
    directory_path = "/home/zumbie/Downloads/HENTAI/testfuck"

    panelWork = PanelWork(directory_path)
    ml = Ml()
    

    save_images_path = os.path.join("../", "MangaTranslatedImages")
    os.makedirs(save_images_path, exist_ok=True)
    # images = os.listdir(directory_path)
    # for i in images:
    #     os.rename(os.path.join(directory_path, i), os.path.join(directory_path, i.replace("pg", "")))
    images = os.listdir(directory_path)
    images=sorted(images)
    translated_images = []

    for image_path, i in tqdm(zip(images, range(len(images)))):
        if image_path.endswith(".pdf" or ".txt"):
            continue

        img_rgb = panelWork.preprocess(image_path)
        panel_boxes = ml.bboxes(img_rgb)
        # print(panel_boxes)
