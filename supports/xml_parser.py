import torch
import torchvision
from torchvision.transforms import functional as F
import os
import cv2
import xml.etree.ElementTree as ET

class MangaVOCDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        
        self.image_files = sorted(os.listdir(images_dir))
        self.valid_images = []

        for img_name in self.image_files:
            xml_name = img_name.replace(".jpg", ".xml")
            xml_path = os.path.join(annotations_dir, xml_name)

            if not os.path.exists(xml_path):
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()

            objects = root.findall("object")

            # Keep only images that have at least 1 object
            if len(objects) > 0:
                self.valid_images.append(img_name)
        
        self.image_files = self.valid_images
        print(f"Found {len(self.image_files)} valid images")
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(img)

        # Load XML
        xml_name = img_name.replace(".jpg", ".xml")
        xml_path = os.path.join(self.annotations_dir, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text

            # Single class → japanese text
            if name == "japanese" or name == "english":
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # class id 1


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target