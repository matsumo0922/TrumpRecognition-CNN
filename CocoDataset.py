import os.path
import torch.utils.data.dataset
import torchvision.datasets.coco
import nltk

from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(torchvision.datasets.coco.CocoDetection):
    def __getitem__(self, index):
        ann_id = self.ids[index]
        cat_id = self.coco.anns[ann_id]["category_id"]
        img_id = self.coco.anns[ann_id]["image_id"]
        path = self.coco.loadImgs(img_id)[0]["file_name"]

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(cat_id, dtype=torch.int8)

    def get_label(self, cat_id):
        return self.coco.cats[cat_id]["name"]

