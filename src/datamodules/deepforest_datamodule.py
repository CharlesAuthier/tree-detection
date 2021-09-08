import math
from typing import Optional, Tuple, Dict
import os

from PIL import Image
import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from torchvision.transforms import functional as F

from deepforest import get_data
from deepforest import utilities
from deepforest import preprocess


class DeepForestDataModule(LightningDataModule):
    """
    Example of LightningDataModule for NRCan tree and hedge detection dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        classes_dict: dict = {'tree': 1},
        img_size: int = 256,
        img_overlap: int = 0.0,
        train_val_test_split: Tuple[int, int, int] = (75, 5, 20),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.num_classes_from_dict = len(classes_dict.keys())
        self.img_size = img_size
        self.img_overlap = img_overlap
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 28, 28)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.num_classes_from_dict

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        get_data("2019_YELL_2_528000_4978000_image_crop2.xml")
        get_data("2019_YELL_2_528000_4978000_image_crop2.png")
        # TODO transfer data maybe

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        # convert hand annotations from xml into retinanet format
        # The get_data function is only needed when fetching sample package data
        YELL_xml = get_data("2019_YELL_2_528000_4978000_image_crop2.xml")
        annotation = utilities.xml_to_annotations(YELL_xml)
        # load the image file corresponding to the annotaion file
        YELL_train = get_data("2019_YELL_2_528000_4978000_image_crop2.png")
        image_path = os.path.dirname(YELL_train)
        # Write converted dataframe to file. Saved alongside the images
        annotation.to_csv(os.path.join(image_path, "train_example.csv"), index=False)
        # Find annotation path
        annotation_path = os.path.join(image_path, "train_example.csv")
        # crop images will save in a newly created directory
        # os.mkdir(os.getcwd(),'train_data_folder')
        crop_dir = os.path.join(self.data_dir, 'DeepForest')
        train_annotations = preprocess.split_raster(
            path_to_raster=YELL_train,
            annotations_file=annotation_path,
            base_dir=crop_dir,
            patch_size=self.img_size,
            patch_overlap=self.img_overlap
        )

        # trainset = MNIST(self.data_dir, train=True, transform=self.transforms)
        # testset = MNIST(self.data_dir, train=False, transform=self.transforms)
        dfdataset = DeepForestDataset(crop_dir, train_annotations)  # TODO add a transform like mnist

        # ajust the split number base on poucentage TODO need to change for something better
        tvt_split = [(element / 100) * len(dfdataset) for element in self.train_val_test_split]
        tvt_split = [math.ceil(element) for element in tvt_split]
        tvt_split[-1] = len(dfdataset) - (tvt_split[0] + tvt_split[1])

        # random split
        self.data_train, self.data_val, self.data_test = random_split(dfdataset, tvt_split)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        images, targets = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["boxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }
        return images, annotations


def get_target_ds(name, df):
    rows = df[df["image_path"] == name]
    label = rows["label"].replace('Tree', 1)
    return label.values, rows[['xmin', 'ymin', 'xmax', 'ymax']].values


class DeepForestDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, df):
        super(DeepForestDataset, self).__init__()
        self.images_path = images_path
        self.df = df

    def __len__(self):
        return len(self.df.image_path.unique())

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_path, self.df.image_path.unique()[idx])
        target = {}
        with Image.open(str(img_path)) as img:
            img = F.to_tensor(img)
        _, new_h, new_w = img.shape
        labels, boxes = get_target_ds(self.df.image_path.unique()[idx], self.df)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],))
        image_id = torch.tensor([idx])
        labels = torch.as_tensor(labels, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        target["boxes"] = boxes
        target["labels"] = labels
        # target["areas"] = areas
        target["img_scale"] = torch.tensor([1.0])
        target["img_size"] = (new_h, new_w)
        # target["iscrowd"] = iscrowd
        # target["image_id"] = image_id
        return img, target
