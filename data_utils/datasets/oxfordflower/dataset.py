from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import os.path
import sys
import pickle

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_url ,download_and_extract_archive,verify_str_arg
from torchvision.datasets.vision import VisionDataset

from data_utils.transforms import get_transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../../../'))

from data_utils.transforms import get_transforms
from utils import logging
logger = logging.get_logger("lmeraser")

class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        # root: Union[str, Path],
        args,
        split: str = "train",
        percentage: float = 0.8,
        sub_percentage: float = 1.0,  # use part of the training set
        # transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        # download: bool = False,
        download: bool = True,
    ) -> None:
        super().__init__(
            args.data_dir, 
            transform=get_transforms(split, args.crop_size, args.pretrained_model), 
            target_transform=target_transform
        )
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(args.data_dir) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(f"Value of download : bool = {download} & check_integrity = {self._check_integrity()} \n Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='Meta Training for Visual Prompts')
    parser.add_argument('--dataset', type=str, default="oxfordflower")
    # parser.add_argument('--data_dir', type=str, default="/data-x/g12/huangqidong/")
    parser.add_argument('--data_dir', type=str, default="Priyansh-Rishav/lmeraser/Dataset_Download/dataset_dir/g12/huangqidong/")
    parser.add_argument('--pretrained_model', type=str, default="vit-b-22k")
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=102)
    args = parser.parse_args()



    dataset_train = Flower102(args, "train")
    dataset_val = Flower102(args, "val")
    dataset_test = Flower102(args, "test")
    logger.info(f"Nums of classes: {len(dataset_train.classes)}")
    logger.info(
        f"Sample nums: [train]-{len(dataset_train)}, [val]-{len(dataset_val)}, [test]-{len(dataset_test)}"
    )

    for sample in DataLoader(dataset_train, batch_size=32):
        logger.info(sample["image"].shape)
        logger.info(sample["label"].shape)
        break