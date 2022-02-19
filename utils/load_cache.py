import json
import os

import torch
import tqdm
import numpy as np

from configs import Config
from loguru import logger

import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset


class LoadCache(Dataset):
    def __init__(self, cache_path: str, path: str, word: bool, image_channel: int, resize: list, charset: list):
        self.cache_path = cache_path
        self.path = path
        self.word = word
        self.ImageChannel = image_channel
        self.resize = resize
        self.charset = charset

        logger.info("\nReading Cache File... ----> {}".format(self.cache_path))

        with open(self.cache_path, 'r', encoding='utf-8') as f:
            caches = f.readlines()
        self.caches = []
        for cache in tqdm.tqdm(caches):
            cache = cache.replace("\r", "").replace("\n", "").split("\t")
            self.caches.append(cache)
        del caches

        self.caches_num = len(self.caches)
        logger.info("\nRead Cache File End! Caches Num is {}.".format(self.caches_num))

    def __len__(self):
        return self.caches_num

    def __getitem__(self, idx):
        try:
            data = self.caches[idx]
            image_name = data[0]
            image_label = data[1]
            image_path = os.path.join(self.path, image_name)
            if not self.word:
                image_label = list(image_label)
            else:
                image_label = [image_label]
            if self.ImageChannel == 1:
                mode = torchvision.io.ImageReadMode.GRAY
            else:
                mode = torchvision.io.ImageReadMode.RGB
            image = torchvision.io.read_image(image_path, mode=mode)  # shape c, h, w
            image_shape = image.shape
            image_height = image_shape[1]
            image_width = image_shape[2]
            width = self.resize[0]
            height = self.resize[1]
            if self.resize[0] == -1:
                image = torchvision.transforms.Resize((height, int(image_width * (height / image_height))))(image)
            else:
                image = torchvision.transforms.Resize((height, width))(image)
            image = torchvision.transforms.ToPILImage()(image)
            label = [int(self.charset.index(item)) for item in list(image_label)]
            return image, label

        except Exception as e:
            logger.error("\nError: {}, File: {}".format(str(e), self.caches[idx][0]))
            return None, None


class GetLoader:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects",
                                         project_name)
        if os.path.exists(self.project_path):
            self.cache_path = os.path.join(self.project_path, "cache")
            if os.path.exists(self.cache_path):
                self.cache_train_path = os.path.join(self.cache_path, "cache.train.tmp")
                self.cache_val_path = os.path.join(self.cache_path, "cache.val.tmp")

                if not os.path.exists(self.cache_train_path):
                    logger.error("\nCache Train File {} is not exists!".format(self.cache_train_path))
                    exit()
                if not os.path.exists(self.cache_val_path):
                    logger.error("\nCache Val File {} is not exists!".format(self.cache_val_path))
                    exit()

            else:
                logger.error("\nCache dir {} is not exists!".format(self.cache_path))
                exit()
        else:
            logger.error("\nProject {} is not exists!".format(project_name))
            exit()

        self.config = Config(project_name)
        self.conf = self.config.load_config()

        self.charset = self.conf['Model']['CharSet']
        logger.info("\nCharsets is {}".format(json.dumps(self.charset, ensure_ascii=False)))

        self.resize = [int(self.conf['Model']['ImageWidth']), int(self.conf['Model']['ImageHeight'])]
        logger.info("\nImage Resize is {}".format(json.dumps(self.resize)))

        self.ImageChannel = self.conf['Model']['ImageChannel']

        self.word = self.conf['Model']['Word']

        self.path = self.conf['System']['Path']

        self.batch_size = self.conf['Train']['BATCH_SIZE']

        self.val_batch_size = self.conf['Train']['TEST_BATCH_SIZE']

        logger.info("\nImage Path is {}".format(self.path))

        self.transform_list = []
        self.transform_list.append(torchvision.transforms.ToTensor())
        if self.ImageChannel == 1:
            self.transform_list.append(torchvision.transforms.Normalize(mean=[0.456],
                                                                   std=[0.224]))
        else:
            if self.ImageChannel != 3:
                logger.error("ImageChannel must be 1 or 3!")
                exit()
            self.transform_list.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]))
        self.transform = torchvision.transforms.Compose(self.transform_list)
        tarin_loader = LoadCache(self.cache_train_path, self.path, self.word, self.ImageChannel, self.resize, self.charset)
        val_loader = LoadCache(self.cache_val_path, self.path, self.word, self.ImageChannel, self.resize, self.charset)
        self.loaders = {
            'train': DataLoader(dataset=tarin_loader, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                num_workers=0, collate_fn=self.collate_to_sparse),
            'val': DataLoader(dataset=val_loader, batch_size=self.val_batch_size, shuffle=True, drop_last=True,
                                num_workers=0, collate_fn=self.collate_to_sparse),
        }

    def collate_to_sparse(self, batch):
        values = []
        images = []
        shapes = []
        max_width = 0
        for n, (img, seq) in enumerate(batch):
            if img is None or seq is None:
                continue
            if len(seq) == 0: continue
            if max_width < img.size[0]:
                max_width = img.size[0]
            values.extend(seq)
            images.append(img)
            shapes.append(len(seq))
        images_pad = []
        for img in images:
            img = torchvision.transforms.Pad((0, 0, int(max_width - img.size[0]), 0))(img)
            if self.transform is not None:
                img = self.transform(img)
            images_pad.append(img)
        images_pad = torch.stack(images_pad, dim=0)
        return [images_pad, torch.FloatTensor(values), torch.IntTensor(shapes)]

