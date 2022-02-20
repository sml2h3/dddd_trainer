import json
import os
import random

import tqdm

from configs import Config
from loguru import logger


class CacheData:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects",
                                         project_name)
        if os.path.exists(self.project_path):
            self.cache_path = os.path.join(self.project_path, "cache")
        else:
            logger.error("Project {} is not exists!".format(project_name))
            exit()
        self.config = Config(project_name)
        self.conf = self.config.load_config()
        self.bath_path = self.conf['System']['Path']
        self.allow_ext = []

    def cache(self, base_path: str, search_type="name"):
        self.bath_path = base_path
        self.allow_ext = self.conf["System"]["Allow_Ext"]
        if search_type == "name":
            self.__get_label_from_name(base_path=base_path)
        else:
            self.__get_label_from_file(base_path=base_path)

    def __get_label_from_name(self, base_path: str):
        files = os.listdir(base_path)
        logger.info("\nFiles number is {}.".format(len(files)))
        self.__collect_data(files, base_path)

    def __get_label_from_file(self, base_path: str):
        labels_path = os.path.join(base_path, "labels.txt")
        images_path = os.path.join(base_path, "images")
        if not os.path.exists(labels_path):
            logger.error("\nThe file labels.txt not found in path ----> {}".format(base_path))
            exit()
        if not os.path.exists(images_path) or not os.path.isdir(images_path):
            logger.error("\nThe dir {} not found in path ----> {}".format(images_path, base_path))
            exit()
        files = os.listdir(images_path)
        logger.info("\nFiles number is {}.".format(len(files)))
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_lines = f.readlines()
        labels_lines = [line.replace("\r", "").replace("\n", "") for line in labels_lines]
        labels_filename_lines = [line.split("\t")[0] for line in labels_lines]
        logger.info("\nLabels number is {}.".format(len(labels_lines)))
        logger.info("\nChecking labels.txt ...")
        error_files = set(labels_filename_lines).difference(set(files))
        logger.info("\nCheck labels.txt end! {} errors!".format(len(error_files)))
        del files
        self.__collect_data(labels_lines, images_path, error_files, is_file=True)

    def __collect_data(self, lines, base_path, error_files, is_file=False):
        labels = []
        caches = []

        for file in tqdm.tqdm(lines):
            if is_file:
                line_list = file.split('\t')
                filename = line_list[0]
                label = line_list[1]
            else:
                filename = file
                label = "_".join(filename.split("_")[:-1])
            if filename in error_files:
                continue
            label = label.replace(" ", "")
            if filename.split('.')[-1] in self.allow_ext:
                if " " in filename:
                    logger.warning("The {} has black. We will remove it!".format(filename))
                    continue
                caches.append('\t'.join([filename, label]))
                if not self.conf['Model']['Word']:
                    label = list(label)
                    labels.extend(label)
                else:
                    labels.append(label)

            else:
                logger.warning("\nFile({}) has a suffix that is not allowed! We will remove it!".format(file))
        labels = list(set(labels))
        if not self.conf['Model']['Word']:
            labels.insert(0, " ")
        logger.info("\nCoolect labels is {}".format(json.dumps(labels, ensure_ascii=False)))
        self.conf['System']['Path'] = base_path
        self.conf['Model']['CharSet'] = labels
        self.config.make_config(config_dict=self.conf, single=self.conf['Model']['Word'])
        logger.info("\nWriting Cache Data!")
        del lines
        logger.info("\nCache Data Number is {}".format(len(caches)))
        logger.info("\nWriting Train and Val File.".format(len(caches)))
        val = self.conf['System']['Val']
        if 0 < val < 1:
            val_num = int(len(caches) * val)
        elif 1 < val < len(caches):
            val_num = int(val)
        else:
            logger.error("val setting vaild!")
            exit()
        random.shuffle(caches)
        train_set = caches[val_num:]
        val_set = caches[:val_num]
        del caches
        with open(os.path.join(self.cache_path, "cache.train.tmp"), 'w', encoding="utf-8") as f:
            f.write("\n".join(train_set))
        with open(os.path.join(self.cache_path, "cache.val.tmp"), 'w', encoding="utf-8") as f:
            f.write("\n".join(val_set))
        logger.info("\nTrain Data Number is {}".format(len(train_set)))
        logger.info("\nVal Data Number is {}".format(len(val_set)))
