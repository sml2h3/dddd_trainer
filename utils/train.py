import json
import os
import random
import time

import tqdm

from configs import Config
from loguru import logger
from utils import load_cache
from nets import Net


class Train:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects",
                                         project_name)
        self.checkpoints_path = os.path.join(self.project_path, "checkpoints")
        self.models_path = os.path.join(self.project_path, "models")
        self.config = Config(project_name)
        self.conf = self.config.load_config()

        self.test_step = self.conf['Train']['TEST_STEP']
        self.target = self.conf['Train']['TARGET']
        self.target_acc = self.target['Accuracy']
        self.min_epoch = self.target['Epoch']
        self.max_loss = self.target['Cost']
        logger.info("\nTaget:\nmin_Accuracy: {}\nmin_Epoch: {}\nmax_Loss: {}".format(self.target_acc, self.min_epoch,
                                                                                     self.max_loss))

        logger.info("\nBuilding Net...")
        self.net = Net(self.conf)
        logger.info(self.net)
        logger.info("\nBuilding End")

        self.use_gpu = self.conf['System']['GPU']
        if self.use_gpu:
            self.gpu_id = self.conf['System']['GPU_ID']
            logger.info("\nUSE GPU ----> {}".format(self.gpu_id))
            self.device = self.net.get_device(self.gpu_id)
            self.net.to(self.device)
        else:
            self.gpu_id = -1
            self.device = self.net.get_device(self.gpu_id)
            logger.info("\nUSE CPU".format(self.gpu_id))
        logger.info("\nGet Data Loader...")
        loaders = load_cache.GetLoader(project_name)
        self.train = loaders.loaders['train']
        self.val = loaders.loaders['val']
        logger.info("\nGet Data Loader End!")

        self.epoch = 0
        self.step = 0
        self.loss = 0
        self.avg_loss = 0
        self.start_time = time.time()
        self.now_time = time.time()

    def start(self):
        val_iter = iter(self.val)
        while True:
            for idx, (inputs, labels, labels_length) in enumerate(self.train):
                self.now_time = time.time()
                inputs = self.net.variable_to_device(inputs, device=self.device)

                loss, lr = self.net.trainer(inputs, labels, labels_length)

                self.avg_loss += loss

                self.step += 1

                if self.step % 100 == 0 and self.step % self.test_step != 0:
                    logger.info("{}\tEpoch: {}\tStep: {}\tLastLoss: {}\tAvgLoss: {}\tLr: {}".format(
                        time.strftime("[%Y-%m-%d-%H_%M_%S]", time.localtime(self.now_time)), self.epoch, self.step,
                        str(loss), str(self.avg_loss / 100), lr
                    ))
                    self.avg_loss = 0
                if self.step % 2000 == 0 and self.step != 0:
                    model_path = os.path.join(self.checkpoints_path, "checkpoint_{}_{}_{}.tar".format(
                        self.project_name, self.epoch, self.step,
                    ))
                    self.net.scheduler.step()
                    self.net.save_model(model_path,
                                        {"net": self.net.state_dict(), "optimizer": self.net.optimizer.state_dict(),
                                         "epoch": self.epoch, "step": self.step})

                if self.step % self.test_step == 0:
                    try:
                        test_inputs, test_labels, test_labels_length = next(val_iter)
                    except Exception:
                        del val_iter
                        val_iter = iter(self.val)
                        test_inputs, test_labels, test_labels_length = next(val_iter)
                    if test_inputs.shape[0] < 5:
                        continue
                    test_inputs = self.net.variable_to_device(test_inputs, self.device)
                    self.net = self.net.train(False)
                    pred_labels, labels_list, correct_list, error_list = self.net.tester(test_inputs, test_labels,
                                                                                          test_labels_length)
                    self.net = self.net.train()
                    accuracy = len(correct_list) / test_inputs.shape[0]
                    logger.info("{}\tEpoch: {}\tStep: {}\tLastLoss: {}\tAvgLoss: {}\tLr: {}\tAcc: {}".format(
                        time.strftime("[%Y-%m-%d-%H_%M_%S]", time.localtime(self.now_time)), self.epoch, self.step,
                        str(loss), str(self.avg_loss / 100), lr, accuracy
                    ))
                    self.avg_loss = 0
                    if accuracy > self.target_acc and self.epoch > self.min_epoch and self.avg_loss < self.max_loss:
                        logger.info("\nTraining Finished!Exporting Model...")
                        dummy_input = self.net.get_random_tensor()
                        input_names = ["input1"]
                        output_names = ["output"]

                        if self.net.backbone.startswith("effnet"):
                            self.net.cnn.set_swish(memory_efficient=False)
                        self.net = self.net.eval().cpu()
                        dynamic_ax = {'input1': {3: 'image_wdith'}, "output": {1: 'seq'}}
                        self.net.export_onnx(self.net, dummy_input,
                                             os.path.join(self.models_path, "{}_{}_{}_{}_{}.onnx".format(
                                                 self.project_name, str(accuracy), self.epoch, self.step,
                                                 time.localtime(self.now_time)))
                                             , input_names, output_names, dynamic_ax)
                        logger.info("\nExport Finished!Using Time: {}min".format(str(int(int(self.now_time * 1000) - int(self.start_time * 1000)) / 60)))
                        exit()

            self.epoch += 1



if __name__ == '__main__':
    Train("test1")
