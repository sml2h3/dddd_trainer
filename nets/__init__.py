import json

from .backbone import *
import torch


class Net(torch.nn.Module):
    def __init__(self, conf):
        super(Net, self).__init__()
        self.backbones_list = {
            "ddddocr": DdddOcr,
            "effnetv2_l": effnetv2_l,
            "effnetv2_m": effnetv2_m,
            "effnetv2_xl": effnetv2_xl,
            "effnetv2_s": effnetv2_s,
            "mobilenetv2": mobilenetv2,
            "mobilenetv3_s": MobileNetV3_Small,
            "mobilenetv3_l": MobileNetV3_Large
        }

        self.optimizers_list = {
            "SGD": torch.optim.SGD,
            "Adam": torch.optim.Adam,
        }
        self.conf = conf
        self.image_channel = self.conf['Model']['ImageChannel']
        self.resize = [int(self.conf['Model']['ImageWidth']), int(self.conf['Model']['ImageHeight'])]
        self.charset = self.conf['Model']['CharSet']
        self.charset_len = len(self.charset)
        self.backbone = self.conf['Train']['CNN']['NAME']
        self.paramters = []

        if self.backbone in self.backbones_list:
            test_cnn = self.backbones_list[self.backbone](nc=1)
            x = torch.randn(1, 1, 64, 224)
            test_features = test_cnn(x)
            del x
            del test_cnn
            self.out_size = test_features.size()[1] * test_features.size()[2]
            self.cnn = self.backbones_list[self.backbone](nc=self.image_channel)
        else:
            raise Exception("{} is not found in backbones! backbone list : {}".format(self.backbone, json.dumps(
                list(self.backbones_list.keys()))))
        self.paramters.append({'params': self.cnn.parameters()})

        self.word = self.conf['Model']['Word']
        if not self.word:
            self.dropout = self.conf['Train']['DROPOUT']
            self.lstm = torch.nn.LSTM(input_size=self.out_size, hidden_size=self.out_size, bidirectional=True,
                                      dropout=self.dropout, num_layers=1)
            self.paramters.append({'params': self.lstm.parameters()})

            self.loss = torch.nn.CTCLoss(blank=0, reduction='mean')
        else:
            self.lstm = None
            self.loss = torch.nn.CrossEntropyLoss()

        self.paramters.append({'params': self.loss.parameters()})

        self.fc = torch.nn.Linear(in_features=self.out_size * 2, out_features=self.charset_len)
        self.paramters.append({'params': self.fc.parameters()})

        self.lr = self.conf['Train']['LR']

        self.optim = self.conf['Train']['OPTIMIZER']
        if self.optim in self.optimizers_list:
            if self.optim == "SGD":
                self.optimizer = self.optimizers_list[self.optim](self.paramters, lr=self.lr, momentum=0.9)
            else:
                self.optimizer = self.optimizers_list[self.optim](self.paramters, lr=self.lr, betas=(0.9, 0.99))
        else:
            raise Exception("{} is not found in optimizers! optimizers list : {}".format(self.optim, json.dumps(
                list(self.optimizers_list.keys()))))

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

    def forward(self, inputs):
        predict = self.get_features(inputs)
        if self.word:
            outputs = predict.max(1)
        else:
            outputs = predict.max(2)[1].transpose(0, 1)
        return outputs

    def get_features(self, inputs):
        outputs = self.cnn(inputs)
        if not self.word:
            outputs = outputs.permute(3, 0, 1, 2)
            w, b, c, h = outputs.shape
            outputs = outputs.view(w, b, c * h)
            outputs, _ = self.lstm(outputs)
            time_step, batch_size, h = outputs.shape
            outputs = outputs.view(time_step * batch_size, h)
            outputs = self.fc(outputs)
            outputs = outputs.view(time_step, batch_size, -1)
        else:
            outputs = self.fc(outputs)
        return outputs

    def training(self, inputs, labels, labels_length):
        outputs = self.get_features(inputs)
        loss, lr = self.get_loss(outputs, labels, labels_length)
        return loss, lr

    def testing(self, inputs, labels, labels_length):
        predict = self.get_feature(inputs)
        pred_decode_labels = []
        labels_list = []
        correct_list = []
        error_list = []
        i = 0
        labels = labels.tolist()
        if self.word:
            outputs = predict.max(1)[1]
            for pred_labels in outputs:
                pred_decode_labels.append(pred_labels)
        else:
            outputs = predict.max(2)[1].transpose(0, 1)
            for pred_labels in outputs:
                decoded = []
                last_item = 0
                for item in pred_labels:
                    item = item.item()
                    if item == last_item:
                        continue
                    else:
                        last_item = item
                    if item != 0:
                        decoded.append(item)
                pred_decode_labels.append(decoded)

        for idx in labels_length.tolist():
            labels_list.append(labels[i: i + idx])
            i += idx
        if len(labels_list) != len(pred_decode_labels):
            raise Exception("origin labels length is {}, but pred labels length is {}".format(
                len(labels_list), len(pred_decode_labels)))
        for ids in range(len(labels_list)):
            if labels_list[ids][0] == pred_decode_labels[ids].item():
                correct_list.append(ids)
            else:
                error_list.append(ids)
        return pred_decode_labels, labels_list, correct_list, error_list

    def get_loss(self, predict, labels, labels_length):
        labels = torch.autograd.Variable(labels)
        if self.word:
            loss = self.loss(predict, labels.long().cuda())
        else:
            log_predict = predict.log_softmax(2).detach().requires_grad_()
            seq_len = torch.IntTensor([log_predict.shape[0]] * log_predict.shape[1])
            loss = self.loss(log_predict.cpu(), labels, seq_len, labels_length)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), self.scheduler.state_dict()['_last_lr'][-1]

    def save_model(self, path, net):
        torch.save(net, path)

    def get_device(self, gpu_id):
        if gpu_id == -1:
            device = torch.device('cpu'.format(str(gpu_id)))
        else:
            device = torch.device('cuda:{}'.format(str(gpu_id)))
        return device

    def variable_to_device(self, inputs, device):
        return torch.autograd.Variable(inputs).to(device)

    def get_random_tensor(self):
        width = self.resize[0]
        height = self.resize[1]
        if width == -1:
            w = 240
            h = height
        else:
            w = width
            h = height
        return torch.randn(1, self.image_channel, h, w, device='cpu')

    def export_onnx(self, net, dummy_input, graph_path, input_names, output_names, dynamic_ax):
        torch.onnx.export(net, dummy_input, graph_path, export_params=True, verbose=False,
                          input_names=input_names, output_names=output_names, dynamic_axes=dynamic_ax,
                          opset_version=12, do_constant_folding=True, _retain_param_name=False)