import logging
import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils.inc_net import IncrementalNet, CosineIncrementalNet
from models.base import BaseLearner
from utils.toolkit import  tensor2numpy
from utils.attack import Attack
import math



EPSILON = 1e-8

class HyperInversion(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if self.args["cosine"]:
            self._network = CosineIncrementalNet(args, False)
        else:
            self._network = IncrementalNet(args, False)
        self._protos = []
        self._vars_static=[]  #add
        self._means_static =[] #add
        self.spatial_means=[]
        self.spatial_vars=[]
        self.input_dims = (args["pca_num"],args["patch_size"],args["patch_size"])

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        if not self.args['resume']:
            if not os.path.exists(self.args["model_dir"]):
                os.makedirs(self.args["model_dir"])
            #self.save_checkpoint("{}{}".format(self.args["model_dir"],self.args["dataset"][0]))
            self.save_checkpoint("{}{}_seed{}".format(self.args["model_dir"],self.args["dataset"][0],self.args["seed"]))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )


        if self.args["cosine"]:
            self._network.update_fc(self._total_classes, self._cur_task)
        else:
            self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        resume = self.args['resume']  # set resume=True to use saved checkpoints
        if self._cur_task == 0:
            if resume:
                #self._network.load_state_dict(torch.load("{}{}_{}.pkl".format(self.args["model_dir"],self.args["dataset"][0],self._cur_task))["model_state_dict"], strict=False)
                self._network.load_state_dict(torch.load("{}{}_seed{}_{}.pkl".format(self.args["model_dir"],self.args["dataset"][0],self.args["seed"],self._cur_task))["model_state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if not resume:
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args["lrate"], weight_decay=self.args["weight_decay"])
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lr_decay"])
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            self._build_protos()
        else:
            resume = self.args['resume']
            if resume:
                self._network.load_state_dict(torch.load("{}{}_seed{}_{}.pkl".format(self.args["model_dir"],self.args["dataset"][0],self.args["seed"],self._cur_task))["model_state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if self._old_network is not None:
                self._old_network.to(self._device)
            if not resume:
                optimizer = optim.SGD(self._network.parameters(), lr=self.args["lrate"], momentum=0.9, weight_decay=self.args["weight_decay"])
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lr_decay"])
                self._update_representation(train_loader, test_loader, optimizer, scheduler)
            self._build_protos()  


            



    def _build_protos(self):
        for class_idx in range(self._known_classes, self._total_classes):
            idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train')
            idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False)
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0) # vectors.mean(0)
            self._protos.append(torch.tensor(class_mean).to(self._device))
            
            datas=[]
            for inputs, _ in idx_loader:
                datas.append(inputs)
            tt_vectors = torch.cat(datas)
            nch = tt_vectors.shape[1] #add
            mean = tt_vectors.mean([0, 2, 3]) #add
            var = tt_vectors.permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8 #add
            self._means_static.append(mean.to(self._device)) #add
            self._vars_static.append(var.to(self._device)) #add

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 25 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(self.args["epochs"]))
        inv_inputs,inv_targets = self.sample(self.args["batch_size"])
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            mappings = torch.ones(self._total_classes, dtype=torch.float32)
            mappings = mappings.to(self._device)
            rnt = 1.0 * self._known_classes/self._total_classes
            mappings[:self._known_classes] = rnt
            mappings[self._known_classes:] =1- rnt
            for i, (inputs, targets) in enumerate(train_loader):
                len_input = inputs.shape[0]
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs_com = inputs
                targets_com = targets

                with torch.no_grad():
                    sel_means = torch.stack(self._means_static)[inv_targets] #add
                    sel_vars = torch.stack(self._vars_static)[inv_targets]#add
                    nch = inv_inputs.shape[1]
                    for i in range(inv_inputs.shape[0]):
                        means_0 = inv_inputs[i].mean([1,2])
                        var_0 = inv_inputs[i].contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8 #add
                        #param2 = sel_means[i]-means_0*sel_vars[i]**(0.5)/var_0**(0.5)
                        param1 = sel_means[i]
                        param2 = sel_vars[i]**(0.5)
                        en = torch.normal(param1,param2)
                        en = en.unsqueeze(dim=1)
                        en = en.unsqueeze(dim=2)
                        param1 = param1.unsqueeze(dim=1)
                        param1 = param1.unsqueeze(dim=2)
                        param2 = param2.unsqueeze(dim=1)
                        param2 = param2.unsqueeze(dim=2)
                        #inv_inputs[i] = (inv_inputs[i] + param1)/2
                        inv_inputs[i] =  self.args["ase_param"] * inv_inputs[i] + (1-self.args["ase_param"])*en


                    inputs_com,targets_com = self.combine_data(((inputs,targets),(inv_inputs,inv_targets)))
                    dw_cls = mappings[targets_com.long()]

                feats = self._network.convnet(inputs_com)["features"]
                logits = self._network.fc(feats)["logits"]
                #logits = self._network(inputs_com)["logits"]
                loss = 0

                loss_clf = F.cross_entropy(logits, targets_com)
                loss += loss_clf
                #logits_old = self._old_network(inputs_com)["logits"]
                #logits_new = self._old_network.fc(feats)["logits"]
                #loss_kd = _KD_loss(logits_new, logits_old,self.args["T"])
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs_com)["logits"],
                    self.args["T"],
                )
                loss += self.args["lamda"] * loss_kd
                feats_old = self._old_network.convnet(inputs_com)["features"] 
                fea_dis = torch.sqrt(torch.sum( torch.pow(feats_old-feats,2),dim=-1 ))
                fea_dis = torch.mean(fea_dis)
                loss = self.args["canshu1"] *fea_dis + loss


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
    
                with torch.no_grad():
                    _, preds = torch.max(logits[:len_input], dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 25 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y
    
    def sample(self,sample_size):
        #torch.cuda.empty_cache()
        content_temp = 1e3
        content_weight= 1
        di_var_scale =1e-3
        gen_criterion = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss(reduction="none").to(self._device)
        smoothing = Gaussiansmoothing(36,5,1,self._device)
        loss_r_feature_layers = []
        r_feature_weight=5e1
        for module in self._old_network.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, r_feature_weight))
        target = torch.randint(self._known_classes,(sample_size,))
        num_per_cls = int(sample_size/self._known_classes)
        for i_cls in range(self._known_classes):
            target[i_cls*num_per_cls:(i_cls+1)*num_per_cls] = i_cls
        shape = (target.shape[0],) + self.input_dims
        inputs = torch.randn(shape).to(self._device).requires_grad_(True)
        target = target.to(self._device)
        gen_opt = torch.optim.Adam([inputs], lr=1e-3)
        for epoch in range(self.args["gen_epoches"]):
            gen_opt.zero_grad()
            #torch.set_grad_enabled(True)
            outputs = self._old_network(inputs)["logits"]
            num_k = self._known_classes
            loss = gen_criterion(outputs/content_temp,target)*content_weight
            inputs_smooth = smoothing(F.pad(inputs, (2, 2, 2, 2), mode='constant'))
            loss_var = mse_loss(inputs, inputs_smooth).mean()
            #gen_proto = torch.stack(self._protos)[torch.argmax(outputs, dim=1)]
            #loss_kd = _KD_loss(feats,gen_proto,self.args["T"])
            #loss = loss + loss_kd
            loss = loss + di_var_scale * loss_var

            for mod in loss_r_feature_layers: 
                loss_distr = mod.r_feature * r_feature_weight / len(loss_r_feature_layers)
                loss = loss + loss_distr
            
            sel_means = torch.stack(self._means_static)[target] #add
            sel_vars = torch.stack(self._vars_static)[target]#add
            nch = inputs.shape[1]
            loss_mean_var = 0
            for i in range(inputs.shape[0]):#add
                mean_0 = inputs[i].mean([1,2])#add
                var_0 = inputs[i].contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8 #add
                mean_tar = sel_means[i]#add
                var_tar = sel_vars[i]#add
                loss_mean_var = loss_mean_var+ torch.log(var_0**(0.5) / (var_tar)**(0.5)).mean() - 0.5 * (1.0 - (var_tar + (mean_0-mean_tar)**2)/var_0).mean()#add

            loss_mean_var = loss_mean_var/inputs.shape[0]
            loss = loss + r_feature_weight*loss_mean_var #add

            loss.backward()
            gen_opt.step()
            #torch.set_grad_enabled(False)
        return inputs.detach(),target 

      

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, gram_matrix_weight, layer_weight):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None
        self.gram_matrix_weight = gram_matrix_weight
        self.layer_weight = layer_weight

    def hook_fn(self, module, input, output):

        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        r_feature = torch.log(var**(0.5) / (module.running_var.data.type(var.type()) + 1e-8)**(0.5)).mean() - 0.5 * (1.0 - (module.running_var.data.type(var.type()) + 1e-8 + (module.running_mean.data.type(var.type())-mean)**2)/var).mean()

        self.r_feature = r_feature

            
    def close(self):
        self.hook.remove()

class Gaussiansmoothing(nn.Module):
  
    def __init__(self, channels, kernel_size, sigma,device, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).to(device)
        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        '''
        t_conv = torch.nn.Conv2d(channels,channels,kernel_size= kernel_size,bias=False,groups=self.groups)
        t_conv.weight.data = self.weight
        t_conv.requires_grad_(False)
        t_conv.to(device)
        self.t_conv = t_conv
        '''

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


