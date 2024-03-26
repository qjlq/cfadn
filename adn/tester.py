import os
import os.path as path
import argparse
import torch
from adn.utils import \
    get_config, update_config, save_config,\
    get_last_checkpoint, add_post, Logger
from adn.datasets import get_dataset
from torch.utils.data import DataLoader


class Tester(object):
    def __init__(self, name, model_class, description="", project_dir="."):
        self.name = name
        self.model_class = model_class
        self.project_dir = project_dir
        self.description = description

    def parse_args(self):
        default_config = path.join(self.project_dir, "config", self.name + ".yaml")
        run_config = path.join(self.project_dir, "runs", self.name + ".yaml")

        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument("run_name", help="name of the run")
        parser.add_argument("--default_config", default=default_config, help="default configs")
        parser.add_argument("--run_config", default=run_config, help="run configs")

        args = parser.parse_args()
        return args

    def get_opts(self, args): 
        opts = get_config(args.default_config)
        run_opts = get_config(args.run_config)
        if args.run_name in run_opts and "test" in run_opts[args.run_name]:
            run_opts = run_opts[args.run_name]["test"]
            update_config(opts, run_opts)
        update_config(opts, args)
        run_dir = path.join(opts.checkpoints_dir, args.run_name)
        if not path.isdir(run_dir): os.makedirs(run_dir)
        save_config(opts, path.join(run_dir, "test_options.yaml"))

        self.run_dir = run_dir
        self.run_name = args.run_name
        self.opts = opts
        return opts

    def get_image(self, data):
        return data

    def get_loader(self, opts):
        self.dataset = get_dataset(**opts.dataset)
        loader = DataLoader(self.dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False) # load data
        self.loader = add_post(loader, self.get_image)
        return self.loader

    def get_checkpoint(self, opts):
        if opts.last_epoch == 'last':
            checkpoint, epoch = get_last_checkpoint(self.run_dir)
        else:
            epoch = opts.last_epoch
            #here import the train model
            checkpoint = path.join(self.run_dir, "net_{}.pt".format(epoch))
        if not path.isfile(checkpoint): raise IOError("Checkpoint not found!")
        self.epoch = epoch
        return checkpoint

    def get_model(self, opts, checkpoint):
        self.model = self.model_class(**opts.model)
        if opts.use_gpu: self.model.cuda() # use gpu
        self.model.resume(checkpoint)

        # 测试使用，测试并行
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        # device_ids = [0, 1]  # 可用GPU
        # self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        # self.model = self.model.cuda(device=device_ids[0])  # 模型加载到设备0
        # self.model.resume(checkpoint)


        return self.model

    def get_logger(self, opts):
        self.logger = Logger(self.run_dir, self.epoch, self.run_name)
        return self.logger

    def evaluate(self, model, data):
        pass

    def run(self):  ## tester.run start here
        args = self.parse_args() #load the conf file
        opts = self.get_opts(args) #load the option conf file
        loader = self.get_loader(opts) #load data
        checkpoint = self.get_checkpoint(opts) # if last run train not complete then resume the train from that point
        model = self.get_model(opts, checkpoint) # 加载模型

        device_ids = [0, 1]  # 可用GPU
        model = torch.nn.DataParallel(model, device_ids=device_ids)

        logger = self.get_logger(opts) #log set

        with torch.no_grad(): #使用 torch.no_grad() 来禁用自动求导
            for data in logger(loader):
                self.evaluate(model, data)
