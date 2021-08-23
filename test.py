import os
import torch
from ensemble import Ensemble
from torch.utils.data import DataLoader
from ETL.get_dataset import get_dataset
from model.build_BiSeNet import BiSeNet
from datetime import datetime
from loss import DiceLoss
from tasks import get_classes
import argparser
from val import getVal
import numpy as np
from makeWriter import getWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(args, model, data_loader):
    writer = getWriter(data_loader, args, model)
    # init loss func
    losses = {"dice": DiceLoss(), "crossentropy": torch.nn.CrossEntropyLoss(
        ignore_index=255)}
    loss_func = losses[args.loss]
    precision, miou, loss = getVal(args, model, data_loader, loss_func)
    writer.add_scalar("precision_val", precision)
    writer.add_scalar("miou_val", miou)
    writer.add_scalar("loss_val", loss)
    writer.close()


def main(params):
    # parse the parameters
    parser = argparser.get_argparser()
    args = parser.parse_args(params)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    _, _, test_dst, _ = get_dataset(args)
    dataloader_test = DataLoader(test_dst, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, )
    # build model
    models = []
    model_id = 0
    num_classes = get_classes(args.trained)
    for classes in num_classes:
        model = BiSeNet(classes, args.context_path)
        if args.use_gpu:
            model = model.to(device)
        # load pretrained model
        args.pretrained_model_path = os.path.join(args.save_model_dir, str(model_id),f"best_model.pth")
        print("load model from %s ..." % args.pretrained_model_path)
        model.load_state_dict(torch.load(args.pretrained_model_path))
        models.append(model)
        model_id+=1

    ensemble = Ensemble(models, args.alpha)
    test(args, ensemble, dataloader_test)


def fix_seed(seed=44):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    params = [
        "--alpha", "2",
        "--step", "0",
        "--task", "offline-ov",
        "--trained","15-5-ov",
        "--num_epochs", "30",
        "--batch_size", "32",
        "--learning_rate", "0.005",
        "--context_path", "resnet50",  # set resnet18, resnet50 or resnet101
        "--optimizer", "sgd",
        "--crop_size", "320",
        "--data_root", "/content/root_drive/MyDrive/data",
        "--save_model_dir", "/content/root_drive/MyDrive/models",
        "--num_workers", "12",
        "--validation_step", "2",
        "--num_classes", "21",
        "--cuda", "0",
        "--use_gpu", "True",
        "--use_amp", "True",
        "--use_lrScheduler", "False",
        "--further_data_aug", "False",
        "--pretrained_model_path", "/root_drive/MyDrive/models/res18_20_01_sgd/0.pth"
    ]
print("started:", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
fix_seed(44)
main(params)
