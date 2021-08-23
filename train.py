from loss import DiceLoss
from utils import poly_lr_scheduler
from tqdm import tqdm
from torch.cuda import amp
from model.build_BiSeNet import BiSeNet
from torch.utils.data import DataLoader
from ETL.get_dataset import get_dataset
from ETL.transform import *
import contextlib
from pprint import pprint
import argparser
from val import getVal
from datetime import datetime
import sys
import os
import warnings
from makeWriter import getWriter
from tasks import get_steps

warnings.filterwarnings(action="ignore")
sys.path.append(os.getcwd())

# setup the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@contextlib.contextmanager
def dummy_cm():
    yield


def train(args, model, optimizer, dataloader_train, dataloader_val, scaler):
    # Prepare the tensorboard
    writer = getWriter(dataloader_train, args, model)
    # init loss func
    losses = {"dice": DiceLoss(), "crossentropy": torch.nn.CrossEntropyLoss(
        ignore_index=255)}
    loss_func = losses[args.loss]
    if args.use_lrScheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            threshold=0.0001,
            min_lr=0,
        )
    max_miou = 0
    step = 0
    # start training
    for epoch in range(1, args.num_epochs + 1):
        model.train()

        # lr = optimizer.param_groups[0]['lr']
        lr = poly_lr_scheduler(optimizer, args.learning_rate,
                               iter=epoch, max_iter=args.num_epochs)
        loss_record = []
        principal_loss_record = []
        # progress bar
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description("epoch: {}/{}".format(epoch, args.num_epochs))
        for i, (data, label) in enumerate(dataloader_train):
          # print(label)
          label = label.type(torch.LongTensor)
          if args.use_gpu:
              data = data.to(device)
              label = label.to(device)

          # forward
          scaler=None
          if scaler:
              cm = amp.autocast()
          else:
              cm = dummy_cm()

          with cm:
              optimizer.zero_grad()
              output, output_sup1, output_sup2 = model(data)
              loss1 = loss_func(output, label)
              loss2 = loss_func(output_sup1, label)
              loss3 = loss_func(output_sup2, label)
              loss = loss1 + loss2 + loss3

          # backward
          
          if scaler:
              scaler.scale(loss).backward()
              scaler.step(optimizer)
              scaler.update()
          else:
              loss.backward()
              optimizer.step()
          if args.use_lrScheduler:
              scheduler.step(loss)

          tq.update(args.batch_size)
          tq.set_postfix(loss=f"{loss:.4f}", lr=lr)
          step += 1
          # log the progress
          writer.add_scalar("loss_step", loss, step)
          loss_record.append(loss.item())
          principal_loss_record.append(loss1.item())

        tq.close()
        loss_train_mean = np.mean(loss_record)
        pri_train_mean = np.mean(principal_loss_record)
        writer.add_scalar("epoch/loss_epoch_train",
                          float(loss_train_mean), epoch)
        writer.add_scalar("epoch/pri_loss_epoch_train",
                          float(pri_train_mean), epoch)


        if epoch % args.checkpoint_step == 0 or epoch == args.num_epochs:
            if not os.path.isdir(args.save_model_path):
                os.makedirs(args.save_model_path, exist_ok=True)
            torch.save(model.state_dict(),os.path.join(args.save_model_path, f"epoch_{epoch}_.pth"),)

        if (epoch % args.validation_step == 0 or epoch == args.num_epochs or epoch == 1):
            precision, miou, val_loss = getVal(args, model, dataloader_val, loss_func)
            if miou > max_miou:
                max_miou = miou
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_model_path,
                                 f"best_model.pth"),)

            writer.add_scalar("epoch/precision_val", precision, epoch)
            writer.add_scalar("epoch/miou_val", miou, epoch)
            writer.add_scalar("epoch/loss_val", loss, epoch)
            print("epoch: {}, train_loss: {}, val_loss: {}, val_precision: {}, val_miou: {}".format(
                epoch, pri_train_mean, val_loss, precision, miou
            ))
        writer.flush()

    writer.close()


def get_optim(args, model):
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        raise ValueError(
            f"optimizer not supported optimizer: {args.optimizer}")

    return optimizer


def main(params):
    # parse the parameters
    parser = argparser.get_argparser()
    args = parser.parse_args(params)

    # model_id = 0
    # create dataset and dataloader
    #steps = get_steps(args.task)
    #for step in steps:
    #args.step = step
    args.save_model_path = os.path.join(args.save_model_dir, str(args.step))
    print("Training with following arguments:")
    pprint(vars(args), indent=4, compact=True)
    print("Running on: {}".format(device if args.use_gpu else torch.device('cpu')))
    # get_dataset
    train_dst, val_dst, _, n_classes = get_dataset(args)
    dataloader_train = DataLoader(train_dst, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  drop_last=True, )
    dataloader_val = DataLoader(val_dst, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, )

    # build model
    args.num_classes = n_classes
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if args.use_gpu:
        model = model.to(device)
    
    # build optimizer
    optimizer = get_optim(args, model)
    scaler = amp.GradScaler() if args.use_amp else None

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val, scaler)
    print("Training completed.", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    #model_id += 1


def fix_seed(seed=44):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    params = [
        "--num_epochs", "20",
        "--batch_size", "16",
        "--task", "15-5s-ov",
        "--step","0",
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

    ]
    print("started:", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    fix_seed(44)
    main(params)
    # main(sys.argv[1:])
