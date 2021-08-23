import torch
from tqdm import tqdm
from utils import compute_global_accuracy, fast_hist, per_class_iu
import numpy as np

def getVal(args, model, dataloader, loss_func):
    # print("start val!")
    # label_info = get_label_info(csv_path)
    tq = tqdm(total=len(dataloader) * args.batch_size)
    tq.set_description("validating:")
    with torch.no_grad():
        model.eval()
        precision_record = []
        loss_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()

            output = model(data)
            loss = loss_func(output, label)
            loss_record.append(loss.item())
            # get RGB predict image
            _, prediction = output.max(dim=1)  # B, H, W
            label = label.cpu().numpy()
            prediction = prediction.cpu().numpy()

            # compute per pixel accuracy
            precision = compute_global_accuracy(prediction, label)
            hist += fast_hist(label.flatten(),
                              prediction.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        tq.close()
        loss_mean = np.mean(loss_record)
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)[:-1]
        miou = np.mean(miou_list)
        return precision, miou, loss_mean
