import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getWriter(data, args, model):
    comment = "Optimizer {}, lr {}, batch_size {}".format(
        args.optimizer, args.learning_rate, args.batch_size
    )
    writer = SummaryWriter(comment=comment)
    images, _ = iter(data).next()
    grid = torchvision.utils.make_grid(images)
    images = images.to(device) if args.use_gpu else images
    writer.add_image("images", grid, 0)
    #writer.add_graph(model, images)
    return writer
