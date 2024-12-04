import sys

from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging

import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn

from skimage.measure import label
from torch.utils.data import DataLoader

from utils import losses, ramps, test_3d_patch, feature_memory, contrastive_losses
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.BCP_utils import mix_loss, update_ema_teacher, update_ema_variables


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_split/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='NEW', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=4000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=60000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')

args = parser.parse_args()

def get_pseudo_label(out, thres):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(np.array(batch_list)).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="VNet")

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    DICE = losses.mask_DiceLoss(nclass=2)
    model.train()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][
                                                                                  :args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs, feature = model(volume_batch)

            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)

            loss = (loss_ce + loss_dice) / 2


            iter_num += 1
            writer.add_scalar('pre/loss', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(
                'iteration %d : sumloss: %03f, seg_dice_loss: %03f, seg_ce_loss: %03f' % (
                    iter_num, loss, loss_dice, loss_ce))

            # if iter_num % 100 == 0:
            #     model.eval()
            #     dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
            #                                                 stride_xy=18, stride_z=4)
            #     if dice_sample > best_dice:
            #         best_dice = round(dice_sample, 4)
            #         save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
            #         save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
            #
            #         save_net_opt(model, optimizer, save_mode_path)
            #         save_net_opt(model, optimizer, save_best_path)
            #
            #         logging.info("save best model to {}".format(save_mode_path))
            #     writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
            #     writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
            #     model.train()

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="VNet")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="VNet")
    mse = nn.MSELoss()
    for param in ema_model.parameters():
        param.detach_()  # ema_model set
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')

    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)

    model.train()
    ema_model.train()

    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    DICE = losses.mask_DiceLoss(nclass=2)
    max_epoch = self_max_iterations // len(trainloader) + 1

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):

            labeled_volume, labeled_label = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][
                                                                                      :args.labeled_bs]
            labeled_volume, labeled_label = labeled_volume.cuda(), labeled_label.cuda()

            unlabeled_volume = sampled_batch['image'][args.labeled_bs:]
            unlabeled_volume = unlabeled_volume.cuda()


            outputs_u, feature_u = model(unlabeled_volume)
            outputs_l, feature_l = model(labeled_volume)

            with torch.no_grad():
                ema_output, feature_ema = ema_model(unlabeled_volume)
                ema_output_l, feature_ema = ema_model(labeled_volume)

            # thres_t = 0.5
            # thres = 0
            # dice = 10
            # with torch.no_grad():
            #     for i in range(10):
            #         pseudo_label = get_pseudo_label(ema_output_l, thres)
            #         dice_t = mse(pseudo_label, labeled_label)
            #         if dice_t < dice:
            #             dice = dice_t
            #             thres_t = thres
            #         thres += 0.1
            #     print(thres_t)
            #     pseudo_label = get_pseudo_label(ema_output, thres_t)
            #
            # loss, dice_loss, loss_ce = mix_loss(labeled_label, outputs_l, pseudo_label,
            #                                     outputs_u, u_weight=args.u_weight)


            S_sup = DICE(outputs_l,labeled_label)
            T_sup = DICE(ema_output_l,labeled_label)

            if T_sup < S_sup:
                print("Teacher win")
                thres_t = 0.5
                thres = 0
                dice = 10
                with torch.no_grad():
                    for i in range(10):
                        pseudo_label = get_pseudo_label(ema_output_l, thres)
                        dice_t = mse(pseudo_label, labeled_label)
                        if dice_t < dice:
                            dice = dice_t
                            thres_t = thres
                        thres += 0.1
                    print(thres_t)
                    pseudo_label = get_pseudo_label(ema_output, thres_t)

                loss, dice_loss, loss_ce = mix_loss(labeled_label, outputs_l, pseudo_label,
                                                    outputs_u, u_weight=args.u_weight)
            else:
                print("Student win")
                dice_loss = S_sup
                loss_ce = F.cross_entropy(outputs_l, labeled_label)
                loss = (dice_loss + loss_ce) / 2

            iter_num += 1

            writer.add_scalar('Self/loss', loss, iter_num)
            writer.add_scalar('Self/dice_loss', dice_loss, iter_num)
            writer.add_scalar('Self/loss_ce', loss_ce, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(
                'iteration %d : loss: %03f, seg_dice_loss: %03f, seg_ce_loss: %03f' % (
                    iter_num, loss, dice_loss, loss_ce))

            update_ema_variables(model, ema_model, 0.99)
            # update_ema_teacher_dy(model, ema_model, 0.99, iter_num)


            # if 28700<iter_num < 28900
            # if iter_num % 100 == 0 and 30000<iter_num:
            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes,
                                                            patch_size=patch_size,
                                                            stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)

                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))

                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)

                model.train()

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "./model/BCP/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)

    print("Strating training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('../code/LA_BCP_train.py', self_snapshot_path)

    # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
