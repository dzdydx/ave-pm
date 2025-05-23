import os
import time
import random
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
 
from losses import AVPSLoss, segment_contrastive_loss, video_contrastive_loss, LabelFreeSelfSupervisedNCELoss
import pdb

import yaml
import argparse

parser = argparse.ArgumentParser(description="Training args")

parser.add_argument("--config", type=str, default="config/dy_cpsp.yml")
# configs
config_path = parser.parse_args().config
with open(config_path, "r") as f:
    args_map = yaml.safe_load(f)

args = type('DynamicObject', (), args_map)()  #动态构造了一个对象 args，可以用 args.lr、args.epochs 方式访问这些参数

 # =================================  seed config ============================
SEED = args.seed
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =============================================================================


# select GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

'''Create snapshot_pred dir for copying code and saving models '''
if not os.path.exists(args.snapshot_pref):
    os.makedirs(args.snapshot_pref, exist_ok=True)

if os.path.isfile(args.resume):
    args.snapshot_pref = os.path.dirname(args.resume)

logger = Prepare_logger(args, eval=args.evaluate)

if not args.evaluate:
    logger.info(f'\nCreating folder: {args.snapshot_pref}')
    logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(args_map, indent=4)))
else:
    logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

"""dataset selection"""
from dataset.dataset import AVEDataset

def main():
    '''Dataloader selection'''
    is_select = args.is_select
    audio_process_mode = args.audio_preprocess_mode
    data_root = args.data_root
    meta_root = args.meta_root
    ave = args.ave
    avepm = args.avepm
    if ave == False and avepm == False:
        raise ValueError("Please choose one of the two datasets: AVE or AVEPM")
    if ave == True and avepm == True:
        raise ValueError("Please choose only one of the two datasets: AVE or AVEPM")
    if args.is_select:
        args.category_num = 10
    else:
        if args.ave:
            args.category_num = 28
        elif args.avepm:
            args.category_num = 86
            
    preprocess_mode = args.preprocess
    v_feature_root = args.v_feature_root
    a_feature_root = args.a_feature_root

    # MARK: Dataset
    train_dataloader = DataLoader(
        AVEDataset(
            split='train',
            data_root=data_root,
            meta_root=meta_root,
            ave=ave,
            avepm=avepm,
            preprocess_mode=preprocess_mode,
            audio_process_mode=audio_process_mode,
            is_select=is_select,
            a_feature_root=a_feature_root,
            v_feature_root=v_feature_root),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        AVEDataset(
            split='val',
            data_root=data_root,
            meta_root=meta_root,
            ave=ave,
            avepm=avepm,
            preprocess_mode=preprocess_mode,
            audio_process_mode=audio_process_mode,
            is_select=is_select,
            a_feature_root=a_feature_root,
            v_feature_root=v_feature_root),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        AVEDataset(
            split='test',
            data_root=data_root,
            meta_root=meta_root,
            ave=ave,
            avepm=avepm,
            preprocess_mode=preprocess_mode,
            audio_process_mode=audio_process_mode,
            is_select=is_select,
            a_feature_root=a_feature_root,
            v_feature_root=v_feature_root),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )


    ''' Model selection '''
    if 'psp' in args.model:
        print(f'Load {args.model} network')
        if args.vis_fea_type == 'vgg':
            from model.psp_family import fully_psp_net as main_model
            mainModel = main_model(vis_fea_type=args.vis_fea_type, flag=args.model, category_num=args.category_num, thr_val=args.threshold_value) 
    elif args.model == 'avel':
        print(f'Load AVEL network')
        from model.avel import TBMRF_Net as main_model
        mainModel = main_model(vis_fea_type=args.vis_fea_type, flag='fully', category_num=args.category_num)

    '''optimizer setting'''
    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    
    """loss"""
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError


    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart testing..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    best_accuracy, best_accuracy_epoch = 0, 0
    '''Training and Evaluation'''
    for epoch in range(args.n_epoch):
        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, val_dataloader, criterion, criterion_event, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                logger.info(f'best accuracy at epoch-{epoch}: {best_accuracy:.4f}')
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='FullySupervised',
                    epoch=epoch + 1,
                    seed=SEED
                )
        scheduler.step()
    
    print(f'best accuracy at epoch-{best_accuracy_epoch}: {best_accuracy:.4f}')


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    # Note: here we set the model to a double type precision,
    # since the extracted features are in a double type.
    # This will also lead to the size of the model double increases.
    model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # For a model in a float precision
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        visual_feature = visual_feature.double()
        audio_feature = audio_feature.double()
        labels = labels.double().cuda()

        """forward"""
        if args.model == 'cmran': # should be one of ['avel', 'avsdn', 'cmran', 'psp', 'cpsp', 'sspsp']
            is_event_scores, event_scores = model(visual_feature, audio_feature) # for CMRAN
            is_event_scores = is_event_scores.transpose(1, 0).squeeze().contiguous() # [10, 32], for CMRAN
        elif args.model == 'sspsp':
            is_event_scores, event_scores, final_v_fea, final_a_fea = model(visual_feature, audio_feature)
        else:
            is_event_scores, event_scores, avps, fusion = model(visual_feature, audio_feature)
        # print('is_event_scores.shape: ', is_event_scores.shape) [32, 10]
        # print('event_scores.shape: ', event_scores.shape) [32, 28]
        
        """some processing on the labels"""
        labels_foreground = labels[:, :, :-1]  # [32, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1) # [32, 10], [32, 10]
        labels_event, _ = labels_evn.max(-1) # [32]

        event_flag, pos_flag, neg_flag = get_flag_by_gt(labels_BCE)
        event_class_flag = labels_event

        """compute loss and backward"""
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        loss = loss_is_event + loss_event_class
        loss_avps, loss_scon, loss_vcon = torch.tensor(0).cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda()
        if args.avps_flag:
            soft_labels = labels_BCE / (labels_BCE.sum(-1, keepdim=True) + 1e-6)
            loss_avps = AVPSLoss(avps, soft_labels)
            loss += args.lambda_avps * loss_avps
        if args.scon_flag:
            loss_scon = segment_contrastive_loss(fusion, event_flag, pos_flag, neg_flag)
            loss += args.lambda_scon * loss_scon
        if args.vcon_flag:
            loss_vcon = video_contrastive_loss(fusion, event_class_flag)
            loss += args.lambda_vcon * loss_vcon
        if args.model == 'sspsp':
            loss_sscon = LabelFreeSelfSupervisedNCELoss(final_a_fea, final_v_fea)
            loss += args.lambda_sscon * loss_sscon

        loss.backward()

        '''Compute Accuracy'''
        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels, bg_flag=args.category_num)
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #     logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), visual_feature.size(0)*10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # '''Add loss of a iteration in Tensorboard'''
        # writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg: .3f})\t'
                f'loss_is_event {loss_is_event.item():.3f}\t'
                f'loss_event_class {loss_event_class.item():.3f}\t'
                f'loss_avps {loss_avps.item():.3f}\t'
                f'loss_scon {loss_scon.item():.3f}\t'
                f'loss_vcon {loss_vcon.item():.3f}'
            )

        # '''Add loss of an epoch in Tensorboard'''
        # writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg



@torch.no_grad()
def validate_epoch(model, val_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()

    model.eval()
    model.double()

    for n_iter, batch_data in enumerate(val_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # For a model in a float type
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        visual_feature = visual_feature.double()
        audio_feature = audio_feature.double()
        labels = labels.double().cuda()
        bs = visual_feature.size(0)
        if args.model == 'cmran':
            is_event_scores, event_scores = model(visual_feature, audio_feature) # for CMRAN
            is_event_scores = is_event_scores.transpose(1, 0).squeeze() # for CMRAN
        else:
            is_event_scores, event_scores, _, _ = model(visual_feature, audio_feature)
            
        """some processing on the labels"""
        labels_foreground = labels[:, :, :-1]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        loss = loss_is_event + loss_event_class

        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels, bg_flag=args.category_num)
        accuracy.update(acc.item(), bs * 10)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(val_dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})'
            )

    '''Add loss in an epoch to Tensorboard'''
    # if not eval_only:
    #     writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
    #     writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    logger.info(
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
    )
    return accuracy.avg



def get_flag_by_gt(is_event_scores):
    # is_event_scores: [bs, 10]
    scores_pos_ind = is_event_scores #> 0.5
    pred_temp = scores_pos_ind.long() # [B, 10]
    pred = pred_temp.unsqueeze(1) # [B, 1, 10]

    pos_flag = pred.repeat(1, 10, 1) # [B, 10, 10]
    pos_flag *= pred.permute(0, 2, 1)
    neg_flag = (1 - pred).repeat(1, 10, 1) # [B, 10, 10]
    neg_flag *= pred.permute(0, 2, 1)

    return pred_temp, pos_flag, neg_flag



def compute_accuracy_supervised(is_event_scores, event_scores, labels, bg_flag=28):
    # labels = labels[:, :, :-1]  # 28 denote background
    _, targets = labels.max(-1)
    # pos pred
    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5
    scores_mask = scores_pos_ind == 0
    _, event_class = event_scores.max(-1) # foreground classification, [B]
    pred = scores_pos_ind.long() # [B, 10]
    pred *= event_class[:, None]
    # add mask
    pred[scores_mask] = bg_flag # 28 denotes bg
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())

    return acc


def save_checkpoint(state_dict, top1, task, epoch, seed):
    model_name = f'{args.snapshot_pref}/model_seed_{seed}_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


if __name__ == '__main__':
    main()
