import os
import numpy as np
import argparse
from datetime import datetime
import logging
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from lib.networks import PVT_GCASCADE, MERIT_GCASCADE
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from utils.utils import powerset
from utils.training_checkpoint import load_training_checkpoint, save_training_checkpoint

from ptflops import get_model_complexity_info


l = [0, 1, 2, 3]
ss = [x for x in powerset(l)]
print(ss)
SIZE_RATES = [0.75, 1.0, 1.25]
WEIGHT_DECAY = 1e-4
STRUCTURE_LOSS_ID = 'weighted_bce_plus_weighted_iou_v1_all_15_decoder_combinations'
        
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
 
    return (wbce + wiou).mean()


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, opt.img_size)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res1, res2, res3, res4 = model(image) # forward
        
        
        res = F.upsample(res1 + res2 + res3 + res4, size=gt.shape, mode='bilinear', align_corners=False) # additive aggregation and upsampling
        res = res.sigmoid().data.cpu().numpy().squeeze() # apply sigmoid activation for binary segmentation
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        # eval Dice
        input = res >= 0.5
        target = np.array(gt >= 0.5)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1

def train(
    train_loader,
    model,
    optimizer,
    epoch,
    test_path,
    model_name='PVT-GCASCADE',
    resume_checkpoint_path=None,
    resume_config=None,
):
    model.train()
    global best
    global total_train_time
    time_before_epoch_start = time.time()
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in SIZE_RATES:
            optimizer.zero_grad()
            
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
        
            # ---- rescale ----
            trainsize = int(round(opt.img_size * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            # ---- forward ----
            P = model(images)
                       
            loss = 0.0
          
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                for idx in range(len(s)):
                    iout += P[s[idx]]
                loss += structure_loss(iout, gts)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        # ---- train visualization ----
        if i % 50 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' loss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))
            
    time_after_epoch_end = time.time()
    total_train_time += (time_after_epoch_end - time_before_epoch_start)
    print('total train time till current epoch: '+ str(total_train_time))
    logging.info('total train time till current epoch: '+ str(total_train_time))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + '' + model_name + '-last.pth')
    # choose the best model

    global dict_plot
   
    if (epoch + 1) % 1 == 0:
    	total_dice = 0
    	total_images = 0
    	for dataset in ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'CVC-300']:
    	    dataset_dice, n_images = test(model, test_path, dataset)
    	    total_dice += (n_images*dataset_dice)
    	    total_images += n_images
    	    logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
    	    print(dataset, ': ', dataset_dice)
    	    dict_plot[dataset].append(dataset_dice)
    	meandice = total_dice/total_images
    	dict_plot['test'].append(meandice)
    	print('Validation dice score: {}'.format(meandice))
    	logging.info('Validation dice score: {}'.format(meandice))
    	if meandice > best:
            print('##################### Dice score improved from {} to {}'.format(best, meandice))
            logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
            best = meandice
            torch.save(model.state_dict(), save_path + '' + model_name + '-best.pth')

    # This is deliberately last: it represents a completed epoch whose validation
    # result, best score, model-only files, optimizer, and RNG state agree.
    if resume_checkpoint_path is not None:
        save_training_checkpoint(
            resume_checkpoint_path,
            model,
            optimizer,
            epoch,
            best,
            total_train_time,
            resume_config,
        )
        print('Saved resumable checkpoint: {}'.format(resume_checkpoint_path))
        logging.info('Saved resumable checkpoint: {}'.format(resume_checkpoint_path))


def resume_config(opt):
    """Settings that must match for a faithful epoch-boundary continuation."""
    return {
        'architecture': '{}_GCASCADE'.format(opt.encoder),
        'encoder': opt.encoder,
        'skip_aggregation': opt.skip_aggregation,
        'n_class': 1,
        'graph_k': 11,
        'graph_padding': 5,
        'graph_convolution': 'mr',
        'graph_activation': 'gelu',
        'optimizer': opt.optimizer,
        'learning_rate': float(opt.lr),
        'weight_decay': WEIGHT_DECAY,
        'batch_size': int(opt.batchsize),
        'image_size': int(opt.img_size),
        'gradient_clip': float(opt.clip),
        'decay_rate': float(opt.decay_rate),
        'decay_epoch': int(opt.decay_epoch),
        'augmentation': str(opt.augmentation),
        'train_path': os.path.abspath(os.path.normpath(opt.train_path)),
        'test_path': os.path.abspath(os.path.normpath(opt.test_path)),
        'loss': STRUCTURE_LOSS_ID,
        'decoder_output_combinations': 15,
        'multi_scale_rates': list(SIZE_RATES),
    }


def ensure_fresh_outputs_do_not_exist(
    save_path, model_name, checkpoint_path, log_path=None
):
    outputs = [
        os.path.join(save_path, model_name + '-best.pth'),
        os.path.join(save_path, model_name + '-last.pth'),
    ]
    if checkpoint_path is not None:
        outputs.append(checkpoint_path)
    if log_path is not None:
        outputs.append(log_path)
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            'Fresh training would overwrite existing output(s): {}. '
            'Choose a new --experiment_name/--train_save path, or use --resume.'
            .format(existing)
        )
    
if __name__ == '__main__':
    dict_plot = {'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'CVC-300':[],'test':[]}
    name = ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'CVC-300', 'test']
    ##################model_name#############################
    default_model_name = 'Polyp_PVT_GCASCADE_img_size352bs4_Run1'
    
    #current_time = time.strftime("%H%M%S")
    #print("The current time is", current_time)
    #model_name = model_name +'_t'+current_time
    ###############################################
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--encoder', type=str,
                        default='PVT', help='Name of encoder: PVT or MERIT')
    
    parser.add_argument('--skip_aggregation', type=str,
                        default='additive', help='Type of skip-aggregation: additive or concatenation')

    parser.add_argument('--epoch', type=int,
                        default=200,
                        help='exclusive epoch limit; 200 runs epochs 1 through 199')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')

    parser.add_argument('--img_size', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=300, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./data/polyp/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./data/polyp/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--experiment_name', type=str,
                        default=default_model_name,
                        help='model filename prefix and default output-directory name')

    parser.add_argument('--train_save', type=str, default=None,
                        help='output directory (default: ./model_pth/<experiment_name>/)')

    parser.add_argument('--resume', type=str, default=None,
                        help='full-state checkpoint to resume from')

    parser.add_argument('--save_resume_checkpoint', type=str, default=None,
                        help='full-state checkpoint path written after each completed epoch; '
                             'defaults to --resume when resuming')

    opt = parser.parse_args()
    model_name = opt.experiment_name
    if opt.train_save is None:
        opt.train_save = os.path.join('.', 'model_pth', model_name) + os.sep
    elif not opt.train_save.endswith(('/', '\\')):
        opt.train_save += os.sep
    if opt.resume is not None and opt.save_resume_checkpoint is None:
        opt.save_resume_checkpoint = opt.resume
    log_path = os.path.join(opt.train_save, 'train.log')
    if opt.resume is None:
        ensure_fresh_outputs_do_not_exist(
            opt.train_save, model_name, opt.save_resume_checkpoint, log_path
        )
    os.makedirs(opt.train_save, exist_ok=True)

    logging.basicConfig(filename=log_path,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    #torch.cuda.set_device(0)  # set your gpu device
    if opt.encoder=='PVT':
        model = PVT_GCASCADE(n_class=1, img_size=opt.img_size, k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation=opt.skip_aggregation)
    elif opt.encoder=='MERIT':
        model = MERIT_GCASCADE(n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation=opt.skip_aggregation)
    else:
        raise ValueError('Implementation not found for encoder: {}'.format(opt.encoder))

    print('Model %s created' % (opt.encoder+'-GCASCADE: '))
    
    model.cuda()
    macs, params = get_model_complexity_info(model, (3, opt.img_size, opt.img_size), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
	
    best = 0

    params = model.parameters()
    

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=WEIGHT_DECAY)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=WEIGHT_DECAY, momentum=0.9)
    else:
        raise ValueError('Unsupported optimizer: {}'.format(opt.optimizer))

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.img_size, shuffle=True,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)
    total_train_time = 0
    best = 0
    start_epoch = 1
    checkpoint_config = resume_config(opt)

    # Restore only after model, optimizer, profiling, and DataLoader construction.
    # The next DataLoader iterator will therefore consume the restored RNG state.
    if opt.resume is not None:
        metadata = load_training_checkpoint(
            opt.resume, model, optimizer, expected_config=checkpoint_config
        )
        start_epoch = metadata['epoch'] + 1
        best = metadata['best_dice']
        total_train_time = metadata['total_train_time']
        print(
            'Resumed completed epoch {} (next epoch {}, best Dice {}, total train time {}).'
            .format(metadata['epoch'], start_epoch, best, total_train_time)
        )
        logging.info(
            'Resumed completed epoch %s; next epoch %s; best Dice %s; total train time %s',
            metadata['epoch'], start_epoch, best, total_train_time
        )

    last_completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(
            train_loader,
            model,
            optimizer,
            epoch,
            opt.test_path,
            model_name=model_name,
            resume_checkpoint_path=opt.save_resume_checkpoint,
            resume_config=checkpoint_config,
        )
        last_completed_epoch = epoch
    if last_completed_epoch > 0:
        average_train_time = total_train_time / last_completed_epoch
        print('avg train time: '+ str(average_train_time))
        logging.info('avg train time: '+ str(average_train_time))
    else:
        print('No epochs were completed.')
