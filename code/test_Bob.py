# coding:utf-8
import os
import argparse
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util.MF_dataset import MF_dataset
from util.util import calculate_accuracy, calculate_result

from model.MFNet_Ray_pre_awgn import MFNet_Ray
from train_Ray import n_class, data_dir, model_dir

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Monte Carlo number
num = 100

def main():
    
    cf = np.zeros((n_class, n_class))

    model = eval(args.model_name)(n_class=n_class)
    if args.gpu >= 0: model.to(device)
    print('| loading model file %s... ' % final_model_file, end='')
    model.load_state_dict(torch.load(final_model_file))
    print('done!')
    

    test_dataset  = MF_dataset(data_dir, 'test', have_label=True)
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader.n_iter = len(test_loader)

    ave_overall_acc = np.zeros((num, 1))
    ave_acc = np.zeros((num, n_class))
    ave_IoU = np.zeros((num, n_class))
    for n in range(num):
        print('|------------------------------Iter num %s----------------------------' % (n))
        loss_avg = 0.
        acc_avg  = 0.
        model.eval()
        with torch.no_grad():
            for it, (images, labels, names) in enumerate(test_loader):
                images = Variable(images)
                labels = Variable(labels)
                if args.gpu >= 0:
                    images = images.to(device)
                    labels = labels.to(device)

                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                acc = calculate_accuracy(logits, labels)
                loss_avg += float(loss)
                acc_avg  += float(acc)

                # print('|- test iter %s/%s. loss: %.4f, acc: %.4f' \
                #         % (it+1, test_loader.n_iter, float(loss), float(acc)))

                predictions = logits.argmax(1)
                for gtcid in range(n_class): 
                    for pcid in range(n_class):
                        gt_mask      = labels == gtcid 
                        pred_mask    = predictions == pcid
                        intersection = gt_mask * pred_mask
                        cf[gtcid, pcid] += int(intersection.sum())

        overall_acc, acc, IoU = calculate_result(cf)

        print('| overall accuracy:', overall_acc)
        print('| accuracy of each class:', acc)
        print('| class accuracy avg:', acc.mean())
        print('| IoU:', IoU)
        print('| class IoU avg:', IoU.mean())

        ave_overall_acc[n, 0] = overall_acc
        ave_acc[n, :] = acc
        ave_IoU[n, :] = IoU


    print('| average overall accuracy:', np.mean(ave_overall_acc, axis=0))
    print('| average accuracy of each class:', np.mean(ave_acc, axis=0))
    print('| average class accuracy avg:', np.mean(ave_acc, axis=0).mean())
    print('| average IoU:', np.mean(ave_IoU, axis=0))
    print('| average class IoU avg:', np.mean(ave_IoU, axis=0).mean())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test MFNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='MFNet_Ray')
    parser.add_argument('--batch_size',  '-B',  type=int, default=64)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()

    model_dir        = os.path.join(model_dir, args.model_name)
    final_model_file = os.path.join(model_dir, 'final.pth')
    assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

    main()
