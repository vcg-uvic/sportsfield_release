
import time

from torch.utils.data import DataLoader
import numpy as np

from utils import metrics, utils
from models import end_2_end_optimization
from options import options
from datasets import aligned_dataset


def main():
    utils.fix_randomness()

    opt = options.set_end2end_optim_options()
    assert opt.iou_space == 'part_and_whole'

    test_dataset = aligned_dataset.AlignedDatasetFactory.get_aligned_dataset(opt, 'test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0,)

    e2e = end_2_end_optimization.End2EndOptimFactory.get_end_2_end_optimization_model(opt)

    iou = metrics.IOU(opt)
    orig_iou_list = []
    optim_iou_list = []
    original_homography_list = []
    optim_homography_list = []
    gt_homography_list = []
    t0 = time.time()
    for i, data_batch in enumerate(test_loader):
        frame, _, gt_homography = data_batch
        orig_homography, optim_homography = e2e.optim(
            frame, test_dataset.template)
        orig_iou = iou(orig_homography, gt_homography)
        optim_iou = iou(optim_homography, gt_homography)
        orig_iou_list.append(orig_iou)
        optim_iou_list.append(optim_iou)
        original_homography_list.append(utils.to_numpy(orig_homography.data))
        optim_homography_list.append(utils.to_numpy(optim_homography.data))
        gt_homography_list.append(utils.to_numpy(gt_homography.data))
    t1 = time.time()
    orig_iou_list = np.array(orig_iou_list)
    orig_iou_part_list = np.concatenate(orig_iou_list[:, 0])
    orig_iou_whole_list = np.concatenate(orig_iou_list[:, 1])
    print('----- Summary -----')
    print('original IOU part mean:', orig_iou_part_list.mean())
    print('original IOU part median:', np.median(orig_iou_part_list))
    print('original IOU whole mean:', orig_iou_whole_list.mean())
    print('original IOU whole median:', np.median(orig_iou_whole_list))

    optim_iou_list = np.array(optim_iou_list)
    optim_iou_part_list = np.concatenate(optim_iou_list[:, 0])
    optim_iou_whole_list = np.concatenate(optim_iou_list[:, 1])
    print('optimized IOU part mean:', optim_iou_part_list.mean())
    print('optimized IOU part median:', np.median(optim_iou_part_list))
    print('optimized IOU whole mean:', optim_iou_whole_list.mean())
    print('optimized IOU whole median:', np.median(optim_iou_whole_list))
    print('----- -----')
    print('spent {0} seconds for {1} images'.format((t1 - t0), (optim_iou_whole_list.shape[0])))
    print('{0} seconds per single image'.format((t1 - t0) / (optim_iou_whole_list.shape[0])))
    print('----- End -----')


if __name__ == '__main__':
    main()
