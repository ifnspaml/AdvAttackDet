from torch.utils.data import DataLoader

from dataloader.pt_data_loader.specialdatasets import StandardDataset
from dataloader.definitions.labels_file import labels_cityscape_seg
import dataloader.pt_data_loader.mytransforms as tf


def cityscapes_validation(resize_height, resize_width, batch_size, num_workers):
    """A loader that loads images and ground truth for segmentation from the
    cityscapes validation set
    """

    labels = labels_cityscape_seg.getlabels()
    num_classes = len(labels_cityscape_seg.gettrainid2label())

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize((resize_height, resize_width), image_types=('color', )),
        tf.ConvertSegmentation(),
        tf.ConvertDepth(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        # tf.NormalizeZeroMean() first the attack needs to be applied
        tf.AddKeyValue('domain', 'cityscapes_val_seg'),
        tf.AddKeyValue('validation_mask', 'validation_mask_cityscapes'),
        tf.AddKeyValue('validation_clamp', 'validation_clamp_cityscapes'),
        tf.AddKeyValue('purposes', ('segmentation', 'depth')),
        tf.AddKeyValue('num_classes', num_classes)
    ]

    dataset = StandardDataset(
        dataset='cityscapes',
        trainvaltest_split='validation',
        video_mode='mono',
        stereo_mode='mono',
        labels_mode='fromid',
        labels=labels,
        keys_to_load=['color', 'segmentation', 'depth'],
        data_transforms=transforms,
        disable_const_items=True,
        output_filenames=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the cityscapes validation set for segmentation validation",
          flush=True)

    return loader


def cityscapes_seq_validation(resize_height, resize_width, batch_size, num_workers):
    """A loader that loads images and ground truth for segmentation from the
    cityscapes validation set
    """

    labels = labels_cityscape_seg.getlabels()
    num_classes = len(labels_cityscape_seg.gettrainid2label())

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize((resize_height, resize_width), image_types=('color', )),
        tf.ConvertSegmentation(),
        tf.ConvertDepth(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        # tf.NormalizeZeroMean() first the attack needs to be applied
        tf.AddKeyValue('domain', 'cityscapes_val_seg'),
        tf.AddKeyValue('validation_mask', 'validation_mask_cityscapes'),
        tf.AddKeyValue('validation_clamp', 'validation_clamp_cityscapes'),
        tf.AddKeyValue('purposes', ('segmentation', 'depth')),
        tf.AddKeyValue('num_classes', num_classes)
    ]

    dataset = StandardDataset(
        dataset='cityscapes_sequence',
        trainvaltest_split='validation',
        video_mode='video',
        stereo_mode='mono',
        labels_mode='fromid',
        labels=labels,
        keys_to_load=['color', 'segmentation', 'depth'],
        data_transforms=transforms,
        video_frames=(0, 1),
        disable_const_items=False,
        output_filenames=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the cityscapes validation set for segmentation validation",
          flush=True)

    return loader


def kitti_2015_train(resize_height, resize_width, batch_size, num_workers):
    """A loader that loads images and ground truth for segmentation from the
    kitti_2015 train set
    """

    labels = labels_cityscape_seg.getlabels()
    num_classes = len(labels_cityscape_seg.gettrainid2label())

    transforms = [
        tf.CreateScaledImage(True),
        tf.Resize((resize_height, resize_width), image_types=('color', )),
        tf.ConvertSegmentation(),
        tf.ConvertDepth(),
        tf.CreateColoraug(),
        tf.ToTensor(),
        # tf.NormalizeZeroMean() first the attack needs to be applied
        tf.AddKeyValue('domain', 'kitti_2015_val_seg'),
        tf.AddKeyValue('validation_mask', 'validation_mask_kitti_kitti'),
        tf.AddKeyValue('validation_clamp', 'validation_clamp_kitti'),
        tf.AddKeyValue('purposes', ('segmentation', 'depth')),
        tf.AddKeyValue('num_classes', num_classes)
    ]

    dataset = StandardDataset(
        dataset='kitti_2015',
        trainvaltest_split='train',
        video_mode='video',
        stereo_mode='mono',
        labels_mode='fromid',
        labels=labels,
        video_frames=(0, 1),
        keys_to_load=['color', 'segmentation', 'depth'],
        data_transforms=transforms,
        disable_const_items=False,
        output_filenames=True
    )

    loader = DataLoader(
        dataset, batch_size, False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"  - Can use {len(dataset)} images from the kitti_2015 train set for segmentation validation", flush=True)

    return loader
