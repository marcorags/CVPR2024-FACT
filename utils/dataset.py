#!/usr/bin/python3

from pathlib import Path
import numpy as np
import os
import torch
from ..home import get_project_base
from yacs.config import CfgNode
from .utils import shrink_frame_label

BASE = get_project_base()

def load_feature(feature_dir, video, transpose):
    file_name = os.path.join(feature_dir, video+'.npy')
    feature = np.load(file_name)

    if transpose:
        feature = feature.T
    if feature.dtype != np.float32:
        feature = feature.astype(np.float32)
    
    return feature #[::sample_rate]

def load_action_mapping(map_fname, sep=" "):
    label2index = dict()
    index2label = dict()
    with open(map_fname, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            tokens = line.split(sep)
            l = sep.join(tokens[1:])
            i = int(tokens[0])
            label2index[l] = i
            index2label[i] = l

    return label2index, index2label

class Dataset(object):
    """
    self.features[video]: the feature array of the given video (frames x dimension)
    self.input_dimension: dimension of video features
    self.n_classes: number of classes
    """

    def __init__(self, video_list, nclasses, load_video_func, bg_class):
        """
        """

        self.video_list = video_list
        self.load_video = load_video_func

        # store dataset information
        self.nclasses = nclasses
        self.bg_class = bg_class
        self.data = {}
        self.data[video_list[0]] = load_video_func(video_list[0])
        self.input_dimension = self.data[video_list[0]][0].shape[1] 
    
    def __str__(self):
        string = "< Dataset %d videos, %d feat-size, %d classes >"
        string = string % (len(self.video_list), self.input_dimension, self.nclasses)
        return string
    
    def __repr__(self):
        return str(self)

    def get_vnames(self):
        return self.video_list[:]

    def __getitem__(self, video):
        if video not in self.video_list:
            raise ValueError(video)

        if video not in self.data:
            self.data[video] = self.load_video(video)

        return self.data[video]

    def __len__(self):
        return len(self.video_list)


class DataLoader():

    def __init__(self, dataset: Dataset, batch_size, shuffle=False):

        self.num_video = len(dataset)
        self.dataset = dataset
        self.videos = list(dataset.get_vnames())
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.num_batch = int(np.ceil(self.num_video/self.batch_size))

        self.selector = list(range(self.num_video))
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.selector)
            # self.selector = self.selector.tolist()

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_video:
            if self.shuffle:
                np.random.shuffle(self.selector)
                # self.selector = self.selector.tolist()
            self.index = 0
            raise StopIteration

        else:
            video_idx = self.selector[self.index : self.index+self.batch_size]
            if len(video_idx) < self.batch_size:
                video_idx = video_idx + self.selector[:self.batch_size-len(video_idx)]
            videos = [self.videos[i] for i in video_idx]
            self.index += self.batch_size

            batch_sequence = []
            batch_train_label = []
            batch_eval_label = []
            for vname in videos:
                sequence, train_label, eval_label = self.dataset[vname]
                batch_sequence.append(torch.from_numpy(sequence))
                batch_train_label.append(torch.LongTensor(train_label))
                batch_eval_label.append(eval_label)


            return videos, batch_sequence, batch_train_label, batch_eval_label


#------------------------------------------------------------------
#------------------------------------------------------------------

def create_dataset(cfg: CfgNode):

    if cfg.dataset == "fsjump":
        dataset_path = Path(BASE) / 'data' / 'fsjump'
        # dataset_path = Path(BASE) / 'CVPR2024-FACT' / 'data' / 'fsjump'
        map_fname = dataset_path / 'mapping.txt'
        feature_path = dataset_path / 'features'
        train_split_fname = dataset_path / 'splits' / 'train.txt'
        test_split_fname = dataset_path / 'splits' / 'test.txt'
        # train_split_fname = BASE + f'data/fsjump/splits/train'
        # test_split_fname = BASE + f'data/fsjump/splits/test'
        # val_split_fname = BASE + f'data/fsjump/splits/val'
        feature_transpose = False
        average_transcript_len = 4.5
        bg_class = [] 
    
    groundTruth_path = os.path.join(dataset_path, 'groundTruth')

    ################################################
    ################################################
    print("Loading Feature from", feature_path)
    print("Loading Label from", groundTruth_path)

    label2index, index2label = load_action_mapping(map_fname)
    nclasses = len(label2index)

    """
    load video interface:
        Input: video name
        Output:
            feature, label_for_training, label_for_evaluation
    """
    def load_video(vname):
        feature = load_feature(feature_path, vname, feature_transpose) # should be T x D or T x D x H x W

        with open(os.path.join(groundTruth_path, vname + '.txt')) as f:
            gt_label = [ label2index[line] for line in f.read().split('\n')[:-1] ]


        if feature.shape[0] != len(gt_label):
            l = min(feature.shape[0], len(gt_label))
            feature = feature[:l]
            gt_label = gt_label[:l]

        # downsample if necessary
        sr = cfg.sr
        if sr > 1:
            feature = feature[::sr]
            gt_label_sampled = shrink_frame_label(gt_label, sr)
        else:
            gt_label_sampled = gt_label

        return feature, gt_label_sampled, gt_label

    
    ################################################
    ################################################
    
    with open(test_split_fname, 'r') as f:
        test_video_list = f.read().split('\n')[0:-1]
    # if cfg.dataset in ['fsjump']: 
    #     test_video_list = [ v[:-4] for v in test_video_list ] 
    test_dataset = Dataset(test_video_list, nclasses, load_video, bg_class)

    if cfg.aux.debug:
        dataset = test_dataset
    else:
        with open(train_split_fname, 'r') as f:
            video_list = f.read().split('\n')[0:-1]
        # if cfg.dataset in ['fsjump']: 
            # video_list = [ v[:-4] for v in video_list ] 
        dataset = Dataset(video_list, nclasses, load_video, bg_class)
        
    dataset.average_transcript_len = average_transcript_len
    dataset.label2index = label2index
    dataset.index2label = index2label
    test_dataset.average_transcript_len = average_transcript_len
    test_dataset.label2index = label2index
    test_dataset.index2label = index2label

    return dataset, test_dataset