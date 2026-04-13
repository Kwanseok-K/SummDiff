import torch
from torch.utils.data import Dataset
import numpy as np
import random
import logging
import h5py
import json
from torch.nn.utils.rnn import pad_sequence
from model.utils.generate_summary import get_gt
import pickle

logger = logging.getLogger(__name__)

# Default path for the MrHiSum dataset; override via --data_path argument
DEFAULT_MRHISUM_PATH = 'dataset/mrsum_with_features_gtsummary_modified.h5'


class MrSumDataset(Dataset):
    def __init__(self, mode, data_path=None):
        self.dataset = data_path if data_path else DEFAULT_MRHISUM_PATH
        self.split_file = 'dataset/mrsum_split.json'
        self.mode = mode

        self.video_data = h5py.File(self.dataset, 'r')

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())

    def __len__(self):
        self.len = len(self.data[self.mode + '_keys'])
        return self.len

    def __getitem__(self, index):
        video_name = self.data[self.mode + '_keys'][index]
        d = {}
        d['video_name'] = video_name
        d['features'] = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        d['gtscore'] = torch.Tensor(np.array(self.video_data[video_name + '/gtscore']))

        n_frames = d['features'].shape[0]
        cps = np.array(self.video_data[video_name + '/change_points'])
        d['n_frames'] = np.array(n_frames)
        d['picks'] = np.array([i for i in range(n_frames)])
        d['change_points'] = cps
        d['n_frame_per_seg'] = np.array([cp[1] - cp[0] for cp in cps])
        if d['change_points'][-1][1] != n_frames:
            d['n_frame_per_seg'][-1] += 1
        d['gt_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/gt_summary']), axis=0)

        return d


class BatchCollator(object):
    def __call__(self, batch):
        video_name, features, gtscore = [], [], []
        cps, nseg, n_frames, picks, gt_summary = [], [], [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                cps.append(data['change_points'])
                nseg.append(data['n_frame_per_seg'])
                n_frames.append(data['n_frames'])
                picks.append(data['picks'])
                gt_summary.append(data['gt_summary'])
        except:
            print('Error in batch collator')

        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], features)))
        max_len = max(list(map(lambda x: x.shape[0], features)))

        mask = torch.arange(max_len)[None, :] < lengths[:, None]

        frame_feat = pad_sequence(features, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)

        frame_feat = self.pad_to_max_length(frame_feat, max_len)
        gtscore = self.pad_to_max_length(gtscore, max_len)

        batch_data = {
            'video_name': video_name, 'features': frame_feat, 'gtscore': gtscore,
            'mask': mask, 'n_frames': n_frames, 'picks': picks,
            'n_frame_per_seg': nseg, 'change_points': cps, 'gt_summary': gt_summary
        }
        return batch_data

    def pad_to_max_length(self, padded_sequence, max_length):
        if padded_sequence.size(1) < max_length:
            padding_size = max_length - padded_sequence.size(1)
            padding = torch.zeros((padded_sequence.size(0), padding_size, *padded_sequence.size()[2:]), dtype=padded_sequence.dtype)
            padded_sequence = torch.cat([padded_sequence, padding], dim=1)
        return padded_sequence


def load_split(file):
    outputs = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            _, _, train_videos, test_videos = line.split('/')
            train_videos = train_videos.split(',')
            test_videos = test_videos.split(',')
            test_videos[-1] = test_videos[-1].replace('\n', '')
            outputs.append({'train': train_videos, 'test': test_videos})
    return outputs


class SummaryDataset(Dataset):
    def __init__(self, mode, dataset, split, train_val):
        if dataset == 'summe':
            self.dataset = f'dataset/summe/eccv16_dataset_{dataset}_google_pool5.h5'
        if dataset == 'tvsum':
            self.dataset = f'dataset/tvsum/eccv16_dataset_{dataset}_google_pool5.h5'
        if train_val:
            if dataset == 'summe':
                file = f'dataset/{dataset}_val_splits.json'
            else:
                file = f'dataset/{dataset}_val_splits.json'
            with open(file, 'r') as f:
                self.split = json.loads(f.read())[split][mode]
        else:
            file = f'dataset/{dataset}_splits.txt'
            self.split = load_split(file)[split][mode]

        self.mode = mode
        self.video_data = h5py.File(self.dataset, 'r')

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        video_name = self.split[index]
        d = {}
        d['video_name'] = video_name
        d['features'] = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        d['gtscore'] = torch.Tensor(np.array(self.video_data[video_name + '/gtscore']))

        cps = np.array(self.video_data[video_name + '/change_points'])
        d['n_frames'] = np.array(self.video_data[video_name + '/n_frames'])
        d['picks'] = np.array(self.video_data[video_name + '/picks'])
        d['change_points'] = cps
        d['n_frame_per_seg'] = np.array(self.video_data[video_name + '/n_frame_per_seg'])
        d['user_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/user_summary']), axis=0)
        d['gt_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/gtsummary']), axis=0)

        return d


class SummaryDataset_multi(Dataset):
    def __init__(self, mode, dataset, split, train_val):
        self.file = f'dataset/{dataset}/eccv16_dataset_{dataset}_google_pool5.h5'
        if train_val:
            if dataset == 'summe':
                file = f'dataset/{dataset}_val_splits.json'
            else:
                file = f'dataset/{dataset}_val_splits.json'
            with open(file, 'r') as f:
                self.split = json.loads(f.read())[split][mode]
        else:
            file = f'dataset/{dataset}_splits.txt'
            self.split = load_split(file)[split][mode]

        self.dataset = dataset
        self.mode = mode
        self.video_data = h5py.File(self.file, 'r')
        self.user_summary = []
        if dataset == 'tvsum':
            summary = get_gt()
            for vid in self.split:
                vid = int(vid.split('_')[-1])
                for j in range(len(summary[vid - 1])):
                    self.user_summary.append([summary[vid - 1][j], vid])

        if dataset == 'summe':
            for vid in self.split:
                summary = np.array(self.video_data[vid + '/user_summary'])
                vid = vid.split('_')[-1]
                for j in range(len(summary)):
                    self.user_summary.append([summary[j][::15], int(vid)])

    def __len__(self):
        return len(self.user_summary)

    def __getitem__(self, index):
        user_summary, idx = self.user_summary[index]

        video_name = f'video_{idx}'
        d = {}
        d['video_name'] = video_name
        d['features'] = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        d['gtscore'] = torch.Tensor(np.array(user_summary))

        cps = np.array(self.video_data[video_name + '/change_points'])
        d['n_frames'] = np.array(self.video_data[video_name + '/n_frames'])
        d['picks'] = np.array(self.video_data[video_name + '/picks'])
        d['change_points'] = cps
        d['n_frame_per_seg'] = np.array(self.video_data[video_name + '/n_frame_per_seg'])
        d['user_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/user_summary']), axis=0)
        d['gt_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/gtsummary']), axis=0)

        return d


class SummaryBatchCollator(BatchCollator):
    def __init__(self, max_len=-1):
        self.max_len = max_len

    def __call__(self, batch):
        video_name, features, gtscore = [], [], []
        cps, nseg, n_frames, picks, gt_summary, user_summary = [], [], [], [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                cps.append(data['change_points'])
                nseg.append(data['n_frame_per_seg'])
                n_frames.append(data['n_frames'])
                picks.append(data['picks'])
                gt_summary.append(data['gt_summary'])
                user_summary.append(data['user_summary'])
        except:
            print('Error in batch collator')

        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], features)))
        if self.max_len == -1:
            max_len = max(list(map(lambda x: x.shape[0], features)))
        else:
            max_len = self.max_len

        mask = torch.arange(max_len)[None, :] < lengths[:, None]

        frame_feat = pad_sequence(features, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)

        frame_feat = self.pad_to_max_length(frame_feat, max_len)
        gtscore = self.pad_to_max_length(gtscore, max_len)

        batch_data = {
            'video_name': video_name, 'features': frame_feat, 'gtscore': gtscore,
            'mask': mask, 'n_frames': n_frames, 'picks': picks,
            'n_frame_per_seg': nseg, 'change_points': cps,
            'gt_summary': gt_summary, 'user_summary': user_summary
        }
        return batch_data
