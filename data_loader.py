import logging
import random
from os import PathLike
from pathlib import Path
from typing import Any, List, Dict

import h5py
import numpy as np
import yaml

logger = logging.getLogger()


class VideoDataset(object):
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.datasets = self.get_datasets(keys)

    def __getitem__(self, index):
        key = self.keys[index]
        video_path = Path(key)
        dataset_name = str(video_path.parent)
        video_name = video_path.name
        video_file = self.datasets[dataset_name][video_name]

        seq = video_file['features'][...].astype(np.float32)
        gtscore = video_file['gtscore'][...].astype(np.float32)
        cps = video_file['change_points'][...].astype(np.int32)
        n_frames = video_file['n_frames'][...].astype(np.int32)
        nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)
        classes = video_file['classes'][...].astype(np.int32)
        user_summary = None
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)

        gtscore -= gtscore.min()
        gtscore /= gtscore.max()

        support_candidate = []

        for single_video_name in self.datasets[dataset_name]:
            support_video_file = self.datasets[dataset_name][single_video_name]
            support_classes = support_video_file['classes'][...].astype(np.int32)
            if single_video_name != video_name and support_classes == classes:
                support_candidate.append(support_video_file)

        random_index = random.randint(0, len(support_candidate) - 1)
        random_element = support_candidate[random_index]
        supp_seq = random_element['features'][...].astype(np.float32)
        supp_gtscore = random_element['gtscore'][...].astype(np.float32)
        supp_cps = random_element['change_points'][...].astype(np.int32)
        supp_n_frames = random_element['n_frames'][...].astype(np.int32)
        supp_nfps = random_element['n_frame_per_seg'][...].astype(np.int32)
        supp_picks = random_element['picks'][...].astype(np.int32)
        supp_classes = random_element['classes'][...].astype(np.int32)
        supp_user_summary = None
        if 'user_summary' in random_element:
            supp_user_summary = random_element['user_summary'][...].astype(np.float32)

        supp_gtscore -= supp_gtscore.min()
        supp_gtscore /= supp_gtscore.max()

        support_video = {"features": supp_seq,
                         "gtscore": supp_gtscore,
                         "change_points": supp_cps,
                         "n_frames": supp_n_frames,
                         "n_frame_per_seg": supp_nfps,
                         "picks": supp_picks,
                         "classes": supp_classes,
                         "user_summary": supp_user_summary}

        return key, seq, gtscore, cps, n_frames, nfps, picks, user_summary, classes, support_video

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def get_datasets(keys: List[str]) -> Dict[str, h5py.File]:
        dataset_paths = {str(Path(key).parent) for key in keys}
        datasets = {path: h5py.File(path, 'r') for path in dataset_paths}
        return datasets


class DataLoader(object):
    def __init__(self, dataset: VideoDataset, shuffle: bool):
        self.dataset = dataset
        self.shuffle = shuffle
        self.data_idx = list(range(len(self.dataset)))

    def __iter__(self):
        self.iter_idx = 0
        if self.shuffle:
            random.shuffle(self.data_idx)
        return self

    def __next__(self):
        if self.iter_idx == len(self.dataset):
            raise StopIteration
        curr_idx = self.data_idx[self.iter_idx]
        batch = self.dataset[curr_idx]
        self.iter_idx += 1
        return batch


class AverageMeter(object):
    def __init__(self, *keys: str):
        self.totals = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self._check_attr(key)
            self.totals[key] += value
            self.counts[key] += 1

    def __getattr__(self, attr: str) -> float:
        self._check_attr(attr)
        total = self.totals[attr]
        count = self.counts[attr]
        return total / count if count else 0.0

    def _check_attr(self, attr: str) -> None:
        assert attr in self.totals and attr in self.counts


def get_ckpt_dir(model_dir: PathLike) -> Path:
    return Path(model_dir) / 'checkpoint'


def get_ckpt_path(model_dir: PathLike, split_path: PathLike, split_index: int) -> Path:
    split_path = Path(split_path)
    return get_ckpt_dir(model_dir) / f'{split_path.name}.{split_index}.pt'


def load_yaml(path: PathLike) -> Any:
    with open(path) as f:
        obj = yaml.safe_load(f)
    return obj


def dump_yaml(obj: Any, path: PathLike) -> None:
    with open(path, 'w') as f:
        yaml.dump(obj, f)
