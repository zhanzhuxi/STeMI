import argparse
import datetime
import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_logger(log_dir: str) -> None:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    now_time = now.strftime('log_%Y-%m-%d-%H-%M-%S.log')
    fh = logging.FileHandler(str(log_dir / now_time))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--splits', type=str, nargs='+', default=['splits/tvsum_few_shot.yml'])
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--model-dir', type=str, default='models/tvsum')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--nms-thresh', type=float, default=0.4)
    parser.add_argument('--temporal_scales', type=int, default=4)
    parser.add_argument('--spatial_scales', type=int, default=4)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--sample-rate', type=int, default=15)
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)
    parser.add_argument('--lambda-reg', type=float, default=1.0)
    parser.add_argument('--lambda-ctr', type=float, default=1.0)
    parser.add_argument('--lambda_rec_x', type=float, default=1.0)
    parser.add_argument('--lambda_rec_s', type=float, default=1.0)

    return parser


def get_arguments() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args
