import random
from pathlib import Path

import logging

import torch

import vs_helper
import data_loader
import init
from model import STeMI
from evaluate import evaluate
from losses import calc_ctr_loss, calc_cls_loss, calc_loc_loss, reconstruction_loss

logger = logging.getLogger()


def train(args, split, save_path):
    model = STeMI(num_feature=args.num_feature, num_hidden=args.num_hidden, num_head=args.num_head,
                    temporal_scales=args.temporal_scales, spatial_scales=args.spatial_scales)
    model = model.to(args.device)
    model.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    max_val_fscore = -1
    train_set = data_loader.VideoDataset(split['train_keys'])
    train_loader = data_loader.DataLoader(train_set, shuffle=True)
    val_set = data_loader.VideoDataset(split['test_keys'])
    val_loader = data_loader.DataLoader(val_set, shuffle=False)

    for epoch in range(args.max_epoch):
        random.seed(epoch + args.seed)
        model.train()
        for _, seq, gtscore, change_points, n_frames, nfps, picks, _, _, support_video in train_loader:

            gtscore = torch.tensor(gtscore, dtype=torch.float32).to(args.device)
            change_points = torch.tensor(change_points, dtype=torch.int64).to(args.device)
            n_frames = torch.tensor(n_frames, dtype=torch.int64).to(args.device)
            nfps = torch.tensor(nfps, dtype=torch.int64).to(args.device)
            picks = torch.tensor(picks, dtype=torch.int64).to(args.device)
            keyshot_summ, _ = vs_helper.get_keyshot_summ(gtscore, change_points, n_frames, nfps, picks)
            target = vs_helper.downsample_summ(keyshot_summ)
            support_video["gtscore"] = torch.tensor(support_video["gtscore"], dtype=torch.float32).to(args.device)
            support_video["change_points"] = torch.tensor(support_video["change_points"], dtype=torch.int64).to(
                args.device)
            support_video["n_frames"] = torch.tensor(support_video["n_frames"], dtype=torch.int64).to(args.device)
            support_video["n_frame_per_seg"] = torch.tensor(support_video["n_frame_per_seg"], dtype=torch.int64).to(
                args.device)
            support_video["picks"] = torch.tensor(support_video["picks"], dtype=torch.int64).to(args.device)
            support_keyshot_summ, _ = vs_helper.get_keyshot_summ(support_video["gtscore"],
                                                                support_video["change_points"],
                                                                support_video["n_frames"],
                                                                support_video["n_frame_per_seg"],
                                                                support_video["picks"])
            support_target = vs_helper.downsample_summ(support_keyshot_summ)
            selected_indices = torch.where(support_target == 1)[0]
            if not target.any():
                continue

            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)
            support_seq = torch.tensor(support_video["features"], dtype=torch.float32).unsqueeze(0).to(args.device)
            cls_label = target
            loc_label = vs_helper.get_loc_label(target)
            ctr_label = vs_helper.get_ctr_label(target, loc_label)
            pred_cls, pred_loc, pred_ctr, recons_x, recons_support = model(seq, support_seq, selected_indices)
            cls_label = cls_label.float()
            loc_label = loc_label.float()
            ctr_label = ctr_label.float()
            cls_loss = calc_cls_loss(pred_cls, cls_label)
            loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label)
            ctr_loss = calc_ctr_loss(pred_ctr, ctr_label, cls_label)
            rec_s = reconstruction_loss(recons_support, support_seq)
            rec_x = reconstruction_loss(recons_x, seq)
            loss = cls_loss + args.lambda_reg * loc_loss + args.lambda_ctr * ctr_loss + args.lambda_rec_x * rec_x + args.lambda_rec_s * rec_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_fscore, _ = evaluate(model, val_loader, args.nms_thresh, args.device)

        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))

        logger.info(f'Epoch: {epoch}/{args.max_epoch}\t'
                    f'F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f}\t')

    return max_val_fscore


def main():
    args = init.get_arguments()
    init.init_logger(args.model_dir)
    init.set_random_seed(args.seed)

    logger.info(vars(args))

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_loader.get_ckpt_dir(model_dir).mkdir(parents=True, exist_ok=True)

    data_loader.dump_yaml(vars(args), model_dir / 'args.yml')

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_loader.load_yaml(split_path)

        results = {}
        stats = data_loader.AverageMeter('fscore')

        for split_idx, split in enumerate(splits):
            logger.info(f'Start training on {split_path.stem}: split {split_idx}')
            ckpt_path = data_loader.get_ckpt_path(model_dir, split_path, split_idx)
            fscore = train(args, split, ckpt_path)
            stats.update(fscore=fscore)
            results[f'split{split_idx}'] = float(fscore)

        results['mean'] = float(stats.fscore)
        data_loader.dump_yaml(results, model_dir / f'{split_path.stem}.yml')

        logger.info(f'Training done on {split_path.stem}. F-score: {stats.fscore:.4f}')


if __name__ == '__main__':
    main()
