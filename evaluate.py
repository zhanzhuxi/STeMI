import logging

import torch

import data_loader
import vs_helper

logger = logging.getLogger()


def evaluate(model, val_loader, nms_thresh, device):
    model.eval()
    stats = data_loader.AverageMeter('fscore', 'diversity')

    with torch.no_grad():
        for test_key, seq, _, cps, n_frames, nfps, picks, user_summary, classes, support_video in val_loader:

            cps = torch.tensor(cps, dtype=torch.int64).to(device)
            n_frames = torch.tensor(n_frames, dtype=torch.int64).to(device)
            nfps = torch.tensor(nfps, dtype=torch.int64).to(device)
            picks = torch.tensor(picks, dtype=torch.int64).to(device)
            user_summary = torch.tensor(user_summary, dtype=torch.float32).to(device)

            support_video["gtscore"] = torch.tensor(support_video["gtscore"], dtype=torch.float32).to(device)
            support_video["change_points"] = torch.tensor(support_video["change_points"], dtype=torch.int64).to(device)
            support_video["n_frames"] = torch.tensor(support_video["n_frames"], dtype=torch.int64).to(device)
            support_video["n_frame_per_seg"] = torch.tensor(support_video["n_frame_per_seg"], dtype=torch.int64).to(device)
            support_video["picks"] = torch.tensor(support_video["picks"], dtype=torch.int64).to(device)
            support_keyshot_summ, _ = vs_helper.get_keyshot_summ(support_video["gtscore"],
                                                                support_video["change_points"],
                                                                support_video["n_frames"],
                                                                support_video["n_frame_per_seg"],
                                                                support_video["picks"])
            support_target = vs_helper.downsample_summ(support_keyshot_summ)
            selected_indices = torch.where(support_target == 1)[0]
            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)
            support_seq = torch.from_numpy(support_video["features"]).unsqueeze(0).to(device)

            pred_cls, pred_bboxes = model.predict(seq_torch, support_seq, selected_indices)

            pred_bboxes = torch.clamp(pred_bboxes, 0, seq_len).round().to(torch.int32)
            pred_cls, pred_bboxes = vs_helper.nms(pred_cls, pred_bboxes, nms_thresh)
            pred_summ, pred_score_upsampled = vs_helper.bbox2summary(seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)
            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            fscore = vs_helper.get_summ_f1score(pred_summ, user_summary, eval_metric)
            pred_summ = vs_helper.downsample_summ(pred_summ)
            pred_summ = pred_summ.cpu()
            diversity = vs_helper.get_summ_diversity(pred_summ, seq)

            stats.update(fscore=fscore, diversity=diversity)

    return stats.fscore, stats.diversity

