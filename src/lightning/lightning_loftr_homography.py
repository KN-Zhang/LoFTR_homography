
from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch, time
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.loftr import LoFTR
from src.loftr.utils.supervision import compute_supervision_fine, compute_supervision_coarse_homography
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    compute_MA,
    aggregate_metrics_my
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler

import wandb


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, total_epochs, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = LoFTR(config=_config['loftr'])
        self.loss = LoFTRLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir
        self.total_epochs = total_epochs
        self.runtime = []
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        self.config.TRAINER.SCHEDULER = 'CosineAnnealing'
        self.config.TRAINER.COSA_TMAX = self.total_epochs
        return [optimizer], [scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):
        batch = self.dict_remap(batch)
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse_homography(batch, self.config)
        
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)
        
        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)
            
        with self.profiler.profile("Compute losses"):
            self.loss(batch)
    
    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names
    
    def _compute_metrics_my(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_MA(batch)  # compute epi_errs for each match
            bs = batch['image0'].size(0)
            metrics = {"ACE": batch["ACE"]}
            ret_dict = {'metrics': metrics}
        return ret_dict
        
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            wandb.log(batch['loss_scalars'])
        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        pass


    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        ret_dict = self._compute_metrics_my(batch)
        
        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
        }
        
    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
            val_metrics_4tb = aggregate_metrics_my(metrics)
            for thr in [3, 5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])
            

            if self.trainer.global_rank == 0:
                wandb.log(val_metrics_4tb)

        for thr in [3, 5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFTR"):
            batch = self.dict_remap(batch)
            start_time = time.time()
            self.matcher(batch)
            self.runtime.append(time.time()-start_time)

        ### plot some matching results
        # if batch_idx == 28:
        #     error = kornia.geometry.homography.oneway_transfer_error(batch['mkpts0_f'], batch['mkpts1_f'], batch['H_s2t']).squeeze(0).cpu()
        #     inliers = np.random.permutation(np.arange(0, len(error)))[:50] ##error<14
        #     # warped_img1 = KGT.warp_perspective(im_A, torch.from_numpy(H_pred).float().unsqueeze(0), (w2, h2), align_corners=True) #warp img2
        #     # make_matching_figure(im_A, im_B, warped_img1, im_w_A, pos_a[inliers], pos_b[inliers], error[inliers], path='match.png', kpts0=None, kpts1=None)
        #     plt.clf()
        #     kpts0, kpts1 = batch['mkpts0_f'].cpu(), batch['mkpts1_f'].cpu()
        #     draw_LAF_matches(
        #         KF.laf_from_center_scale_ori(
        #             kpts0[inliers].view(1, -1, 2),
        #             torch.ones(kpts0[inliers].shape[0]).view(1, -1, 1, 1),
        #             torch.ones(kpts0[inliers].shape[0]).view(1, -1, 1),
        #         ),
        #         KF.laf_from_center_scale_ori(
        #             kpts1[inliers].view(1, -1, 2),
        #             torch.ones(kpts1[inliers].shape[0]).view(1, -1, 1, 1),
        #             torch.ones(kpts1[inliers].shape[0]).view(1, -1, 1),
        #         ),
        #         torch.arange(kpts0[inliers].shape[0]).view(-1, 1).repeat(1, 2),
        #         kornia.tensor_to_image(batch['image0_rgb'].cpu()),
        #         kornia.tensor_to_image(batch['image1_rgb'].cpu()),
        #         error[inliers]<3,
        #         draw_dict={"inlier_color": (0.2, 1, 0.4), "tentative_color": (1, 0, 0), "feature_color": (0.2, 0.5, 1), "vertical": False},
        #     )
        #     plt.axis('off')
        #     plt.savefig(f'/home/kz23d522/data/iclr25/plot/ir/loftr/{batch_idx}.png')            

        ret_dict = self._compute_metrics_my(batch)

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics_my(metrics)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)

    def dict_remap(self, old_dict):
        new_dict = {
            "image0": old_dict["im_A"],
            "image1": old_dict["im_B"],
            "H_s2t": old_dict["H_s2t"],
            "warped_img1": old_dict["warped_img1"],
            "dataset_name": old_dict["dataset_name"],
            "im_A_path": old_dict["im_A_path"],
            "im_B_path": old_dict["im_B_path"],
            "original_h": old_dict["original_h"],
        }

        if old_dict.get("mask") is not None:
            new_dict["mask"] = old_dict["mask"]

        old_dict.clear()
        old_dict.update(new_dict)

        return old_dict
