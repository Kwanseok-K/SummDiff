# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from collections import OrderedDict

from networks.mlp import SimpleMLP
from networks.summ_diff.summ_diff import build_model

from model.utils.evaluation_metrics import evaluate_summary
from model.utils.generate_summary import generate_summary, get_gt
from model.utils.evaluate_map import generate_mrsum_seg_scores, top50_summary, top15_summary

from copy import deepcopy
import random

import warnings
warnings.filterwarnings("ignore")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class Solver(object):
    def __init__(self, config=None, train_loader=None, val_loader=None, test_loader=None, ckpt_path=None):

        self.model, self.optimizer, self.scheduler = None, None, None

        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.MSELoss(reduction='none').to(self.config.device)
        self.ckpt_path = ckpt_path

        if config.p_uncond > 0:
            self.null_video = np.load('dataset/null_video.npy')

    def build(self):
        """ Define your own summarization model here """
        if self.config.model == 'MLP':
            self.model = SimpleMLP(1024, [1024], 1)
            self.model.to(self.config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)

        elif self.config.model == 'SummDiff':
            self.model, self.criterion = build_model(self.config)
            self.model.to(self.config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        else:
            print("Wrong model")
            exit()

        if self.ckpt_path is not None:
            print("Loading Model: ", self.ckpt_path)
            self.model.load_state_dict(torch.load(self.ckpt_path))
            self.model.to(self.config.device)

    def train(self):
        best_f1score = -1.0
        best_map50 = -1.0
        best_map15 = -1.0
        best_srho = -1.0
        best_ktau = -1.0
        best_f1score_epoch = 0
        best_map50_epoch = 0
        best_map15_epoch = 0
        best_srho_epoch = 0
        best_ktau_epoch = 0

        if self.config.ema == True:
            self.ema_model = deepcopy(self.model).to(self.config.device)
            requires_grad(self.ema_model, False)
            update_ema(self.ema_model, self.model, 0)
            self.ema_model.eval()

        if self.config.dataset == 'summe':
            torch.manual_seed(204)
            random.seed(204)
            np.random.seed(204)
            torch.cuda.manual_seed(204)
            torch.cuda.manual_seed_all(204)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False

        for epoch_i in range(self.config.epochs):
            print("[Epoch: {0:6}]".format(str(epoch_i) + "/" + str(self.config.epochs)))
            self.model.train()

            loss_history = []
            num_batches = int(len(self.train_loader))
            iterator = iter(self.train_loader)

            for batch_idx in tqdm(range(num_batches)):

                self.optimizer.zero_grad()
                data = next(iterator)

                frame_features = data['features'].to(self.config.device)
                gtscore = data['gtscore'].to(self.config.device)
                mask = data['mask'].to(self.config.device)
                n_frames = data['n_frames']
                gt_summary = data['gt_summary']

                if self.config.clamp:
                    gtscore = torch.clamp(gtscore, 0.05, 0.95)

                if self.config.model == 'SummDiff':
                    if self.config.p_uncond > 0:
                        is_null = torch.rand(frame_features.shape[0]).to(self.config.device) < self.config.p_uncond
                        frame_features[is_null] = torch.tensor(self.null_video).to(self.config.device)

                    score, weights = self.model(gtscore, frame_features, mask, n_frames)
                    loss = self.criterion(score, gtscore, gt_summary, mask).mean()
                else:
                    score, attn_weights = self.model(frame_features, mask=mask)
                    loss = self.criterion(score[mask], gtscore[mask]).mean()

                loss.backward()
                loss_history.append(loss.item())
                self.optimizer.step()
                if self.config.ema == True:
                    update_ema(self.ema_model, self.model, self.config.ema_decay)

                if self.config.individual == True and batch_idx + 1 != num_batches:
                    val_f1score, val_map50, val_map15, val_kTau, val_sRho = self.evaluate(dataloader=self.val_loader)
                    if best_f1score <= val_f1score:
                        best_f1score = val_f1score
                        best_f1score_epoch = epoch_i
                        f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'best_f1.pkl')
                        torch.save(self.model.state_dict(), f1_save_ckpt_path)
                        if self.config.ema == True:
                            torch.save(self.ema_model.state_dict(), f1_save_ckpt_path.split('.')[0] + '_ema.pkl')
                    if best_map50 <= val_map50:
                        best_map50 = val_map50
                        best_map50_epoch = epoch_i
                        map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'best_map50.pkl')
                        torch.save(self.model.state_dict(), map50_save_ckpt_path)
                        if self.config.ema == True:
                            torch.save(self.ema_model.state_dict(), map50_save_ckpt_path.split('.')[0] + '_ema.pkl')
                    if best_map15 <= val_map15:
                        best_map15 = val_map15
                        best_map15_epoch = epoch_i
                        map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'best_map15.pkl')
                        torch.save(self.model.state_dict(), map15_save_ckpt_path)
                        if self.config.ema == True:
                            torch.save(self.ema_model.state_dict(), map15_save_ckpt_path.split('.')[0] + '_ema.pkl')
                    if best_srho <= val_sRho:
                        best_srho = val_sRho
                        best_srho_epoch = epoch_i
                        srho_save_ckpt_path = os.path.join(self.config.best_srho_save_dir, f'best_srho.pkl')
                        torch.save(self.model.state_dict(), srho_save_ckpt_path)
                        if self.config.ema == True:
                            torch.save(self.ema_model.state_dict(), srho_save_ckpt_path.split('.')[0] + '_ema.pkl')
                    if best_ktau <= val_kTau:
                        best_ktau = val_kTau
                        best_ktau_epoch = epoch_i
                        ktau_save_ckpt_path = os.path.join(self.config.best_ktau_save_dir, f'best_ktau.pkl')
                        torch.save(self.model.state_dict(), ktau_save_ckpt_path)
                        if self.config.ema == True:
                            torch.save(self.ema_model.state_dict(), ktau_save_ckpt_path.split('.')[0] + '_ema.pkl')

            loss = np.mean(np.array(loss_history))
            print(f"train epoch loss: {loss}", flush=True)

            val_f1score, val_map50, val_map15, val_kTau, val_sRho = self.evaluate(dataloader=self.val_loader)

            if best_f1score <= val_f1score:
                best_f1score = val_f1score
                best_f1score_epoch = epoch_i
                f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'best_f1.pkl')
                if self.config.ema == True:
                    torch.save(self.ema_model.state_dict(), f1_save_ckpt_path.split('.')[0] + '_ema.pkl')
                torch.save(self.model.state_dict(), f1_save_ckpt_path)

            if best_map50 <= val_map50:
                best_map50 = val_map50
                best_map50_epoch = epoch_i
                map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'best_map50.pkl')
                if self.config.ema == True:
                    torch.save(self.ema_model.state_dict(), map50_save_ckpt_path.split('.')[0] + '_ema.pkl')
                torch.save(self.model.state_dict(), map50_save_ckpt_path)

            if best_map15 <= val_map15:
                best_map15 = val_map15
                best_map15_epoch = epoch_i
                map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'best_map15.pkl')
                if self.config.ema == True:
                    torch.save(self.ema_model.state_dict(), map15_save_ckpt_path.split('.')[0] + '_ema.pkl')
                torch.save(self.model.state_dict(), map15_save_ckpt_path)

            if best_srho <= val_sRho:
                best_srho = val_sRho
                best_srho_epoch = epoch_i
                srho_save_ckpt_path = os.path.join(self.config.best_srho_save_dir, f'best_srho.pkl')
                torch.save(self.model.state_dict(), srho_save_ckpt_path)
                if self.config.ema == True:
                    torch.save(self.ema_model.state_dict(), srho_save_ckpt_path.split('.')[0] + '_ema.pkl')

            if best_ktau <= val_kTau:
                best_ktau = val_kTau
                best_ktau_epoch = epoch_i
                ktau_save_ckpt_path = os.path.join(self.config.best_ktau_save_dir, f'best_ktau.pkl')
                torch.save(self.model.state_dict(), ktau_save_ckpt_path)
                if self.config.ema == True:
                    torch.save(self.ema_model.state_dict(), ktau_save_ckpt_path.split('.')[0] + '_ema.pkl')

            if self.scheduler is not None and self.config.dataset == 'mrhisum':
                self.scheduler.step()

            print("   [Epoch {0}] Train loss: {1:.05f}".format(epoch_i, loss))
            print('    VAL  F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(val_f1score, val_map50, val_map15))
            print('    VAL  KTau {0:0.5} | SRho {1:0.5}'.format(val_kTau, val_sRho))

        print('   Best Val F1 score {0:0.5} @ epoch{1}'.format(best_f1score, best_f1score_epoch))
        print('   Best Val MAP-50   {0:0.5} @ epoch{1}'.format(best_map50, best_map50_epoch))
        print('   Best Val MAP-15   {0:0.5} @ epoch{1}'.format(best_map15, best_map15_epoch))
        print('   Best Val SRho     {0:0.5} @ epoch{1}'.format(best_srho, best_srho_epoch))
        print('   Best Val KTau     {0:0.5} @ epoch{1}'.format(best_ktau, best_ktau_epoch))

        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write('   Best Val F1 score {0:0.5} @ epoch{1}\n'.format(best_f1score, best_f1score_epoch))
        f.write('   Best Val MAP-50   {0:0.5} @ epoch{1}\n'.format(best_map50, best_map50_epoch))
        f.write('   Best Val MAP-15   {0:0.5} @ epoch{1}\n\n'.format(best_map15, best_map15_epoch))
        f.flush()
        f.close()

        if self.config.train_val == False:
            return f1_save_ckpt_path, map50_save_ckpt_path, map15_save_ckpt_path, srho_save_ckpt_path, ktau_save_ckpt_path, best_f1score, best_map50, best_map15, best_srho, best_ktau
        else:
            return f1_save_ckpt_path, map50_save_ckpt_path, map15_save_ckpt_path, srho_save_ckpt_path, ktau_save_ckpt_path

    def evaluate(self, dataloader=None, test=False):
        """ Evaluate the model on a given dataloader. """
        self.model.eval()

        fscore_history = []
        kTau_history = []
        sRho_history = []
        map50_history = []
        map15_history = []
        WSE_history = []
        CIS_history = []
        WIR_history = []
        IR_history = []

        if self.config.dataset == 'tvsum':
            user_scores = get_gt()

        dataloader = iter(dataloader)

        if self.config.dataset == 'summe':
            torch.manual_seed(204)
            random.seed(204)
            np.random.seed(204)
            torch.cuda.manual_seed(204)
            torch.cuda.manual_seed_all(204)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False

        for data in dataloader:
            frame_features = data['features'].to(self.config.device)
            gtscore = data['gtscore'].to(self.config.device)

            if len(frame_features.shape) == 2:
                frame_features = frame_features.unsqueeze(0)
            if len(gtscore.shape) == 1:
                gtscore = gtscore.unsqueeze(0)

            mask = None
            if 'mask' in data:
                mask = data['mask'].to(self.config.device)

            with torch.no_grad():
                if self.config.model == 'SummDiff':
                    if mask is None:
                        mask = torch.arange(frame_features.size(1))[None, :] < data['n_frames'][:, None]
                        mask = mask.to(self.config.device)
                    if self.config.p_uncond > 0:
                        frame_features = torch.cat([frame_features, torch.tensor(self.null_video).to(self.config.device).unsqueeze(0).repeat(frame_features.shape[1], 1).unsqueeze(0)], dim=0)
                        mask = torch.cat([mask, mask], dim=0)
                    if self.config.ema == True:
                        score, _ = self.ema_model(None, frame_features, mask, data['n_frames'])
                    else:
                        score, _ = self.model(None, frame_features, mask, data['n_frames'])
                    score = score['pred_scores']
                    score = score[:int(data['n_frames'][0])]
                    gtscore = gtscore.squeeze()[:int(data['n_frames'][0])]
                else:
                    score, attn_weights = self.model(frame_features, mask=mask)
                    score = score.squeeze()[:int(data['n_frames'][0])]
                    gtscore = gtscore.squeeze()[:int(data['n_frames'][0])]

            # Summarization metric
            score = score.squeeze().cpu()
            gt_summary = data['gt_summary'][0]
            cps = data['change_points'][0]
            n_frames = data['n_frames']
            nfps = data['n_frame_per_seg'][0].tolist()
            picks = data['picks'][0]

            machine_summary = generate_summary(score, cps, n_frames, nfps, picks)
            if self.config.dataset == 'mrhisum':
                f_score, kTau, sRho = evaluate_summary(machine_summary, gt_summary, score, gtscore=gtscore,
                                                       dataset=self.config.dataset,
                                                       eval_method='avg')
            elif self.config.dataset == 'summe':
                user_summary = data['user_summary'][0].squeeze()
                if torch.is_tensor(user_summary):
                    user_summary = user_summary.cpu().numpy()
                f_score, kTau, sRho = evaluate_summary(machine_summary, user_summary, score, gtscore=gtscore,
                                                       dataset=self.config.dataset,
                                                       eval_method='max')
            elif self.config.dataset == 'tvsum':
                user_summary = data['user_summary'][0].squeeze()
                if torch.is_tensor(user_summary):
                    user_summary = user_summary.cpu().numpy()
                video_name = data['video_name'][0]
                vid = int(video_name.split('_')[-1])
                f_score, kTau, sRho = evaluate_summary(machine_summary, user_summary, score,
                                                       gtscore=np.array(user_scores[vid - 1]),
                                                       dataset=self.config.dataset,
                                                       eval_method='avg')

            fscore_history.append(f_score)
            kTau_history.append(kTau)
            sRho_history.append(sRho)

            if test and self.config.dataset == 'mrhisum':
                from model.utils.evaluation_metrics import evaluate_knapsack_opt
                WSE, CIS, WIR, IR = evaluate_knapsack_opt(score, gtscore, gt_summary, cps, n_frames, nfps, picks)
                WSE_history.append(WSE)
                CIS_history.append(CIS)
                WIR_history.append(WIR)
                IR_history.append(IR)

            # Highlight Detection Metric
            gt_seg_score = generate_mrsum_seg_scores(gtscore.squeeze(0), uniform_clip=5)
            gt_top50_summary = top50_summary(gt_seg_score)
            gt_top15_summary = top15_summary(gt_seg_score)

            highlight_seg_machine_score = generate_mrsum_seg_scores(score, uniform_clip=5)
            highlight_seg_machine_score = torch.exp(highlight_seg_machine_score) / (torch.exp(highlight_seg_machine_score).sum() + 1e-7)

            clone_machine_summary = highlight_seg_machine_score.clone().detach().cpu().numpy()
            aP50 = average_precision_score(gt_top50_summary, clone_machine_summary)
            aP15 = average_precision_score(gt_top15_summary, clone_machine_summary)
            map50_history.append(aP50)
            map15_history.append(aP15)

        final_f_score = np.mean(fscore_history)
        final_kTau = np.mean(kTau_history)
        final_sRho = np.mean(sRho_history)
        final_map50 = np.mean(map50_history)
        final_map15 = np.mean(map15_history)

        if test and self.config.dataset == 'mrhisum':
            final_WSE = np.mean(WSE_history)
            final_CIS = np.mean(CIS_history)
            final_WIR = np.mean(WIR_history)
            final_IR = np.mean(IR_history)
            return final_f_score, final_map50, final_map15, final_kTau, final_sRho, final_WSE, final_CIS, final_WIR, final_IR
        else:
            return final_f_score, final_map50, final_map15, final_kTau, final_sRho

    def test(self, ckpt_path):
        if ckpt_path is not None:
            print("Testing Model: ", ckpt_path)
            print("Device: ", self.config.device)
            self.model.load_state_dict(torch.load(ckpt_path))
            if self.config.ema == True:
                self.ema_model = deepcopy(self.model).to(self.config.device)

        if self.config.dataset == 'mrhisum':
            test_fscore, test_map50, test_map15, test_kTau, test_sRho, test_WSE, test_CIS, test_WIR, test_IR = self.evaluate(dataloader=self.test_loader, test=True)
        else:
            test_fscore, test_map50, test_map15, test_kTau, test_sRho = self.evaluate(dataloader=self.test_loader, test=True)

        print("------------------------------------------------------")
        print(f"   TEST RESULT on {ckpt_path}: ")
        print('   TEST MRSum F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(test_fscore, test_map50, test_map15))
        print('   TEST MRSum KTau {0:0.5} | SRho {1:0.5}'.format(test_kTau, test_sRho))
        if self.config.dataset == 'mrhisum':
            print('   TEST MRSum WSE {0:0.5} | CIS {1:0.5} | WIR {2:0.5} | IR {3:0.5}'.format(test_WSE, test_CIS, test_WIR, test_IR))
        print("------------------------------------------------------")

        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write("Testing on Model " + ckpt_path + '\n')
        f.write('Test F-score ' + str(test_fscore) + '\n')
        f.write('Test MAP50   ' + str(test_map50) + '\n')
        f.write('Test MAP15   ' + str(test_map15) + '\n\n')
        if self.config.dataset == 'mrhisum':
            f.write('Test WSE ' + str(test_WSE) + '\n')
            f.write('Test CIS ' + str(test_CIS) + '\n')
            f.write('Test WIR ' + str(test_WIR) + '\n')
            f.write('Test IR ' + str(test_IR) + '\n\n')
        f.flush()

        if self.config.dataset != 'mrhisum':
            return test_fscore, test_map50, test_map15, test_kTau, test_sRho
