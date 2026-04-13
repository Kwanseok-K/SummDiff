import argparse

from model.configs import Config, str2bool
from torch.utils.data import DataLoader
from model.mrsum_dataset import MrSumDataset, BatchCollator, SummaryDataset, SummaryBatchCollator, SummaryDataset_multi
from model.solver import Solver
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SummDiff', help='the name of the model')
    parser.add_argument('--epochs', type=int, default=200, help='the number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='the learning rate')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='l2 regularizer')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='the dropout ratio')
    parser.add_argument('--batch_size', type=int, default=256, help='the batch size')
    parser.add_argument('--tag', type=str, default='dev', help='A tag for experiments')
    parser.add_argument('--ckpt_path', type=str, default=None, help='checkpoint path for inference or weight initialization')
    parser.add_argument('--train', type=str2bool, default='true', help='when use Train')
    parser.add_argument('--save_results', type=str2bool, default=False, help='when save results')

    # Data Config
    parser.add_argument("--v_feat_dim", default=1024, type=int, help="video feature dim")
    parser.add_argument("--train_val", type=str2bool, default=False, help="train + val or just train")
    parser.add_argument("--split", type=int, default=0, help="split number")
    parser.add_argument('--dataset', type=str, default='mrhisum', choices=['mrhisum', 'tvsum', 'summe'], help='the name of the dataset')
    parser.add_argument('--individual', type=str2bool, default=False, help='individual or aggregate label training')
    parser.add_argument('--data_path', type=str, default=None, help='path to the MrHiSum h5 dataset file')

    # Model Config
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding")
    parser.add_argument('--enc_layers', default=2, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--num_scores', default=300, type=int, help="Number of query slots")
    parser.add_argument('--input_dropout', default=0.5, type=float, help="Dropout applied in input")
    parser.add_argument("--n_input_proj", type=int, default=3, help="#layers to encoder input")
    parser.add_argument("--temperature", type=float, default=0.07, help="temperature nce contrastive_align_loss")
    parser.add_argument("--K", type=int, default=200, help="Quantization size for scores")
    parser.add_argument("--denoiser", type=str, default='DiT', choices=['DiT', 'Transformer_dec', 'latentmlp'], help="Denoiser model")
    parser.add_argument("--p_uncond", type=float, default=0.0, help="Probability of sampling from unconditional")
    parser.add_argument("--w", type=float, default=0.1, help="weight for unconditional sampling")
    parser.add_argument("--ema", type=str2bool, default=False, help="use EMA for denoiser")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="decay for EMA")
    parser.add_argument("--sigmoid_temp", type=float, default=1.0, help="temperature for sigmoid")
    parser.add_argument("--eps", type=float, default=1e-3, help="eps for logit")
    parser.add_argument("--scores_embed", type=str, default='learned', choices=['learned', 'sinusoidal'], help="score embedding")
    parser.add_argument("--clamp", type=str2bool, default=False, help="clamp scores")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses")
    parser.add_argument("--contrastive_align_loss", action="store_true", help="Enable contrastive_align_loss")
    parser.add_argument("--contrastive_hdim", type=int, default=64, help="dim for contrastive embeddings")
    parser.add_argument("--span_loss_type", default="l1", type=str, choices=['l1', 'ce'])
    parser.add_argument("--lw_saliency", type=float, default=4., help="weight for saliency loss")
    parser.add_argument("--saliency_margin", type=float, default=0.2)

    # Matcher
    parser.add_argument('--set_cost_span', default=10, type=float)
    parser.add_argument('--set_cost_giou', default=1, type=float)
    parser.add_argument('--set_cost_class', default=4, type=float)

    # Loss coefficients
    parser.add_argument('--span_loss_coef', default=10, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--label_loss_coef', default=4, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)
    parser.add_argument("--contrastive_align_loss_coef", default=0.02, type=float)
    parser.add_argument("--aux_loss_coef", default=0, type=float)
    parser.add_argument("--dec_loss_coef", default=1, type=float)

    opt = parser.parse_args()

    kwargs = vars(opt)
    config = Config(**kwargs)

    if config.dataset == 'mrhisum':
        train_dataset = MrSumDataset(mode='train', data_path=config.data_path)
        val_dataset = MrSumDataset(mode='val', data_path=config.data_path)
        test_dataset = MrSumDataset(mode='test', data_path=config.data_path)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=BatchCollator())
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=BatchCollator())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=BatchCollator())

        solver = Solver(config, train_loader, val_loader, test_loader, ckpt_path=config.ckpt_path)
        solver.build()

        if config.train:
            best_f1_ckpt_path, best_map50_ckpt_path, best_map15_ckpt_path, best_srho_ckpt_path, best_ktau_ckpt_path, *_ = solver.train()
            ckpt_paths = [best_f1_ckpt_path, best_map50_ckpt_path, best_map15_ckpt_path, best_srho_ckpt_path, best_ktau_ckpt_path]
            for path in ckpt_paths:
                solver.test(path)
            if config.ema:
                print('EMA Testing')
                for path in ckpt_paths:
                    solver.test(path.replace('.pkl', '_ema.pkl'))
        else:
            test_model_ckpt_path = config.ckpt_path
            if test_model_ckpt_path is None:
                print("Trained model checkpoint required. Exit program")
                exit()
            if config.ema:
                print('EMA Testing')
                solver.test(test_model_ckpt_path.replace('.pkl', '_ema.pkl'))
            else:
                solver.test(test_model_ckpt_path)

    elif config.dataset == 'tvsum' or config.dataset == 'summe':
        f1_list = []
        kTau_list = []
        sRho_list = []
        map50_list = []
        map15_list = []

        config.query_dim = -1

        for split in range(5):
            config.split = split
            config.tag = config.tag[:-1] + str(split)
            config.set_dataset_dir()
            train_dataset = SummaryDataset('train', config.dataset, config.split, config.train_val)
            test_dataset = SummaryDataset('test', config.dataset, config.split, config.train_val)
            if config.individual:
                train_dataset = SummaryDataset_multi('train', config.dataset, config.split, config.train_val)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=SummaryBatchCollator(config.query_dim))
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=SummaryBatchCollator(config.query_dim))
            if config.train_val:
                val_dataset = SummaryDataset('val', config.dataset, config.split, config.train_val)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=SummaryBatchCollator(config.query_dim))
            else:
                val_loader = test_loader

            solver = Solver(config, train_loader, val_loader, test_loader, ckpt_path=config.ckpt_path)
            solver.build()

            if config.train:
                if config.train_val == False:
                    *ckpt_paths, test_f1, test_map50, test_map15, test_sRho, test_kTau = solver.train()
                    best_ktau_ckpt_path = ckpt_paths[3]
                    if config.ema:
                        print('EMA Testing')
                        solver.test(best_ktau_ckpt_path.replace('.pkl', '_ema.pkl'))
                    else:
                        solver.test(best_ktau_ckpt_path)
                else:
                    best_f1_ckpt_path, best_map50_ckpt_path, best_map15_ckpt_path, best_srho_ckpt_path, best_ktau_ckpt_path = solver.train()
                    ckpt_paths_list = [best_f1_ckpt_path, best_map50_ckpt_path, best_map15_ckpt_path, best_srho_ckpt_path, best_ktau_ckpt_path]
                    for path in ckpt_paths_list:
                        test_f1, test_map50, test_map15, test_kTau, test_sRho = solver.test(path)
                    if config.ema:
                        print('EMA Testing')
                        for path in ckpt_paths_list:
                            test_f1, test_map50, test_map15, test_kTau, test_sRho = solver.test(path.replace('.pkl', '_ema.pkl'))

                f1_list.append(test_f1)
                kTau_list.append(test_kTau)
                sRho_list.append(test_sRho)
                map50_list.append(test_map50)
                map15_list.append(test_map15)

            else:
                test_model_ckpt_path = config.best_ktau_save_dir + '/best_ktau.pkl'
                test_f1, test_map50, test_map15, test_kTau, test_sRho = solver.test(test_model_ckpt_path)
                if config.ema:
                    print('EMA Testing')
                    test_f1, test_map50, test_map15, test_kTau, test_sRho = solver.test(test_model_ckpt_path.replace('.pkl', '_ema.pkl'))
                f1_list.append(test_f1)
                kTau_list.append(test_kTau)
                sRho_list.append(test_sRho)
                map50_list.append(test_map50)
                map15_list.append(test_map15)

        print(f'KTau: {np.mean(kTau_list)}')
        print(f'SRho: {np.mean(sRho_list)}')
