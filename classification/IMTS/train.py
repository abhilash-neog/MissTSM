# -*- coding:utf-8 -*-
import os
import argparse
import warnings
import time
import math
import sys
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
from model import MaskedAutoencoder
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D
from utils.utils import get_1d_sincos_pos_embed
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# Argument parser for command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='physionet', choices=['P12', 'P19', 'physionet', 'mimic3'])
parser.add_argument('--epochs', type=int, default=100)  #
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)

parser.add_argument('--history', type=int, default=48, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in GAT")
parser.add_argument('-ps', '--patch_size', type=float, default=6, help="window size for a patch")
parser.add_argument('--stride', type=float, default=6, help="period stride for patch sliding")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Hidden dim of node embeddings")
parser.add_argument('--alpha', type=float, default=1, help="Proportion of Time decay")
parser.add_argument('--res', type=float, default=1, help="Res")

# model define
parser.add_argument('--encoder_embed_dim', type=int, default=64, help='encoder embedding dimension in the feature space')
parser.add_argument('--encoder_depth', type=int, default=2, help='number of encoder blocks')
parser.add_argument('--encoder_num_heads', type=int, default=2, help='number of encoder multi-attention heads')
parser.add_argument('--decoder_depth', type=int, default=2, help='number of decoder blocks')
parser.add_argument('--decoder_num_heads', type=int, default=4, help='number of decoder multi-attention heads')
parser.add_argument('--decoder_embed_dim', type=int, default=32, help='decoder embedding dimension in the feature space')
parser.add_argument('--mlp_ratio', type=int, default=4, help='mlp ratio for vision transformer')
# parser.add_argument('--finetune_checkpoints_dir', type=str, default='./finetune_checkpoints')
parser.add_argument('--trial', type=int, default=0)
parser.add_argument('--task_name', type=str, default='finetune')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--pct_start', type=float, default=0.3)
parser.add_argument('--mask_ratio', type=float, default=0.5)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--seq_len', type=int, default=156)

# update lr default value

args, unknown = parser.parse_known_args()
print(args)

# Set CUDA environment variables

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from lib.utils import *

# Set device for training - direct GPU selection
args.device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
torch.use_deterministic_algorithms(True, warn_only=True)


# Model save path
model_path = './models/'
if not os.path.exists(model_path):
    os.mkdir(model_path)

# Load command-line arguments
dataset = args.dataset
batch_size = args.batch_size
learning_rate = args.lr
num_epochs = args.epochs

class ModelPlugins():
    
    def __init__(self, 
                 window_len, 
                 enc_embed_dim,
                 dec_embed_dim,
                 task_name,
                 num_feats,
                 batch_size,
                 device):
        
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.window_len = window_len 
        self.device = device
        self.batch_size = batch_size
        self.num_feats = num_feats
        
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.window_len + 1, self.enc_embed_dim), requires_grad=False).to(self.device)
        self.pos_embed = PositionalEncoding2D(enc_embed_dim).to(self.device)
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.window_len + 1, self.dec_embed_dim), requires_grad=False).to(self.device)
        # self.decoder_pos_embed = PositionalEncoding2D(dec_embed_dim).to(self.device)
        
        self.initialize_embeddings()
    
    def initialize_embeddings(self):
        
        enc_z = torch.rand((1, self.window_len + 1, self.num_feats, self.enc_embed_dim)).to(self.device) # +1 for the cls token
        self.pos_embed = self.pos_embed(enc_z)
        
#         pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.window_len, cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.window_len, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))


# Recursive function to determine the layer of patches
def layer_of_patches(n_patch):
    if n_patch == 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + layer_of_patches(n_patch / 2)
    else:
        return layer_of_patches(n_patch + 1)
# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_finetune_details(args, model, train_steps):
    # path = os.path.join(args.finetune_checkpoints_dir, dataset + "_v" + str(args.trial))
    # if not os.path.exists(path):
    #     os.makedirs(path)
    
    torch.autograd.set_detect_anomaly(True)
    # early_stopping = EarlyStopping(patience=args.patience, verbose=True, args=args)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.epochs,
                                        max_lr=args.lr)
    
    return optimizer, scheduler, criterion
    
print('Dataset used: ', dataset)

# Dataset specific parameters
if dataset == 'P12':
    base_path = './data/P12'
    start = 0
    variables_num = 36
    d_static = 9
    args.d_static = 9
    timestamp_num = 215
    n_class = 2
    args.n_class = 2
    split_idx = 1
    args.history = 48
    args.patch_size = args.history  # Single window covering entire history
    args.stride = args.history      # No overlap
    num_feats = 36
    args.seq_len = 171
elif dataset == 'physionet':
    base_path = './data/physionet'
    start = 4
    variables_num = 36
    d_static = 9
    args.d_static = 9
    timestamp_num = 215
    n_class = 2
    args.n_class = 2
    split_idx = 5
    args.patch_size=args.history
    args.stride=args.history
    num_feats = 36
    args.seq_len = 156
elif dataset == 'P19':
    base_path = './data/P19'
    d_static = 6
    args.d_static = 6
    variables_num = 34
    timestamp_num = 60
    n_class = 2
    args.n_class = 2
    split_idx = 1
    args.history = 60
    args.patch_size = args.history
    args.stride = args.history
    args.seq_len = 59
    num_feats = 34
elif dataset == 'mimic3':
    base_path = '../data/mimic3'
    start = 0
    d_static = 0
    args.d_static = 0
    variables_num = 16
    timestamp_num = 292
    n_class = 2
    args.n_class = 2
    split_idx = 0
    args.history = 48

# Evaluation metrics
acc_arr = []
auprc_arr = []
auroc_arr = []


# Run five experiments
for k in range(5):
    # Set different random seed
    torch.manual_seed(k)
    torch.cuda.manual_seed(k)
    np.random.seed(k)

    if dataset == 'P12':
        split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
    elif dataset == 'physionet':
        split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
    elif dataset == 'P19':
        split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
    elif dataset == 'mimic3':
        split_path = ''

    # Prepare data and split the dataset
    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, dataset=dataset)
    print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

    args.ndim = variables_num
    # setting npatches to 1
    print(f"args.patch_size = {args.patch_size}")
    print(f"args.history = {args.history}")
    
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
    args.patch_layer = layer_of_patches(args.npatch)
    args.scale_patch_size = args.patch_size / args.history
    args.task = 'classification'

    # Normalize data and extract required model inputs
    if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet':
        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))

        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

        # Calculate mean and standard deviation of variables in the training set
        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)
        
        Ptrain_tensor, Ptrain_mask_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor \
            = tensorize_normalize_extract_feature_patch(Ptrain, ytrain, mf, stdf, ms, ss, args)
        # print(f"After tensor normalize, Ptrain_tensor = {Ptrain_tensor.squeeze().shape}")
        # print(f"Ptrain_mask_tensor = {Ptrain_mask_tensor.squeeze().shape}")
        # print(f"ytrain_tensor = {ytrain_tensor.shape}")
        Ptrain_tensor = Ptrain_tensor.squeeze()#.to(args.device)
        Ptrain_mask_tensor = Ptrain_mask_tensor.squeeze()#.to(args.device)
        ytrain_tensor = ytrain_tensor.squeeze()#.to(args.device)
        print(f"Ptrain_tensor shape = {Ptrain_tensor.shape}")
        
        Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor \
            = tensorize_normalize_extract_feature_patch(Pval, yval, mf, stdf, ms, ss, args)
        Pval_tensor = Pval_tensor.squeeze()#.to(args.device)
        Pval_mask_tensor = Pval_mask_tensor.squeeze()#.to(args.device)
        yval_tensor = yval_tensor.squeeze()#.to(args.device)
        print(f"Pval_tensor shape = {Pval_tensor.shape}")

        Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor \
            = tensorize_normalize_extract_feature_patch(Ptest, ytest, mf, stdf, ms, ss, args)
        Ptest_tensor = Ptest_tensor.squeeze()#.to(args.device)
        Ptest_mask_tensor = Ptest_mask_tensor.squeeze()#.to(args.device)
        ytest_tensor = ytest_tensor.squeeze()#.to(args.device)
        print(f"Ptest_tensor shape = {Ptest_tensor.shape}")

    elif dataset == 'mimic3':
        T, F = timestamp_num, variables_num
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        for i in range(len(Ptrain)):
            Ptrain_tensor[i][:Ptrain[i][4]] = Ptrain[i][2]

        # Calculate mean and standard deviation of variables in the training set
        mf, stdf = getStats(Ptrain_tensor)

        Ptrain_tensor, Ptrain_mask_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor \
            = tensorize_normalize_exact_feature_mimic3_patch(Ptrain, ytrain, mf, stdf, args)
        Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor \
            = tensorize_normalize_exact_feature_mimic3_patch(Pval, yval, mf, stdf, args)
        Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor \
            = tensorize_normalize_exact_feature_mimic3_patch(Ptest, ytest, mf, stdf, args)

    # Load the model
    model = MaskedAutoencoder(args=args, num_feats=num_feats).to(args.device)
    mpl = ModelPlugins(window_len=args.seq_len, 
                        enc_embed_dim=model.embed_dim, 
                        dec_embed_dim=model.decoder_embed_dim,
                        num_feats=num_feats, 
                        task_name=args.task_name, 
                        batch_size=args.batch_size,
                        device=args.device)

    

    # params = (list(model.parameters()))
    # print('model', model)
    # print('parameters:', count_parameters(model))

    # Cross-entropy loss, Adam optimizer
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Upsample minority class
    idx_0 = np.where(ytrain == 0)[0]
    idx_1 = np.where(ytrain == 1)[0]
    n0, n1 = len(idx_0), len(idx_1)
    expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
    expanded_n1 = len(expanded_idx_1)
    K0 = n0 // int(batch_size / 2)
    K1 = expanded_n1 // int(batch_size / 2)
    n_batches = np.min([K0, K1])
    train_steps = int(n_batches)

    best_val_epoch = 0
    best_aupr_val = best_auc_val = 0.0
    best_loss_val = 100.0

    print('Stop epochs: %d, Batches/epoch: %d, Total batches: %d' % (
        num_epochs, n_batches, num_epochs * n_batches))

    optimizer, scheduler, criterion = setup_finetune_details(args, model, train_steps)
    min_vali_loss=None
    masked_penalize=False
    losses = []

    start = time.time()

    # Training loop
    for epoch in range(num_epochs):
        # if epoch - best_val_epoch > 5:
        #     break
        """Training"""
        model.train()

        # Shuffle data
        np.random.shuffle(expanded_idx_1)
        I1 = expanded_idx_1
        np.random.shuffle(idx_0)
        I0 = idx_0
        batch_loss, masked_batch_loss, unmasked_batch_loss = 0, 0, 0
        for n in tqdm(range(n_batches)):
            # Get current batch data
            idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
            idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
            idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
            P, P_mask, P_static, P_time, y = \
                Ptrain_tensor[idx], Ptrain_mask_tensor[idx], Ptrain_static_tensor[idx] if d_static != 0 else None, \
                    Ptrain_time_tensor[idx], ytrain_tensor[idx]
            
            P = P.to(args.device)
            P_mask = P_mask.to(args.device)
            y = y.to(args.device)
            P = torch.nan_to_num(P).float().to(args.device)
            
            outputs = model(P, P_mask, mpl)
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # loss /= self.accum_iter
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        # Calculate training set evaluation metrics
        train_probs = torch.squeeze(outputs)
        train_probs = train_probs.cpu().detach().numpy()
        train_y = y.cpu().detach().numpy()
        train_auroc = roc_auc_score(train_y, train_probs[:, 1])
        train_auprc = average_precision_score(train_y, train_probs[:, 1])
        print(f"train auroc: {train_auroc}, train auprc: {train_auprc}")

        """Validation"""
        model.eval()
        with torch.no_grad():
            Pval_tensor = torch.nan_to_num(Pval_tensor).to(args.device)
            Pval_mask_tensor = Pval_mask_tensor.to(args.device)
            out_val = model(Pval_tensor, Pval_mask_tensor, mpl)
            # out_val = evaluate_model_patch(model, Pval_tensor, Pval_mask_tensor, Pval_static_tensor, Pval_time_tensor,
            #                             n_classes=n_class, batch_size=batch_size)
            out_val = torch.squeeze(out_val)
            out_val = out_val.detach().cpu().numpy()

            y_val_pred = np.argmax(out_val, axis=1)
            acc_val = np.sum(yval.ravel() == y_val_pred.ravel()) / yval.shape[0]
            val_loss = torch.nn.CrossEntropyLoss().to(args.device)(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)).long())
            auc_val = roc_auc_score(yval, out_val[:, 1])
            aupr_val = average_precision_score(yval, out_val[:, 1])
            print(
                "Validation: Epoch %d, train_loss:%.4f, train_auprc:%.2f, train_auroc:%.2f, val_loss:%.4f, acc_val: %.2f, aupr_val: %.2f, auc_val: %.2f" %
                (epoch, loss.item(), train_auprc * 100, train_auroc * 100,
                 val_loss.item(), acc_val * 100, aupr_val * 100, auc_val * 100))

            # Save the model weights with the best AUPRC on the validation set
            if aupr_val > best_aupr_val:
                best_auc_val = auc_val
                best_aupr_val = aupr_val
                best_val_epoch = epoch
                save_time = str(int(time.time()))
                torch.save(model.state_dict(),
                           model_path + '_' + dataset + '_' + save_time + '_' + str(k) + '.pt')




    end = time.time()
    time_elapsed = end - start
    print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

    """Testing"""
    model.eval()
    model.load_state_dict(
        torch.load(model_path + '_' + dataset + '_' + save_time + '_' + str(k) + '.pt'))
    with torch.no_grad():
        Ptest_tensor = torch.nan_to_num(Ptest_tensor).to(args.device)
        Ptest_mask_tensor = torch.nan_to_num(Ptest_mask_tensor).to(args.device)

        out_test = model(Ptest_tensor, Ptest_mask_tensor, mpl)
        # out_test = evaluate_model_patch(model, Ptest_tensor, Ptest_mask_tensor, Ptest_static_tensor, Ptest_time_tensor,
        #                                 n_classes=n_class, batch_size=batch_size).numpy()
        
        # Convert PyTorch tensor to NumPy
        out_test = out_test.detach().cpu().numpy()
        
        denoms = np.sum(np.exp(out_test.astype(np.float64)), axis=1).reshape((-1, 1))
        y_test = ytest.copy()
        probs = np.exp(out_test.astype(np.float64)) / denoms
        ypred = np.argmax(out_test, axis=1)
        acc = np.sum(y_test.ravel() == ypred.ravel()) / y_test.shape[0]
        auc = roc_auc_score(y_test, probs[:, 1])
        aupr = average_precision_score(y_test, probs[:, 1])

        print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))
        print('classification report', classification_report(y_test, ypred))
        print(confusion_matrix(y_test, ypred, labels=list(range(n_class))))

    acc_arr.append(acc * 100)
    auprc_arr.append(aupr * 100)
    auroc_arr.append(auc * 100)

print('args.dataset', args.dataset)
# Display the mean and standard deviation of five runs
mean_acc, std_acc = np.mean(acc_arr), np.std(acc_arr)
mean_auprc, std_auprc = np.mean(auprc_arr), np.std(auprc_arr)
mean_auroc, std_auroc = np.mean(auroc_arr), np.std(auroc_arr)
print('------------------------------------------')
print('Accuracy = %.1f±%.1f' % (mean_acc, std_acc))
print('AUPRC    = %.1f±%.1f' % (mean_auprc, std_auprc))
print('AUROC    = %.1f±%.1f' % (mean_auroc, std_auroc))