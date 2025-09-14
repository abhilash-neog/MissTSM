import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args['lradj'] == 'type1':
        lr_adjust = {epoch: args['lr'] * (0.5 ** ((epoch - 1) // 1))}
    elif args['lradj'] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args['lradj'] == 'type3':
        lr_adjust = {epoch: args['lr'] if epoch < 1 else args['lr'] * (0.9 ** ((epoch - 3) // 1))}
    elif args['lradj'] == 'constant':
        lr_adjust = {epoch: args['lr']}
    elif args['lradj'] == '3':
        lr_adjust = {epoch: args['lr'] if epoch < 10 else args['lr'] * 0.1}
    elif args['lradj'] == '4':
        lr_adjust = {epoch: args['lr'] if epoch < 15 else args['lr'] * 0.1}
    elif args['lradj'] == '5':
        lr_adjust = {epoch: args['lr'] if epoch < 25 else args['lr'] * 0.1}
    elif args['lradj'] == '6':
        lr_adjust = {epoch: args['lr'] if epoch < 5 else args['lr'] * 0.1}
    elif args['lradj'] == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        torch.save(model, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
        
def transfer_weights(weights_path, model, exclude_head=True, device='cpu'):
    new_state_dict = torch.load(weights_path,  map_location=device).state_dict()
    
    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items():
        # if exclude_head and 'head' in name: continue
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        # if len(unmatched_layers) > 0:
        #     print(f'check unmatched_layers: {unmatched_layers}')
        # else:
        print(f"weights from {weights_path} successfully transferred!\n")
    # model = model.to(device)
    return model