from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        print(f"Model size = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, gt=None):
        data_set, data_loader = data_provider(self.args, flag, gt)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, reduction='mean'):
        criterion = nn.MSELoss(reduction=reduction)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask_x, batch_mask_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                batch_mask_x = batch_mask_x.float().to(self.device)
                batch_mask_y = batch_mask_y.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_mask_y = batch_mask_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                mask = batch_mask_y.detach().cpu()

                loss = criterion(pred, true)
                if self.args.misstsm:
                    loss = (loss * mask.float()).sum()
                    non_zero_elements = mask.sum()
                    loss = loss / non_zero_elements
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # path = os.path.join(self.args.checkpoints, setting)
        path = os.path.join(self.args.checkpoints, self.args.data + "_" + str(self.args.pred_len)) + "_" + str(self.args.trial)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, root_path=self.args.root_path)

        model_optim = self._select_optimizer()
        if self.args.misstsm:
            criterion = self._select_criterion(reduction='none')
        else:
            criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask_x, batch_mask_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                batch_mask_x = batch_mask_x.float().to(self.device)
                batch_mask_y = batch_mask_y.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        batch_mask_y = batch_mask_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs, batch_y)
                        if self.args.misstsm:
                            loss = (loss * batch_mask_y.float()).sum()
                            non_zero_elements = batch_mask_y.sum()
                            loss = loss / non_zero_elements
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_mask_y = batch_mask_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    if self.args.misstsm:
                        loss = (loss * batch_mask_y.float()).sum()
                        non_zero_elements = batch_mask_y.sum()
                        loss = loss / non_zero_elements

                        # if torch.isnan(loss):
                        #     print(f"NaN loss detected at step {i}")
                        #     print(f"Outputs contain NaN: {torch.isnan(outputs).any()}")
                        #     print(f"Targets contain NaN: {torch.isnan(batch_y).any()}")
                        #     print(f"Mask sum: {batch_mask_y.sum()}")
                        #     # continue  # Skip this batch

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        # best_model_path = path + '/' + 'checkpoint.pth'
        best_model_path = path + '/' + 'checkpoint_'+self.args.root_path.split('/')[-1]+'.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            path = os.path.join(self.args.checkpoints, self.args.data + "_" + str(self.args.pred_len)) + "_" + str(self.args.trial)
            best_model_path = path + '/' + 'checkpoint_'+self.args.root_path.split('/')[-1]+'.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks_x = []
        masks_y = []

        folder_path = self.args.output_path
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask_x, batch_mask_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                batch_mask_x = batch_mask_x.float().to(self.device)
                batch_mask_y = batch_mask_y.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_mask_y = batch_mask_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_mask_y = batch_mask_y.detach().cpu().numpy()
                batch_mask_x = batch_mask_x.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y
                mask_y = batch_mask_y
                mask_x = batch_mask_x

                preds.append(pred)
                trues.append(true)
                masks_x.append(mask_x)
                masks_y.append(mask_y)

                if i % 100 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    full_mask = np.concatenate((mask_x[0, :, -1], mask_y[0, :, -1]), axis=0)

                    # Set the missing values to NaN based on the mask
                    gt[full_mask == 0] = np.nan

                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        masks_y = np.array(masks_y)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        masks_y = masks_y.reshape(-1, masks_y.shape[-2], masks_y.shape[-1])

        mae, mse, rmse = metric(preds, trues, masks_y)
        print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_long_term_forecast.txt", 'a')
        f = open(folder_path+"result_"+self.args.root_path.split('/')[-1]+".txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

    
    def test_masked(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        test_data_gt, test_loader_gt = self._get_data(flag='test', gt=True)
        
        if test:
            print('loading model')
            path = os.path.join(self.args.checkpoints, self.args.data + "_" + str(self.args.pred_len)) + "_" + str(self.args.trial)
            best_model_path = path + '/' + 'checkpoint_'+self.args.root_path.split('/')[-1]+'.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        masks_y = []
        masks_x = []

        folder_path = self.args.output_path
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask_x, batch_mask_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                batch_mask_x = batch_mask_x.float().to(self.device)
                batch_mask_y = batch_mask_y.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_mask_x, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_mask_y = batch_mask_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_mask_y = batch_mask_y.detach().cpu().numpy()
                batch_mask_x = batch_mask_x.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                    print(f"Data scaled and inverse transformed")

                pred = outputs
                mask_y = batch_mask_y
                mask_x = batch_mask_x

                preds.append(pred)
                # masks_y.append(mask_y)
                masks_x.append(mask_x)
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_mask_x, batch_mask_y) in enumerate(test_loader_gt):
                
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                true = batch_y.detach().cpu().numpy()
                
                trues.append(true)
                
                batch_mask_y = batch_mask_y.float().to(self.device)
                batch_mask_y = batch_mask_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_mask_y = batch_mask_y.detach().cpu().numpy()
                mask_y = batch_mask_y
                masks_y.append(mask_y)
                
                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = true[0, :, -1]
                    
                    pd = preds[i][0, :, -1]
                    full_mask = masks_x[i][0, :, -1]
                    
                    inputx = input[0, :, -1]
                    inputx[full_mask == 0] = np.nan
                    # print(f"inputx shape = {inputx.shape}")
                    # print(f"gt shape = {gt.shape}")
                    gt = np.concatenate((inputx, gt), axis=0)
                    pd = np.concatenate((inputx, pd), axis=0)
                    
                    visual(true=gt, 
                           preds=pd, 
                           name=os.path.join(folder_path, self.args.root_path.split('/')[-1] + "_" + str(self.args.pred_len) + "_" + str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        masks_y = np.array(masks_y)
        # print(f"masks_y is {masks_y}")

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        masks_y = masks_y.reshape(-1, masks_y.shape[-2], masks_y.shape[-1])

        mae, mse, rmse = metric(preds, trues, masks_y)
        print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_long_term_forecast.txt", 'a')
        f = open(folder_path+"result_"+self.args.root_path.split('/')[-1]+".txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
    
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
#             path = os.path.join(self.args.checkpoints, setting)
#             best_model_path = path + '/' + 'checkpoint.pth'
            path = os.path.join(self.args.checkpoints, self.args.data + "_" + str(self.args.pred_len)) + "_" + str(self.args.trial)
            best_model_path = path + '/' + 'checkpoint_'+self.args.root_path.split('/')[-1]+'.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # np.save(folder_path + 'real_prediction.npy', preds)

        return