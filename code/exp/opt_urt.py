from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, AutoformerS1, Bautoformer, B2autoformer, B3autoformer, B4autoformer, B5autoformer, B6autoformer, B7autoformer, B8autoformer, B9autoformer  
from models import Uautoformer, UautoformerC1, UautoformerC2, Uautoformer2, Transformer, Reformer, Mantra, MantraV1, MantraA, MantraB, MantraD, MantraE
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.slowloss import SlowLearnerLoss, ssl_loss, ssl_loss_v2

from models.new_prop_prototype import MultiHeadURT, SingleHeadMVURT

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import random

import copy
import h5py

warnings.filterwarnings('ignore')


class Opt_URT(Exp_Basic):
    def __init__(self, args):
        super(Opt_URT, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'AutoformerS1': AutoformerS1,
            'Bautoformer': Bautoformer,
            'B2autoformer': B2autoformer,
            'B3autoformer': B3autoformer,
            'B4autoformer': B4autoformer,
            'B5autoformer': B5autoformer,
            'B6autoformer': B6autoformer,
            'B7autoformer': B7autoformer,
            'B8autoformer': B8autoformer,
            'B9autoformer': B9autoformer,
            'Mantra': Mantra,
            'MantraV1': MantraV1,
            'MantraA': MantraA,
            'MantraB': MantraB,
            'MantraD': MantraD,
            'MantraE': MantraE,
            'Uautoformer': Uautoformer,
            'UautoformerC1': UautoformerC1,
            'UautoformerC2': UautoformerC2,
            'Uautoformer2': Uautoformer2,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        # self.slow_model = model_dict[self.args.slow_model].Model(self.args).float().cuda()
        # self.slow_model = model_dict['Autoformer'].Model(self.args).float().cuda()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # slow_model_optim = optim.Adam(self.slow_model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # def _select_slow_optimizer(self):
    #     slow_model_optim = optim.Adam(self.slow_model.parameters(), lr=self.args.learning_rate)
    #     return slow_model_optim
    

    def _select_urt_optimizer(self):
        urt_optim = optim.Adam(self.URT.parameters(), lr=0.0001)
        return urt_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def vali2(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.URT.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

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
                    dec_out = []
                    if self.args.output_attention:
                        for idx in range(0,self.args.n_learner):
                            outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                            dec_out.append(outputs)

                    else:
                        for idx in range(0,self.args.n_learner):
                            outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                            dec_out.append(outputs)

                    dec_out = torch.stack(dec_out)
                    dec_out2 = torch.mean(dec_out,axis=1)
                    dec_out2 = dec_out2.reshape(dec_out2.shape[0],dec_out2.shape[2],dec_out2.shape[1])
                    urt_out = self.URT(dec_out2)

                    a,b,c,d = dec_out.shape
                    fin_out = torch.zeros([b,c,d]).cuda()
                    for k in range(0,d):
                        for l in range(0,a):
                            fin_out[:,:,k] = fin_out[:,:,k] + (dec_out[l,:,:,k] * urt_out[l,k])
                    
                    outputs = fin_out


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        # self.model.train()
        self.URT.train()
        return total_loss


    def train_urt(self, setting):
        
        self.model.load_state_dict(torch.load(os.path.join(str(self.args.checkpoints) + setting, 'checkpoint.pth')))
        self.URT = MultiHeadURT(key_dim=self.args.pred_len , query_dim=self.args.pred_len*self.args.enc_in, hid_dim=4096, temp=1, att="cosine", n_head=self.args.urt_heads).float().cuda()
        
        print("Train URT layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        urt_optim = self._select_urt_optimizer()
        criterion = self._select_criterion()

        best_mse = float(10.0 ** 10)
        # best_urt = self.URT.state_dict()
        print("Best MSE: " + str(best_mse))
        best_urt = copy.deepcopy(self.URT)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.eval()
            self.URT.train()
            epoch_time = time.time()


            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                urt_optim.zero_grad()
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
                        # print("enter with cuda amp")
                        dec_out = []
                        if self.args.output_attention:
                            for idx in range(0,self.args.n_learner):
                                if self.args.use_multi_gpu and self.args.use_gpu:
                                    outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                                else:
                                    outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                                dec_out.append(outputs)

                        else:
                            for idx in range(0,self.args.n_learner):
                                if self.args.use_multi_gpu and self.args.use_gpu:
                                    outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                                else:
                                    outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                                dec_out.append(outputs)

                        dec_out = torch.stack(dec_out)
                        dec_out2 = torch.mean(dec_out,axis=1)
                        dec_out2 = dec_out2.reshape(dec_out2.shape[0],dec_out2.shape[2],dec_out2.shape[1])
                        urt_out = self.URT(dec_out2)

                        a,b,c,d = dec_out.shape
                        fin_out = torch.zeros([b,c,d]).cuda()
                        for k in range(0,d):
                            for l in range(0,a):
                                fin_out[:,:,k] = fin_out[:,:,k] + (dec_out[l,:,:,k] * urt_out[l,k])
                        
                        outputs = fin_out

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                else:
                    dec_out = []
                    if self.args.output_attention:
                        for idx in range(0,self.args.n_learner):
                            if self.args.use_multi_gpu and self.args.use_gpu:
                                outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                            else:
                                outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                            dec_out.append(outputs)

                    else:
                        for idx in range(0,self.args.n_learner):
                            if self.args.use_multi_gpu and self.args.use_gpu:
                                outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                            else:
                                outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                            dec_out.append(outputs)

                    dec_out = torch.stack(dec_out)
                    dec_out2 = torch.mean(dec_out,axis=1)
                    dec_out2 = dec_out2.reshape(dec_out2.shape[0],dec_out2.shape[2],dec_out2.shape[1])
                    urt_out = self.URT(dec_out2)

                    a,b,c,d = dec_out.shape
                    fin_out = torch.zeros([b,c,d]).cuda()
                    for k in range(0,d):
                        for l in range(0,a):
                            fin_out[:,:,k] = fin_out[:,:,k] + (dec_out[l,:,:,k] * urt_out[l,k])
                    
                    outputs = fin_out

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())


              

                urt_optim.zero_grad()
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    # scaler.step(model_optim)
                    scaler.step(urt_optim)
                    # scaler.step(slow_model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # model_optim.step()
                    urt_optim.step()
                    # slow_model_optim.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali2(vali_data, vali_loader, criterion)
            test_loss = self.vali2(test_data, test_loader, criterion)

            if (vali_loss < best_mse):
                best_mse = vali_loss
                # best_urt = self.URT.state_dict()
                best_urt = copy.deepcopy(self.URT)
                print("Update Best URT params")

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

        
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # self.URT.load_state_dict(best_urt)
        self.URT=best_urt


    def test2(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.URT.eval()
        isFirst = True
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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
                        dec_out = []
                        if self.args.output_attention:
                            for idx in range(0,self.args.n_learner):
                                if self.args.use_multi_gpu and self.args.use_gpu:
                                    outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                                else:
                                    outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                                dec_out.append(outputs)


                        else:
                            for idx in range(0,self.args.n_learner):
                                if self.args.use_multi_gpu and self.args.use_gpu:
                                    outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                                else:
                                    outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                                dec_out.append(outputs)


                        dec_out = torch.stack(dec_out)
                        dec_out2 = torch.mean(dec_out,axis=1)
                        dec_out2 = dec_out2.reshape(dec_out2.shape[0],dec_out2.shape[2],dec_out2.shape[1])
                        urt_out = self.URT(dec_out2)

                        a,b,c,d = dec_out.shape
                        fin_out = torch.zeros([b,c,d]).cuda()
                        for k in range(0,d):
                            for l in range(0,a):
                                fin_out[:,:,k] = fin_out[:,:,k] + (dec_out[l,:,:,k] * urt_out[l,k])
                        
                        outputs = fin_out
                else:
                    dec_out = []
                    if self.args.output_attention:
                        for idx in range(0,self.args.n_learner):
                            if self.args.use_multi_gpu and self.args.use_gpu:
                                outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                            else:
                                outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                            dec_out.append(outputs)


                    else:
                        for idx in range(0,self.args.n_learner):
                            if self.args.use_multi_gpu and self.args.use_gpu:
                                outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                            else:
                                outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                            dec_out.append(outputs)


                    dec_out = torch.stack(dec_out)
                    dec_out2 = torch.mean(dec_out,axis=1)
                    dec_out2 = dec_out2.reshape(dec_out2.shape[0],dec_out2.shape[2],dec_out2.shape[1])
                    urt_out = self.URT(dec_out2)

                    a,b,c,d = dec_out.shape
                    fin_out = torch.zeros([b,c,d]).cuda()
                    for k in range(0,d):
                        for l in range(0,a):
                            fin_out[:,:,k] = fin_out[:,:,k] + (dec_out[l,:,:,k] * urt_out[l,k])
                    
                    outputs = fin_out


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                if isFirst:
                    isFirst = False
                    preds = np.array(pred)
                    trues = np.array(true)

                else:
                    preds = np.concatenate((preds,pred), axis=0)
                    trues = np.concatenate((trues,true), axis=0)
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


        


        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)


        fname = "ZZZ_Mantra_ETTm2_pl"+str(self.args.pred_len)+".h5"
        hf = h5py.File(fname, 'w')
        hf.create_dataset('preds', data=preds)
        hf.create_dataset('trues', data=trues)
        hf.close()

        #     np.savetxt(fname, trues[:,:,col],delimiter=",")

        # for col in range (0,preds.shape[-1]):
        # # fname = setting +"_preds_" + str(col) + ".dat"
        #     fname = "ZZZ_Mantra_ETTm2_pl"+str(self.args.pred_len)+"_col"+str(col)+"_preds.csv"
        #     np.savetxt(fname, trues[:,:,col],delimiter=",")
        #     fname = "ZZZ_Mantra_ETTm2_pl"+str(self.args.pred_len)+"_col"+str(col)+"_preds.csv"
        #     np.savetxt(fname, trues[:,:,col],delimiter=",")
        # preds.tofile(fname)
        # # fname = setting +"_trues_" + str(col) + ".dat"
        # fname = "ZZZ_Mantra_ETTm2_pl"+str(self.args.pred_len)+"_trues.dat"
        # trues.tofile(fname)
        # np.savetxt(fname, trues[:,:,col],delimiter=",")

        return




    def test_wo_urt(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        isFirst = True
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                if isFirst:
                    isFirst = False
                    preds = np.array(pred)
                    trues = np.array(true)

                else:
                    preds = np.concatenate((preds,pred), axis=0)
                    trues = np.concatenate((trues,true), axis=0)
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

      
        return

   

    # def MAE(pred, true):
    #     return np.mean(np.abs(pred - true))


    # def MSE(pred, true):
    #     return np.mean((pred - true) ** 2)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
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
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
