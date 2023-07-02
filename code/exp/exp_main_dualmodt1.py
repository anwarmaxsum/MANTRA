from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, AutoformerS1, Bautoformer, B2autoformer, B3autoformer, B4autoformer, Uautoformer, UautoformerC1, UautoformerC2, Uautoformer2, Transformer, Reformer, Mantra, MantraV1, MantraA, MantraB, MantraD
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.slowloss import SlowLearnerLoss, ssl_loss, ssl_loss_v2

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

warnings.filterwarnings('ignore')


class Exp_Main_DualmodT1(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_DualmodT1, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'AutoformerS1': AutoformerS1,
            'Bautoformer': Bautoformer,
            'B2autoformer': B2autoformer,
            'B3autoformer': B3autoformer,
            'B4autoformer': B4autoformer,
            'Mantra': Mantra,
            'MantraV1': MantraV1,
            'MantraA': MantraA,
            'MantraB': MantraB,
            'MantraD': MantraD,
            'Uautoformer': Uautoformer,
            'UautoformerC1': UautoformerC1,
            'UautoformerC2': UautoformerC2,
            'Uautoformer2': Uautoformer2,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        self.slow_model = model_dict[self.args.slow_model].Model(self.args).float().cuda()
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

    def _select_slow_optimizer(self):
        slow_model_optim = optim.Adam(self.slow_model.parameters(), lr=self.args.learning_rate)
        return slow_model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
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
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        slow_model_optim = self._select_slow_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # print("======= Check before epoch =======")
            # self.model.check_params()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)


                # slow_outputs = self.slow_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                # slow_outputs_wo_projection = self.slow_model.forward_for_slowlearner(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                # # slow_outputs_wo_projection = self.slow_model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                # print("check slow output shape")
                # print(slow_outputs.shape)
                # print(slow_outputs_wo_projection.shape)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # print("enter with cuda amp")
                        if self.args.output_attention:
                            # print("enter if")
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            # print("enter else")
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    # print("enter without cuda amp")
                    if self.args.output_attention:
                        # print("enter if")
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        # print("enter else")
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
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
                    # scaler.step(slow_model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    # slow_model_optim.step()


                # print(batch_x.shape)
                # print("check mask")
                # for c in range(0,self.args.n_learner):
                #     print(self.model.encoder[c].attn_layers[0].attention.inner_correlation.top_k_index)



                loss = 0
                s0,s1,s2 = batch_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                m_ones = torch.ones(s0,s1,s2).cuda()
                slow_mark = torch.bernoulli(randuniform).cuda()
                batch_x_slow = torch.clone(batch_x)
                batch_x_slow = batch_x_slow * (m_ones-slow_mark)
                
                #Update for slow model
                # loss = 0
                # s0,s1,s2 = batch_x.shape
                # slow_mark = torch.zeros(s0,s1,s2).cuda()
                # # slow_mark = torch.zeros(s1).cuda()
                # batch_x_slow = torch.clone(batch_x)
                # for c in range(0,self.args.n_learner):
                #     # idx = self.model.decoder[c].layers[self.args.d_layers-1].self_attention.inner_correlation.top_k_index
                #     idx = self.model.encoder[c].attn_layers[0].attention.inner_correlation.top_k_index
                #     # slow_mark[idx] = 1
                #     slow_mark[:,idx,:] = 1
                #     batch_x_slow[:,idx,:] = 0

                # loss = 0
                # s0,s1,s2 = batch_x.shape
                # slow_mark = torch.zeros(s0,s1,s2).cuda()
                # batch_x_slow = torch.clone(batch_x)
                # c = random.randint(0,self.args.n_learner-1)
                # idx = self.model.encoder[c].attn_layers[0].attention.inner_correlation.top_k_index
                # slow_mark[:,idx,:] = 1
                # batch_x_slow[:,idx,:] = 0
                
                # for s in range(0,s1):
                #     if slow_mark[s] == 1:
                #         batch_x_slow[:,s,:] = 0
                dec_inp = torch.zeros_like(batch_y[:, -self.args.seq_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    # print("enter if")
                    # fast_outputs = self.model.forward_for_slowlearner(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    # slow_out = self.slow_model.forward_for_slowlearner(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    # slow_out = self.slow_model.forward_for_slowlearner(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark)[0]
                    # slow_out = self.slow_model.forward_for_slowlearner(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark[:,:dec_inp.shape[1],:])[0]
                    slow_out = self.slow_model.forward(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark[:,:dec_inp.shape[1],:])[0]
                    
                    # # print("check fast and slow output:")
                    # print(fast_outputs.shape)
                    # print(slow_out.shape)
                    # print(fast_outputs.shape)
                    # for nl in range(0,fast_outputs.shape[0]):
                    #     fast_output_nl = fast_outputs[nl]
                    #     slow_out = self.slow_model.forward_and_modulate(batch_x, batch_x_mark, dec_inp, batch_y_mark,fast_output=fast_output_nl)
                    #     # print(slow_out.shape)

                    #     f_dim = -1 if self.args.features == 'MS' else 0
                    #     outputs = slow_out[:, -self.args.pred_len:, f_dim:]
                    #     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    #     loss += criterion(outputs, batch_y)


                        # slow_model_optim.zero_grad()
                        # if self.args.use_amp:
                        #     scaler.scale(loss).backward()
                        #     scaler.step(slow_model_optim)
                        #     scaler.update()
                        # else:
                        #     loss.backward()
                        #     slow_model_optim.step()

                else:
                    # print("enter else")
                    # fast_outputs = self.model.forward_for_slowlearner(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # slow_out = self.slow_model.forward_for_slowlearner(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # slow_out = self.slow_model.forward_for_slowlearner(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark)
                    # slow_out = self.slow_model.forward_for_slowlearner(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark[:,:dec_inp.shape[1],:])
                    slow_out = self.slow_model.forward(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark[:,:dec_inp.shape[1],:])
                    # print("check fast and slow output:")
                    # print(fast_outputs.shape)
                    # print(slow_out.shape)
                    # for nl in range(0,fast_outputs.shape[0]):
                    #     fast_output_nl = fast_outputs[nl]
                    #     slow_out = self.slow_model.forward_and_modulate(batch_x, batch_x_mark, dec_inp, batch_y_mark,fast_output=fast_output_nl)
                    #     # print(slow_out.shape)


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = slow_out[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss += ssl_loss_v2(slow_out, batch_x, slow_mark, s1, s2)

                # loss += criterion(outputs, batch_y)
                # loss += ssl_loss(slow_out, batch_x_slow, slow_mark, s1, s2)
                # loss += ssl_loss(slow_out, batch_x, slow_mark, s1, s2)
                # loss += ssl_loss(slow_out, batch_x, slow_mark, s1, s2)
                # loss += ssl_loss(slow_out, batch_x, batch_x_mark, s1, s2)
                

                slow_model_optim.zero_grad()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(slow_model_optim)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    slow_model_optim.step()
                    model_optim.step()

                # print(fast_outputs.shape)
                # print(outputs.shape)

            # print(">>>>>>> Check after epoch >>>>>>>")
            # self.model.check_params()

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

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
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

        
        # for i in range(0,90):
        #     print(preds[i].shape)

        # for i in range(0,90):
        #     print(preds[i].shape)

        # preds = np.array(preds)
        # trues = np.array(trues)
        # preds = np.stack(preds,axis=0)
        # trues = np.stack(trues,axis=0)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # preds = np.array(preds.flat)
        # trues = np.array(trues.flat)
        # print(preds[0].shape)
        # print(trues[0].shape)

        # # print(trues)
        # print(preds.shape)
        # print(trues.shape)
        print('test shape:', preds.shape, trues.shape)

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
