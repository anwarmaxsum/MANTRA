import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from models import Autoformer
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np

import random
from models import Informer, Autoformer



class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # self.num_models = 3
        self.num_models = configs.n_learner
        self.output_attention = configs.output_attention

        # self.models = []
        # for i in range (0, self.num_models):
        #     self.models.append(Autoformer.Model(configs).float().cuda())
        self.models = nn.ModuleList([Autoformer.Model(configs).float().cuda() for i in range(self.num_models)])

        print("[CREATE] " +str(self.num_models)+" learners of Autoformer")

        # for p in self.parameters():
        #     if p.requires_grad:
        #         print(p.name)
        # for name, param in self.state_dict().items():
        #     print(name)

        for i in range(0,self.num_models):
            model = self.models[i]            
            if (i%2)==0:
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        stdev = np.random.uniform(0, 0.01)
                        # mean = np.random.uniform(-0.01, 0.01)
                        m.weight.data.normal_(0, stdev)
            else:
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        # stdev = random.uniform(0, 1)
                        m.weight.data.uniform_(0, 0.01)


            # if (i%2)==0:
            #     for m in model.modules():
            #         if isinstance(m, nn.Linear):
            #             stdev = np.random.uniform(0, 0.01)
            #             m.weight.data.normal_(0, stdev)
            # else:
            #     for m in model.modules():
            #         if isinstance(m, nn.Linear):
            #             # stdev = random.uniform(0, 1)
            #             m.weight.data.uniform_(0, 0.01)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        # mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        # seasonal_init, trend_init = self.decomp(x_enc)
        # # decoder input
        # trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        # seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        
        # enc
        
        attns=[]
        dec_out = []

        if self.output_attention:
            for i in range (0, self.num_models):
                do, attn = self.models[i].forward(x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
                dec_out.append(do)
                attns.append(attn)
            
            dec_out = torch.stack(dec_out)
            dec_out = torch.mean(dec_out,axis=0)
            attns = torch.stack(attns)
            attns = torch.mean(attns,axis=0)
            return dec_out, attns
        else:
            for i in range (0, self.num_models):
                do = self.models[i].forward(x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
                dec_out.append(do)
            dec_out = torch.stack(dec_out)
            dec_out = torch.mean(dec_out,axis=0)
            return dec_out



    def forward_1learner(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,idx=0):
        # decomp init
        # mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        # seasonal_init, trend_init = self.decomp(x_enc)
        # # decoder input
        # trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        # seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        
        # enc
        
        attns=[]
        dec_out = []

        if self.output_attention:
            i = idx     
            do, attn = self.models[i].forward(x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
            return do, attn
        else:
            i = idx
            do = self.models[i].forward(x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
            return do



    # def parameters(self, only_trainable=True):
    #     for i in range (0, self.num_models):
    #         model = self.models[i]
    #         for param in model.parameters():
    #             # if only_trainable and not param.requires_grad:
    #             #     continue
    #             yield param

# class Model(nn.Module):
    # def __init__(self, configs):
    #     super(Model, self).__init__()
    #     self.num_autoformer = 10
    #     self.autoformer_array=[]
    #     for m in range(0,self.num_autoformer):
    #         aut = Model(configs).float()
    #         self.autoformer_array.append(Autoformer(configs))
    #     print("create autoformer array")
    #     print(self.autoformer_array)


    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
    #             enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
    #     return 0



# # Define your base models
# model1 = autoformer
# model2 = autoformer
# model3 = autoformer

# torch.nn.init.xavier_uniform(model1.weights)
# torch.nn.init.xavier_uniform(model2.weights)
# torch.nn.init.xavier_uniform(model3.weights)

# # Define the weights for each model
# #weights = torch.tensor([0.4, 0.3, 0.3], dtype=torch.float32)

# # Define an input
# x = dec_out


# # Compute the output of each model
# output1 = model1(x)
# output2 = model2(x)
# output3 = model3(x)

# # Concatenate the outputs
# outputs = torch.cat([output1, output2, output3], dim=1)

# # Take the weighted average of the outputs
# weighted_output = torch.sum(outputs * weights, dim=1)

# print(weighted_output)


# modelarray = []
# numModels = 10
# for m in range(0,numModels):
#   autoformer = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))
#   modelarray.Concatenate(autoformer)

# for m in range(0,numModels):
#   torch.nn.init.xavier_uniform(modelarray[m].weight)

#   #modelarray[m].train()
