import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np

from .new_prop_prototype import MultiHeadURT, SingleHeadMVURT

from models import Informer, Autoformer



class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # self.num_models = 1
        self.num_models = configs.n_learner
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.batch_size = configs.batch_size
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)


        

        print("bacth size: " + str(configs.batch_size))
        print("enc in: " + str(configs.enc_in))
        print("label_len: " + str(configs.label_len))
        print("pred_len: " + str(configs.pred_len))
        print("num_fastlearners: " + str(self.num_models))

        # self.flat_dim =  configs.batch_size * configs.enc_in * configs.label_len * configs.pred_len
        self.flat_dim = configs.pred_len *  configs.enc_in
        self.flat_dim = configs.pred_len
        #URT_model  = MultiHeadURT(key_dim=512, query_dim=flat_dim, hid_dim=1024, temp=1, att="dotproduct", n_head=xargs['urt.head'])
        
        # self.URT_model  = MultiHeadURT(key_dim=self.flat_dim, query_dim=8*self.flat_dim , hid_dim=1024, temp=1, att="dotproduct", n_head=1)
        # self.URT_model  = MultiHeadURT(key_dim=self.flat_dim, query_dim=8*self.flat_dim, hid_dim=1024, temp=1, att="dotproduct", n_head=1)
        # self.URT_model  = MultiHeadURT(key_dim=self.flat_dim, query_dim=configs.batch_size*self.flat_dim, hid_dim=1024, temp=1, att="dotproduct", n_head=7)
        # self.URT_model  = MultiHeadURT(key_dim=self.flat_dim, query_dim=configs.batch_size*self.flat_dim, hid_dim=1024, temp=1, att="cosine", n_head=1)
        #self.URT_model  = MultiHeadURT(key_dim=512, query_dim=512*8, hid_dim=1024, temp=1, att="dotproduct", n_head=1)
        # self.URT_model  = MultiHeadURT(key_dim=512, query_dim=self.flat_dim, hid_dim=1024, temp=1, att="dotproduct", n_head=1)

        # self.URT_model  = MultiHeadURT(key_dim=self.flat_dim, query_dim=configs.batch_size*self.flat_dim, hid_dim=1024, pred_length=configs.pred_len, temp=1,  att="dotproduct", n_head=7)
        # self.URT_model  = SingleHeadMVURT(key_dim=self.flat_dim, query_dim=configs.batch_size*self.flat_dim, hid_dim=1024, pred_length=configs.pred_len, temp=1,  att="dotproduct", nv=7)
        # self.URT_model  = MultiHeadURT(key_dim=self.flat_dim, query_dim=configs.batch_size*self.flat_dim, hid_dim=1024, temp=1, att="cosine", n_head=1)
        self.URT_model  = MultiHeadURT(key_dim=configs.pred_len , query_dim=configs.pred_len*configs.enc_in, hid_dim=4096, temp=1, att="cosine", n_head=1)
        # self.URT_model  = MultiHeadURT(key_dim=configs.pred_len , query_dim=configs.pred_len*8, hid_dim=1024, temp=1, att="cosine", n_head=1)
        # self.URT_model  = MultiHeadURT(key_dim=configs.pred_len , query_dim=configs.pred_len*configs.enc_in, hid_dim=1024, temp=1, att="dotproduct", n_head=1)

        # Encoder
        self.encoder=[]
        for i in range (0, self.num_models):
            self.encoder.append(Encoder(
                [
                    EncoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model)
            ))


        # Decoder
        self.decoder=[]
        for i in range (0, self.num_models):
            self.decoder.append(Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                # projection=None
            ))

        self.projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        # self.reshape = torch.Tensor(1, self.num_models, bias=True)
        # self.reshapeTensor = torch.ones(1, 3).cuda()
        # self.reshapeTensor = torch.ones(1,self.num_models).cuda()
        # self.reshapeTensor = torch.rand(1, 3).cuda()
        # self.reshapeTensor = F.softmax(self.reshapeTensor,dim=1)
        # self.projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.postProjection = nn.Linear(self.num_models, 1, bias=True)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        
        # enc
        eo = self.enc_embedding(x_enc, x_mark_enc)
        enc_out=[]
        attns=[]
        for i in range (0, self.num_models):
            self.encoder[i].to(x_enc.device)
            # print(eo.is_cuda)
            # print(self.encoder[i.is_cuda)
            # print(next(self.encoder[i].parameters()).is_cuda)
            e, a = self.encoder[i](eo, attn_mask=enc_self_mask)
            enc_out.append(e)
            attns.append(a)
        

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part=[]
        trend_part=[]
        for i in range (0, self.num_models):
            self.decoder[i].to(x_enc.device)
            s, t  = self.decoder[i](dec_out, enc_out[i], x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
            seasonal_part.append(s)
            trend_part.append(t)


        # final
        #print(trend_part[0].shape)
        #print(seasonal_part[0].shape)
        # dec_out = trend_part + seasonal_part
        dec_out = []
        dec_out2 = []
        for i in range (0, self.num_models):
            dec_out.append(trend_part[i] + seasonal_part[i])
            dec_out2.append(trend_part[i] + seasonal_part[i])
            # dec_out2.append(torch.flatten(trend_part[i] + seasonal_part[i]))
        


        dec_out = torch.stack(dec_out)
        dec_out2 = torch.stack(dec_out2)

        dec_out2 = dec_out[:,:, -self.pred_len:, :]
        # print("dec_out2 before mean")
        # print(dec_out2.shape)
        dec_out2 = torch.mean(dec_out2,axis=1)
        # print("dec_out2 after mean")
        # print(dec_out2.shape)
        dec_out2 = dec_out2.reshape(dec_out2.shape[0],dec_out2.shape[2],dec_out2.shape[1])
        # print("dec_out2 after reshape")
        # print(dec_out2.shape)
        urt_out = self.URT_model(dec_out2)
        # For single head only
        urt_out = urt_out[:,:,0]
        # print(urt_out.shape)
        # for i in range (0,7):
        #     print(urt_out[:,i])
        # print(a[:,0])
        # print(a[0,:])


        # dec_out2 = dec_out2.reshape(dec_out2.shape[0],dec_out2.shape[1],dec_out2.shape[2]*dec_out2.shape[3])
        # dec_out2 = dec_out2.reshape(dec_out2.shape[1],dec_out2.shape[0],dec_out2.shape[2]*dec_out2.shape[3])
        
        #dec_out2 = dec_out2.reshape(dec_out2.shape[1],dec_out2.shape[0],dec_out2.shape[3],dec_out2.shape[2])
        # print(dec_out2.shape)
        # (a,b,c,d) = dec_out2.reshape
        # print(dec_out2.shape[0])
        # print(dec_out2.shape[1])
        # print(dec_out2.shape[2])
        # print(dec_out2.shape[3])

        # print("check decout 2")
        # print(dec_out2[:,:,:,0].shape)
        # if(dec_out2.shape[1] < self.batch_size):
        #     n_repeat = int((self.batch_size - dec_out2.shape[1])/dec_out2.shape[1]) + 1
        #     dec_out3 = dec_out2
        #     for i in range(0,n_repeat):
        #         dec_out3 = torch.cat((dec_out3,dec_out2),dim=1)
        #     a = self.URT_model(dec_out3[:,:self.batch_size,:,0])
        #     a = a[:,:dec_out2.shape[1],:,:]
        # else:
        #     a = self.URT_model(dec_out2[:,:,:,0])
        
        # b = []
        # for i in range(0, a.shape[0]):
        #     t =  a[i]
        #     b.append(torch.mean(t.reshape(-1))) 
        # b = torch.stack(b)
        # idx = torch.argmax(b)
        # # print(b)
        # # urt_output = a[idx]
        # urt_output = torch.mean(a,dim=0)
        # print("selected index: " + str(idx))
        # print(dec_out)
        # print(dec_out.shape)
        # dec_out = dec_out.mean(axis=1)
        a,b,c,d = dec_out.shape
        fin_out = torch.zeros([b,c,d]).cuda()
        # print(fin_out.shape)
        for i in range(0,d):
            for j in range(0,a):
                fin_out[:,:,i] = fin_out[:,:,i] + dec_out[j,:,:,i] * urt_out[j,i]


        # dec_out = torch.mean(dec_out,axis=0)
        # print(dec_out.shape)
        # print(dec_out.shape)
        # print("check attns")
        # print(attns[0])

        # print("Autoformer output")
        # print(dec_out[:, -self.pred_len:, :].shape)
        # print("URT output")
        # print(urt_output.shape)

        if self.output_attention:
            # return dec_out[:, -self.pred_len:, :], attns
            # return dec_out[:, -self.pred_len:, :], attns[0]
            return fin_out[:, -self.pred_len:, :], attns[0]
        else:
            # return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            return fin_out[:, -self.pred_len:, :]  # [B, L, D]

        # if self.output_attention:
        #     # return dec_out[:, -self.pred_len:, :], attns
        #     return urt_output, attns[0]
        # else:
        #     return urt_output  # [B, L, D]


    def forward_for_slowlearner(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        
        # enc
        eo = self.enc_embedding(x_enc, x_mark_enc)
        enc_out=[]
        attns=[]
        for i in range (0, self.num_models):
            self.encoder[i].to(x_enc.device)
            # print(eo.is_cuda)
            # print(self.encoder[i.is_cuda)
            # print(next(self.encoder[i].parameters()).is_cuda)
            e, a = self.encoder[i](eo, attn_mask=enc_self_mask)
            enc_out.append(e)
            attns.append(a)
        

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part=[]
        trend_part=[]
        for i in range (0, self.num_models):
            self.decoder[i].to(x_enc.device)
            s, t  = self.decoder[i](dec_out, enc_out[i], x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
            seasonal_part.append(s)
            trend_part.append(t)


        # final
        #print(trend_part[0].shape)
        #print(seasonal_part[0].shape)
        # dec_out = trend_part + seasonal_part
        dec_out_trend = []
        dec_out_season = []
        # dec_out2 = []
        for i in range (0, self.num_models):
            dec_out_trend.append(trend_part[i])
            dec_out_season.append(seasonal_part[i])
            # dec_out2.append(trend_part[i] + seasonal_part[i])

        dec_out_trend = torch.stack(dec_out_trend)
        dec_out_season = torch.stack(dec_out_season)
        dec_out_season = self.projection(dec_out_season)
        # print("trend and season shape")
        # print(dec_out_trend.shape)
        # print(dec_out_season.shape)

        # print(self.reshapeTensor.shape)
        # print(dec_out.shape)
        # dec_out = torch.matmul(self.reshapeTensor,dec_out)\
        dec_out = dec_out_season + dec_out_trend
        a,b,c,d = dec_out.shape
        dec_out = torch.reshape(dec_out,(b,c,d,a))
        dec_out = self.postProjection(dec_out)
        dec_out = torch.squeeze(dec_out,-1)

        # dec_out = torch.einsum("ij, jklm -> iklm", self.reshapeTensor, dec_out)
        # dec_out = torch.matmul(self.reshapeTensor,dec_out)
        # dec_out = self.reshapeTensor@dec_out
        # print(dec_out.shape)
        # dec_out = torch.squeeze(dec_out,0)
        # print(dec_out.shape)

        # dec_out = self.preprojection(dec_out)
        if self.output_attention:
            # return dec_out[:, -self.pred_len:, :], attns
            return dec_out[:, -self.pred_len:, :], attns[0]
            # return dec_out[:,:, -self.pred_len:, :], attns[0]
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            # return dec_out[:,:, -self.pred_len:, :]  # [B, L, D]




    def check_params(self):
        count = 0
        for p in self.URT_model.parameters():
            if count < 2:
                if p.requires_grad:
                     print(p.name, p.data)
            count = count + 1
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
