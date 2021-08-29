# -*- coding: utf-8 -*-
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import sys
from Data import DataSetTrain, DataSetTestNext, DataSetTestNextNew
import os
import logging

FType = torch.FloatTensor
LType = torch.LongTensor

FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class ATPP:
    def __init__(self, dataset_name, file_path_tr, file_path_te, item_length_path, emb_size=128,
                 neg_size=10, hist_len=2, user_count=992, item_count=5000, directed=True, learning_rate=0.001,
                 decay=0.001, batch_size=1024, test_and_save_step=50, epoch_num=100, top_n=30, sample_time=3,
                 sample_size=100, use_hist_attention=True, use_duration=True, use_corr_matrix=True, num_workers=0,
                 norm_method='hour'):
        self.dataset_name = dataset_name
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.user_count = user_count
        self.item_count = item_count

        self.lr = learning_rate
        self.decay = decay
        self.batch = batch_size
        self.test_and_save_step = test_and_save_step
        self.epochs = epoch_num

        self.top_n = top_n
        self.sample_time = sample_time
        self.sample_size = sample_size

        self.directed = directed
        self.use_hist_attention = use_hist_attention
        self.use_duration = use_duration
        self.use_corr_matrix = use_corr_matrix

        self.num_workers = num_workers
        self.norm_method = norm_method
        self.is_debug = False

        self.data_tr = DataSetTrain(file_path_tr, item_length_path, user_count=self.user_count,
                                    item_count=self.item_count,
                                    neg_size=self.neg_size, hist_len=self.hist_len, directed=self.directed)
        self.data_te_old = DataSetTestNext(file_path_te, self.data_tr, user_count=self.user_count,
                                           item_count=self.item_count,
                                           hist_len=self.hist_len, directed=self.directed)
        self.data_te_new = DataSetTestNextNew(file_path_te, self.data_tr, user_count=self.user_count,
                                              item_count=self.item_count,
                                              hist_len=self.hist_len, directed=self.directed)

        self.node_dim = self.data_tr.get_node_dim()
        self.node_emb = torch.tensor(
            np.random.uniform(-0.5 / self.emb_size, 0.5 / self.emb_size, size=(self.node_dim, self.emb_size)),
            dtype=torch.float)
        self.delta_interval_weight = torch.ones(self.node_dim, dtype=torch.float)
        self.hist_attention_weight_long = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=(self.hist_len - 1, self.emb_size)),
            dtype=torch.float)
        self.hist_attention_weight_short = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=(2, self.emb_size)), dtype=torch.float)
        self.corr_matrix = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=(self.emb_size, self.emb_size)),
            dtype=torch.float)
        self.delta_duration_weight = torch.ones(self.node_dim, dtype=torch.float)

        if torch.cuda.is_available():
            self.node_emb = self.node_emb.cuda()
            self.delta_interval_weight = self.delta_interval_weight.cuda()
            self.hist_attention_weight_long = self.hist_attention_weight_long.cuda()
            self.hist_attention_weight_short = self.hist_attention_weight_short.cuda()
            self.corr_matrix = self.corr_matrix.cuda()
            self.delta_duration_weight = self.delta_duration_weight.cuda()
        self.node_emb.requires_grad = True
        self.delta_interval_weight.requires_grad = True
        self.hist_attention_weight_long.requires_grad = True
        self.hist_attention_weight_short.requires_grad = True
        self.corr_matrix.requires_grad = True
        self.delta_duration_weight.requires_grad = True
        self.opt = Adam(lr=self.lr,
                        params=[self.node_emb, self.delta_interval_weight, self.hist_attention_weight_long,
                                self.hist_attention_weight_short,
                                self.corr_matrix, self.delta_duration_weight], weight_decay=self.decay)
        self.loss = torch.FloatTensor()

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, history_delta, history_length, h_time_mask):
        batch = s_nodes.size()[0]
        s_node_emb = torch.index_select(self.node_emb, 0, s_nodes.view(-1)).view(batch, -1)
        t_node_emb = torch.index_select(self.node_emb, 0, t_nodes.view(-1)).view(batch, -1)
        h_node_emb = torch.index_select(self.node_emb, 0, h_nodes.view(-1)).view(batch, self.hist_len, -1)
        n_node_emb = torch.index_select(self.node_emb, 0, n_nodes.view(-1)).view(batch, self.neg_size, -1)

        self.delta_interval_weight.data.clamp_(min=1e-6)
        delta_interval_weight = torch.index_select(self.delta_interval_weight, 0, s_nodes.view(-1)).unsqueeze(1)
        time_interval = torch.abs(t_times.unsqueeze(1) - h_times)

        time_duration_ratio = torch.ones((batch, self.hist_len), dtype=torch.float)
        if torch.cuda.is_available():
            time_duration_ratio = time_duration_ratio.cuda()
        if self.use_duration:
            self.delta_duration_weight.data.clamp_(min=1e-6)
            delta_duration_weight = torch.index_select(self.delta_duration_weight, 0, s_nodes.view(-1)).unsqueeze(
                1).expand(batch, self.hist_len)

            great_equal_index = torch.ge(history_delta, history_length)
            less_than_index = torch.lt(history_delta, history_length)
            time_interval[great_equal_index] = time_interval[great_equal_index] - history_length[great_equal_index]
            time_interval[less_than_index] = time_interval[less_than_index] - history_delta[less_than_index]
            time_interval = torch.abs(time_interval) + 1e-6
            time_duration_ratio[less_than_index] = torch.exp(
                torch.neg(delta_duration_weight[less_than_index]) * (
                        (history_length[less_than_index] - history_delta[less_than_index]) / history_length[
                    less_than_index]))

        h_index = self.hist_len - 1
        if self.use_hist_attention:
            temp_product_long = torch.mul(h_node_emb[:, :h_index, :], self.hist_attention_weight_long.unsqueeze(0))
            attention_long = softmax(((s_node_emb.unsqueeze(1) - temp_product_long) ** 2).sum(dim=2).neg(), dim=1)
            aggre_hist_node_emb = (attention_long.unsqueeze(2) * h_node_emb[:, :h_index, :] * (
                    time_duration_ratio[:, :h_index] * torch.exp(
                torch.neg(delta_interval_weight) * time_interval[:, :h_index]) * h_time_mask[:, :h_index]).unsqueeze(
                2)).sum(dim=1)
            curr_node_emb = h_node_emb[:, h_index, :] * (
                    time_duration_ratio[:, h_index].unsqueeze(1) * torch.exp(
                torch.neg(delta_interval_weight) * time_interval[:, h_index].unsqueeze(1)) * h_time_mask[:,
                                                                                             h_index].unsqueeze(1))
            new_h_node_emb = torch.cat([aggre_hist_node_emb.unsqueeze(1), curr_node_emb.unsqueeze(1)], dim=1)
            temp_product_short = torch.mul(new_h_node_emb, self.hist_attention_weight_short.unsqueeze(0))
            attention_short = softmax(((s_node_emb.unsqueeze(1) - temp_product_short) ** 2).sum(dim=2).neg(), dim=1)
            pref_embedding = (attention_short.unsqueeze(2) * new_h_node_emb).sum(dim=1)
        else:
            pref_embedding = (h_node_emb * (time_duration_ratio * torch.exp(
                torch.neg(delta_interval_weight) * time_interval) * h_time_mask).unsqueeze(2)).sum(dim=1) / (
                                         self.hist_len * 1.)

        if self.use_corr_matrix:
            new_pref_embedding = torch.matmul(pref_embedding, self.corr_matrix)
        else:
            new_pref_embedding = pref_embedding
        p_lambda = ((new_pref_embedding - t_node_emb) ** 2).sum(dim=1).neg()
        n_lambda = ((new_pref_embedding.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        return p_lambda, n_lambda

    def loss_func(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, history_delta, history_length,
                  h_time_mask):
        p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, history_delta,
                                            history_length, h_time_mask)
        loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) \
               - torch.log(torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(dim=1)
        return loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, history_delta, history_length, h_time_mask):
        self.opt.zero_grad()
        loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, history_delta, history_length,
                              h_time_mask)
        loss = loss.sum()
        self.loss += loss.data
        loss.backward()
        self.opt.step()

    def train_and_test(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            if os.name == 'nt':
                loader = DataLoader(self.data_tr, batch_size=self.batch, shuffle=True, num_workers=0)
            else:
                loader = DataLoader(self.data_tr, batch_size=self.batch, shuffle=True, num_workers=self.num_workers)
            for i_batch, sample_batched in enumerate(loader):
                self.is_debug = False

                if torch.cuda.is_available():
                    self.update(sample_batched['source_node'].type(LType).cuda(),
                                sample_batched['target_node'].type(LType).cuda(),
                                sample_batched['target_time'].type(FType).cuda(),
                                sample_batched['neg_nodes'].type(LType).cuda(),
                                sample_batched['history_nodes'].type(LType).cuda(),
                                sample_batched['history_times'].type(FType).cuda(),
                                sample_batched['history_delta'].type(FType).cuda(),
                                sample_batched['history_length'].type(FType).cuda(),
                                sample_batched['history_masks'].type(FType).cuda())
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_delta'].type(FType),
                                sample_batched['history_length'].type(FType),
                                sample_batched['history_masks'].type(FType))

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data_tr)) + '\n')
            sys.stdout.flush()
            if ((epoch + 1) % self.test_and_save_step == 0) or epoch == 0 or epoch == 4:
                self.recommend(epoch, is_new_item=False)
                self.recommend(epoch, is_new_item=True)

    def recommend(self, epoch, is_new_item=False):
        count_all = 0
        rate_all_sum = 0
        recall_all_sum = np.zeros(self.top_n)
        MRR_all_sum = np.zeros(self.top_n)

        if is_new_item:
            loader = DataLoader(self.data_te_new, batch_size=self.batch, shuffle=False, num_workers=self.num_workers)
        else:
            loader = DataLoader(self.data_te_old, batch_size=self.batch, shuffle=False, num_workers=self.num_workers)
        for i_batch, sample_batched in enumerate(loader):
            if torch.cuda.is_available():
                rate_all, recall_all, MRR_all = \
                    self.evaluate(sample_batched['source_node'].type(LType).cuda(),
                                  sample_batched['target_node'].type(LType).cuda(),
                                  sample_batched['target_time'].type(FType).cuda(),
                                  sample_batched['history_nodes'].type(LType).cuda(),
                                  sample_batched['history_times'].type(FType).cuda(),
                                  sample_batched['history_delta'].type(FType).cuda(),
                                  sample_batched['history_length'].type(FType).cuda(),
                                  sample_batched['history_masks'].type(FType).cuda())
            else:
                rate_all, recall_all, MRR_all = \
                    self.evaluate(sample_batched['source_node'].type(LType),
                                  sample_batched['target_node'].type(LType),
                                  sample_batched['target_time'].type(FType),
                                  sample_batched['history_nodes'].type(LType),
                                  sample_batched['history_times'].type(FType),
                                  sample_batched['history_delta'].type(FType),
                                  sample_batched['history_length'].type(FType),
                                  sample_batched['history_masks'].type(FType))
            count_all += self.batch
            rate_all_sum += rate_all
            recall_all_sum += recall_all
            MRR_all_sum += MRR_all

        rate_all_sum_avg = rate_all_sum * 1. / count_all
        recall_all_avg = recall_all_sum * 1. / count_all
        MRR_all_avg = MRR_all_sum * 1. / count_all
        if is_new_item:
            logging.info('=========== testing next new item epoch: {} ==========='.format(epoch))
            logging.info('count_all_next_new: {}'.format(count_all))
            logging.info('rate_all_sum_avg_next_new: {}'.format(rate_all_sum_avg))
            logging.info('recall_all_avg_next_new: {}'.format(recall_all_avg))
            logging.info('MRR_all_avg_next_new: {}'.format(MRR_all_avg))
        else:
            logging.info('~~~~~~~~~~~~~ testing next item epoch: {} ~~~~~~~~~~~~~'.format(epoch))
            logging.info('count_all_next: {}'.format(count_all))
            logging.info('rate_all_sum_avg_next: {}'.format(rate_all_sum_avg))
            logging.info('recall_all_avg_next: {}'.format(recall_all_avg))
            logging.info('MRR_all_avg_next: {}'.format(MRR_all_avg))

    def evaluate(self, s_nodes, t_nodes, t_times, h_nodes, h_times, history_delta, history_length, h_time_mask):
        batch = s_nodes.size()[0]
        all_item_index = torch.arange(0, self.item_count)
        if torch.cuda.is_available():
            all_item_index = all_item_index.cuda()
        all_node_emb = torch.index_select(self.node_emb, 0, all_item_index).detach()
        h_node_emb = torch.index_select(self.node_emb, 0, h_nodes.view(-1)).detach().view(batch, self.hist_len, -1)
        self.delta_interval_weight.data.clamp_(min=1e-6)
        time_interval = torch.abs(t_times.unsqueeze(1) - h_times)
        delta_interval_weight = torch.index_select(self.delta_interval_weight, 0, s_nodes.view(-1)).detach().unsqueeze(
            1)
        s_node_emb = torch.index_select(self.node_emb, 0, s_nodes.view(-1)).detach().view(batch, -1)
        time_duration_ratio = torch.ones((batch, self.hist_len), dtype=torch.float)
        if torch.cuda.is_available():
            time_duration_ratio = time_duration_ratio.cuda()
        if self.use_duration:
            self.delta_duration_weight.data.clamp_(min=1e-6)
            delta_duration_weight = torch.index_select(self.delta_duration_weight, 0,
                                                       s_nodes.view(-1)).detach().unsqueeze(1).expand(batch,
                                                                                                      self.hist_len)
            great_equal_index = torch.ge(history_delta, history_length)
            less_than_index = torch.lt(history_delta, history_length)
            time_interval[great_equal_index] = time_interval[great_equal_index] - history_length[great_equal_index]
            time_interval[less_than_index] = time_interval[less_than_index] - history_delta[less_than_index]
            time_interval = torch.abs(time_interval) + 1e-6
            time_duration_ratio[less_than_index] = torch.exp(
                torch.neg(delta_duration_weight[less_than_index]) * (
                        (history_length[less_than_index] - history_delta[less_than_index]) / history_length[
                    less_than_index]))
        h_index = self.hist_len - 1
        if self.use_hist_attention:
            temp_product_long = torch.mul(h_node_emb[:, :h_index, :],
                                          self.hist_attention_weight_long.detach().unsqueeze(0))
            attention_long = softmax(((s_node_emb.unsqueeze(1) - temp_product_long) ** 2).sum(dim=2).neg(), dim=1)
            aggre_hist_node_emb = (attention_long.unsqueeze(2) * h_node_emb[:, :h_index, :] * (
                    time_duration_ratio[:, :h_index] * torch.exp(
                torch.neg(delta_interval_weight) * time_interval[:, :h_index]) * h_time_mask[:, :h_index]).unsqueeze(
                2)).sum(dim=1)
            curr_emb = h_node_emb[:, h_index, :] * (
                    time_duration_ratio[:, h_index].unsqueeze(1) * torch.exp(
                torch.neg(delta_interval_weight) * time_interval[:, h_index].unsqueeze(1)) * h_time_mask[:,
                                                                                             h_index].unsqueeze(1))
            new_h_node_emb = torch.cat([aggre_hist_node_emb.unsqueeze(1), curr_emb.unsqueeze(1)], dim=1)
            temp_product_short = torch.mul(new_h_node_emb, self.hist_attention_weight_short.detach().unsqueeze(0))
            attention_short = softmax(((s_node_emb.unsqueeze(1) - temp_product_short) ** 2).sum(dim=2).neg(), dim=1)
            pref_embedding = (attention_short.unsqueeze(2) * new_h_node_emb).sum(dim=1)
        else:
            pref_embedding = (h_node_emb * (time_duration_ratio * torch.exp(
                torch.neg(delta_interval_weight) * time_interval) * h_time_mask).unsqueeze(2)).sum(dim=1) / (
                                         self.hist_len * 1.)

        if self.use_corr_matrix:
            new_pref_embedding = torch.matmul(pref_embedding, self.corr_matrix.detach())
        else:
            new_pref_embedding = pref_embedding
        new_pref_embedding_norm = (new_pref_embedding ** 2).sum(1).view(batch, 1)
        all_node_emb_norm = (all_node_emb ** 2).sum(1).view(1, self.item_count)
        p_lambda = (new_pref_embedding_norm + all_node_emb_norm - 2.0 * torch.matmul(new_pref_embedding,
                                                                                     torch.transpose(all_node_emb, 0,
                                                                                                     1))).neg()
        rate_all_sum = 0
        recall_all = np.zeros(self.top_n)
        MRR_all = np.zeros(self.top_n)
        t_nodes_list = t_nodes.cpu().numpy().tolist()
        p_lambda_numpy = p_lambda.cpu().numpy()
        for i in range(len(t_nodes_list)):
            t_node = t_nodes_list[i]
            p_lambda_numpy_i_item = p_lambda_numpy[i]
            prob_index = np.argsort(-p_lambda_numpy_i_item).tolist()
            gnd_rate = prob_index.index(t_node) + 1
            rate_all_sum += gnd_rate
            if gnd_rate <= self.top_n:
                recall_all[gnd_rate - 1:] += 1
                MRR_all[gnd_rate - 1:] += 1. / gnd_rate
        return rate_all_sum, recall_all, MRR_all
