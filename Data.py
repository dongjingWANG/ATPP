# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import numpy as np

class DataSetTrain(Dataset):

    def __init__(self, train_path, item_length_path, user_count=0, item_count=0, neg_size=5, hist_len=2,
                 directed=False):
        self.neg_size = neg_size
        self.user_count = user_count
        self.item_count = item_count
        self.hist_len = hist_len
        self.directed = directed

        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)

        self.node2hist = dict()
        self.user_item_dict = dict()
        self.user_node_set = set()
        self.item_node_set = set()
        self.degrees = dict()
        with open(train_path, 'r') as infile:
            for line in infile:
                parts = line.strip().split()
                s_node = int(parts[0])
                t_node = int(parts[1])
                time_stamp = float(parts[2])
                interval_time = float(parts[3])
                length_time = float(parts[4])
                if s_node not in self.user_item_dict:
                    self.user_item_dict[s_node] = set()
                self.user_item_dict[s_node].add(t_node)

                self.user_node_set.add(s_node)
                self.item_node_set.add(t_node)

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, time_stamp, interval_time, length_time))

                if not directed:
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, time_stamp, interval_time, length_time))

                if s_node not in self.degrees:
                    self.degrees[s_node] = 0
                if t_node not in self.degrees:
                    self.degrees[t_node] = 0
                self.degrees[s_node] += 1
                self.degrees[t_node] += 1

        self.node_dim = self.user_count + self.item_count

        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

        length_file = open(item_length_path, 'r', encoding='utf-8').readlines()
        self.item_index_length = []
        for line in length_file:
            duration = int(line.strip())
            self.item_index_length.append(duration)

        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()

    def get_node_dim(self):
        return self.node_dim


    def init_neg_table(self):
        total_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        for k in range(self.node_dim):
            if k in self.degrees:
                total_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
            else:
                self.degrees[k] = 0
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                while self.degrees[n_id] == 0:
                    n_id += 1
                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / total_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]

        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]
        hist_delta = [h[2] for h in hist]
        hist_length = [h[3] for h in hist]

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_nodes)] = hist_times
        np_h_delta = np.zeros((self.hist_len,))
        np_h_delta[:len(hist_nodes)] = hist_delta
        np_h_length = np.zeros((self.hist_len,))
        np_h_length[:len(hist_nodes)] = hist_length
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.

        neg_nodes = self.negative_sampling(s_node, t_node)

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_delta': np_h_delta,
            'history_length': np_h_length,
            'history_masks': np_h_masks,
            'neg_nodes': neg_nodes,
        }
        return sample

    def negative_sampling(self, source_node, target_node):
        sampled_nodes = []
        func = lambda: self.neg_table[np.random.randint(0, self.neg_table_size)]
        for i in range(self.neg_size):
            temp_neg_node = func()
            while temp_neg_node in self.user_item_dict[source_node] or temp_neg_node == source_node or temp_neg_node == target_node or temp_neg_node >= self.item_count:
                temp_neg_node = func()
            sampled_nodes.append(temp_neg_node)
        return np.array(sampled_nodes)


class DataSetTestNext(Dataset):

    def __init__(self, file_path, data_tr, user_count=0, item_count=0, hist_len=2, directed=False):
        self.user_count = user_count
        self.item_count = item_count
        self.hist_len = hist_len
        self.directed = directed
        self.node2hist = dict()
        with open(file_path, 'r') as infile:
            for line in infile:
                parts = line.split()
                s_node = int(parts[0])
                t_node = int(parts[1])
                time_stamp = float(parts[2])
                interval_time = float(parts[3])
                length_time = float(parts[4])

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, time_stamp, interval_time, length_time))

                if not directed:
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, time_stamp, interval_time, length_time))

        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]

        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]
        hist_delta = [h[2] for h in hist]
        hist_length = [h[3] for h in hist]

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_delta = np.zeros((self.hist_len,))
        np_h_delta[:len(hist_nodes)] = hist_delta
        np_h_length = np.zeros((self.hist_len,))
        np_h_length[:len(hist_nodes)] = hist_length

        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_delta': np_h_delta,
            'history_length': np_h_length,
            'history_masks': np_h_masks,
        }

        return sample


class DataSetTestNextNew(Dataset):

    def __init__(self, file_path, data_tr, user_count=0, item_count=0, hist_len=2, directed=False):
        self.user_count = user_count
        self.item_count = item_count
        self.hist_len = hist_len
        self.user_item_dict = data_tr.user_item_dict
        self.directed = directed
        self.node2hist = dict()
        with open(file_path, 'r') as infile:
            for line in infile:
                parts = line.split()
                s_node = int(parts[0])
                t_node = int(parts[1])
                time_stamp = float(parts[2])
                interval_time = float(parts[3])
                length_time = float(parts[4])

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, time_stamp, interval_time, length_time))

                if not directed:
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, time_stamp, interval_time, length_time))

        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            s_node_hist = self.node2hist[s_node]
            s_node_current_hist_set = set(self.user_item_dict[s_node])
            for t_idx in range(len(s_node_hist)):
                if s_node_hist[t_idx][0] not in s_node_current_hist_set:
                    self.idx2source_id[idx] = s_node
                    self.idx2target_id[idx] = t_idx
                    idx += 1
                    s_node_current_hist_set.add(s_node_hist[t_idx][0])
        self.data_size = idx
        self.idx2source_id = self.idx2source_id[:self.data_size]
        self.idx2target_id = self.idx2target_id[:self.data_size]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]

        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]
        hist_delta = [h[2] for h in hist]
        hist_length = [h[3] for h in hist]

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_delta = np.zeros((self.hist_len,))
        np_h_delta[:len(hist_nodes)] = hist_delta
        np_h_length = np.zeros((self.hist_len,))
        np_h_length[:len(hist_nodes)] = hist_length
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_delta': np_h_delta,
            'history_length': np_h_length,
            'history_masks': np_h_masks,
        }

        return sample
