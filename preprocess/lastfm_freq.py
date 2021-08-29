# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import pickle
import os

import numpy as np
import copy

def get_item_lengths_list(data):
    item_length_list_dict = dict()
    user_group = data.groupby(['user_id'])
    for user_id, length in user_group.size().sort_values().iteritems():
        temp_user_data = user_group.get_group(user_id)
        temp_time_seq = copy.deepcopy(pd.to_datetime(temp_user_data['timestamp']))
        temp_user_data.loc[:, 'timestamp_new'] = temp_time_seq
        user_data = temp_user_data.sort_values(by='timestamp_new')
        music_seq = user_data['tran_name_id']
        time_seq = user_data['timestamp_new']
        time_seq = time_seq[music_seq.notnull()]
        music_seq = music_seq[music_seq.notnull()]
        delta_time = time_seq.diff(-1).astype('timedelta64[s]') * -1
        item_seq = music_seq.tolist()

        delta_time = delta_time.tolist()
        delta_time[-1] = 0
        for item, time in zip(item_seq, delta_time):
            time = int(time)
            if time <= 0 or time >= 3600:
                continue
            if item in item_length_list_dict:
                length_list_temp = item_length_list_dict.get(item)
            else:
                length_list_temp = []
            length_list_temp.append(time)
            item_length_list_dict[item] = length_list_temp
    return item_length_list_dict


def get_item_length(item_length_list_dict, item2index):
    item_length = dict()
    for item in item_length_list_dict.keys():
        if item in item2index:
            length_list = item_length_list_dict[item]
            max_length = max(length_list, key=length_list.count)
            if length_list.count(max_length) >= 2 and max_length < 3600:
                item_length[item] = max_length
            else:
                print(item + ": ")
                print(length_list)
    return item_length


def generate_data(top_n_music, top_n_user, min_length, max_length, data, BASE_DIR, DATA_SOURCE):
    train_file_name = os.path.join(BASE_DIR, DATA_SOURCE, 'train.lst')
    test_file_name = os.path.join(BASE_DIR, DATA_SOURCE, 'test.lst')

    index2item_path = os.path.join(BASE_DIR, DATA_SOURCE, 'last_music_index2item')
    item2index_path = os.path.join(BASE_DIR, DATA_SOURCE, 'last_music_item2index')
    index2length_path = os.path.join(BASE_DIR, DATA_SOURCE, 'last_music_index2length')

    item_length_path = os.path.join(BASE_DIR, DATA_SOURCE, 'index2length')
    item_length_list_dict_path = os.path.join(BASE_DIR, DATA_SOURCE, 'last_music_item_length_list_dict')

    out_tr_uit = open(train_file_name, 'w', encoding='utf-8')
    out_te_uit = open(test_file_name, 'w', encoding='utf-8')

    if os.path.exists(index2item_path) and os.path.exists(item2index_path):
        index2item = pickle.load(open(index2item_path, 'rb'))
        item2index = pickle.load(open(item2index_path, 'rb'))
        print('Total music and user %d' % len(index2item))
    else:
        print('Build users index')
        sorted_user_series = data.groupby(['user_id']).size().sort_values(ascending=False)
        print('sorted_user_series size is: {}'.format(len(sorted_user_series)))
        user_index2item = sorted_user_series.head(top_n_user).keys().tolist()

        print('Build index2item')
        sorted_item_series = data.groupby(['tran_name_id']).size().sort_values(ascending=False)
        print('sorted_item_series size is: {}'.format(len(sorted_item_series)))
        item_index2item = sorted_item_series.head(top_n_music).keys().tolist()
        print('item_index2item size is: {}'.format(len(item_index2item)))

        index2item = item_index2item + user_index2item
        print('index2item size is: {}'.format(len(index2item)))
        print('build item2index')
        item2index = dict((v, i) for i, v in enumerate(index2item))

        pickle.dump(index2item, open(index2item_path, 'wb'))
        pickle.dump(item2index, open(item2index_path, 'wb'))

    if os.path.exists(item_length_list_dict_path):
        item_length_list_dict = pickle.load(open(item_length_list_dict_path, 'rb'))
        if os.path.exists(item_length_path):
            item_length = pickle.load(open(item_length_path, 'rb'))
        else:
            print('Build item_length')
            item_length = get_item_length(item_length_list_dict, item2index)
            pickle.dump(item_length, open(item_length_path, 'wb'))
    else:
        print('Build item_length_list_dict and item_length')
        item_length_list_dict = get_item_lengths_list(data)
        item_length = get_item_length(item_length_list_dict, item2index)

        pickle.dump(item_length, open(item_length_path, 'wb'))
        pickle.dump(item_length_list_dict, open(item_length_list_dict_path, 'wb'))

    if not os.path.exists(index2length_path):
        print('Build index2length')
        index2length_file = open(index2length_path, 'w', encoding='utf-8')
        for item in index2item:
            if item in item_length:
                index2length_file.write(str(item_length[item]) + '\n')
        index2length_file.close()

    print('start loop')
    count = 0
    valid_user_count = 0
    user_group = data.groupby(['user_id'])
    total = len(user_group)
    for user_id, length in user_group.size().sort_values().iteritems():

        if count % 100 == 0:
            print("=====count %d/%d======" % (count, total))
            print('%s %d' % (user_id, length))
        count += 1

        if user_id not in item2index:
            continue

        temp_user_data = user_group.get_group(user_id)
        old_time_seq = copy.deepcopy(pd.to_datetime(temp_user_data['timestamp']))
        temp_user_data.loc[:, 'timestamp_new'] = old_time_seq
        user_data = temp_user_data.sort_values(by='timestamp_new')
        music_seq = user_data['tran_name_id']
        time_seq = user_data['timestamp_new']
        time_seq = time_seq[music_seq.notnull()]
        time_seq_list = time_seq.tolist()
        music_seq = music_seq[music_seq.notnull()]

        delta_time = time_seq.diff(-1).astype('timedelta64[s]') * -1
        item_seq_list = music_seq.apply(lambda x: (item2index[x]) if pd.notnull(x) and x in item2index else -1).tolist()

        delta_time_list = delta_time.tolist()
        delta_time_list[-1] = 0

        length_time_list = music_seq.apply(
            lambda x: (item_length[x]) if pd.notnull(x) and x in item_length else -1).tolist()

        valid_index = [0]
        need_record = False
        temp_sum = 0
        for i in range(1, len(item_seq_list)):
            if item_seq_list[i] != item_seq_list[i - 1]:
                valid_index.append(i)
                temp_sum = delta_time_list[i]
                if temp_sum >= length_time_list[i]:
                    need_record = True
                else:
                    need_record = False
            else:
                if need_record:
                    valid_index.append(i)
                    temp_sum = delta_time_list[i]
                else:
                    temp_sum += delta_time_list[i]

                if temp_sum >= length_time_list[i]:
                    need_record = True
                else:
                    need_record = False

        temp_item_seq = []
        temp_length_time = []
        temp_time_seq = []
        for i in range(len(valid_index)):
            index = valid_index[i]
            temp_item_seq.append(item_seq_list[index])
            temp_length_time.append(length_time_list[index])
            temp_time_seq.append(time_seq_list[index])
        temp_delta_time = (pd.Series(temp_time_seq).diff(-1).astype('timedelta64[s]') * -1).tolist()
        temp_delta_time.append(0)

        temp_delta_time = np.array(temp_delta_time) / 3600
        temp_length_time = np.array(temp_length_time) / 3600

        time_accumulate = [0]
        for delta in temp_delta_time[:-1]:
            next_time = time_accumulate[-1] + delta
            time_accumulate.append(next_time)

        new_item_seq = []
        new_time_accumulate = []
        new_length_time = []
        new_delta_time = []
        valid_count = 0
        for i in range(len(temp_item_seq)):
            if temp_item_seq[i] != -1 and temp_delta_time[i] != 0.0 and temp_delta_time[i] != -0.0:
                new_item_seq.append(temp_item_seq[i])
                new_time_accumulate.append(time_accumulate[i])
                new_length_time.append(temp_length_time[i])
                new_delta_time.append(temp_delta_time[i])
                valid_count += 1
            if valid_count >= max_length:
                break

        if len(new_item_seq) < min_length:
            continue
        else:
            valid_user_count += 1
            user_index = item2index[user_id]
            index_hash_remaining = user_index % 10
            if index_hash_remaining < 2:
                half_index = int(len(new_item_seq) / 2)
                for i in range(half_index):
                    out_tr_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(
                            new_delta_time[i]) + '\t' + str(new_length_time[i]) + '\n')
                for i in range(half_index, int(len(new_item_seq))):
                    out_te_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(
                            new_delta_time[i]) + '\t' + str(new_length_time[i]) + '\n')

            else:
                for i in range(len(new_item_seq)):
                    out_tr_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\t' + str(
                            new_delta_time[i]) + '\t' + str(new_length_time[i]) + '\n')

    print("valid_user_count is: {}".format(valid_user_count))
    out_tr_uit.close()
    out_te_uit.close()


if __name__ == '__main__':
    BASE_DIR = ''
    DATA_SOURCE = 'last_music'

    min_length = 100
    max_length = 1500
    top_n_user = 900
    top_n_music = 66407
    path = os.path.join(BASE_DIR, DATA_SOURCE, 'userid-timestamp-artid-artname-traid-traname.tsv')

    print("start reading csv")
    data = pd.read_csv(path, sep='\t',
                       error_bad_lines=False,
                       header=None,
                       names=['user_id', 'timestamp', 'artid', 'artname', 'traid', 'tranname'],
                       quotechar=None, quoting=3)
    print("finish reading csv")
    data['tran_name_id'] = data['tranname'] + data['traid']
    data['art_name_id'] = data['artname'] + data['artid']
    generate_data(top_n_music, top_n_user, min_length, max_length, data, BASE_DIR, DATA_SOURCE)
