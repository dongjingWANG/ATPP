# -*- coding: utf-8 -*-
import os
import logging
from Model import ATPP
import config

FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# DID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda

if __name__ == '__main__':

    dataset = ['last_music']
    data_index = 0
    NORM_METHOD = 'hour'
    print('NORM_METHOD: {}'.format(NORM_METHOD))

    # change according to specific dataset
    min_length = 100
    max_length = 1500
    top_n_user = 000  # how many users
    top_item_count = 000  # how many item
    train_path = './preprocess/{}/train.lst'.format(dataset[data_index])
    test_old_path = './preprocess/{}/test.lst'.format(dataset[data_index])
    item_length_path = './preprocess/{}/index2length'.format(dataset[data_index])

    htne = ATPP(dataset[data_index], train_path, test_old_path,
                item_length_path,
                emb_size=config.emb_size, neg_size=config.neg_size,
                hist_len=config.hist_len,
                user_count=top_n_user, item_count=top_item_count,
                directed=True,
                learning_rate=config.learning_rate,
                decay=config.decay, batch_size=config.batch_size,
                test_and_save_step=config.test_and_save_step,
                epoch_num=config.epoch_num, top_n=30,
                use_hist_attention=True, use_duration=True, use_corr_matrix=True,
                num_workers=8, norm_method=NORM_METHOD)

    htne.train_and_test()
