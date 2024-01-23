# Software Name : essl
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

import argparse
import logging
import datasets, models

from config.efficientssl import (
    EfficientSSLConfig,
    DATA_DIR,
    LOG_FILE,
    CHECKPT,
    )

def efficient_train(args):
    config = EfficientSSLConfig()
    config.data_dir = args.data_root
    config.freeze_finetuning = False
    config.pretraining=True

    librispeech_train_all = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='all_train', 
        max_batch_length=config.batch_length)
    librispeech_train_100 = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='train_clean_100', max_batch_length=config.batch_length)
    librispeech_dev_other = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='dev_other', 
        max_batch_length=config.batch_length)
    librispeech_dev_clean = datasets.load_librispeech(
        data_root=config.data_dir, data_sets='dev_clean',
        max_batch_length=config.batch_length)
    
    model = models.EfficientSSL(config)
    logging.info(
        '\n\n--Experiment with efficient SSL freeze {0} pretraining {1}'.format(
        model.freeze_finetuning, model.pretraining))
    model.train_model(
        pretrain_dataloader=librispeech_train_all,
        finetune_dataloader=librispeech_train_100,
        val_dataloader=librispeech_dev_other)
    logging.info('Testing with LibriSpeech dev clean ...')
    model.test_model(test_dataloader=librispeech_dev_clean)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Efficient SSL model')
    parser.add_argument("--data_root", default=DATA_DIR, type=str)
    parser.add_argument("--data_sets", default="all_train", type=str)
    parser.add_argument("--log_file", default=LOG_FILE, type=str)
    parser.add_argument("--run", required=True, default=None, type=str)
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file, format='%(asctime)s %(levelname)s %(message)s', 
        filemode='w', level=logging.INFO)
    
    if args.run == "download":
        datasets.get_librispeech(args.data_root, args.data_sets)
    elif args.run == "train":
        efficient_train(args)