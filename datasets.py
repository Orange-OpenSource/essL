# Software Name : essl
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

"""
Download datasets, uncompress, and store them under data/ local folder

[1] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/scripts/get_librispeech_data.py
"""

import argparse
import fnmatch
import json
import logging
import os
import glob
import random
import subprocess
import tarfile
import urllib.request
import numpy as np
import soundfile as sf

from sox import Transformer
from tqdm import tqdm
import webdataset as wds

import torch
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.processing.features import (
    STFT, 
    Filterbank,
    spectral_magnitude, 
    InputNormalization 
)
from modules.perturb import (
    SpecAugment,
    RandomShift,
    RandomNoisePerturbation
)
from config.efficientssl import (
    EfficientSSLConfig
) 

URLS = {
    # LibriSpeech
    'TRAIN_CLEAN_100': ("http://www.openslr.org/resources/12/train-clean-100.tar.gz"),
    'TRAIN_CLEAN_360': ("http://www.openslr.org/resources/12/train-clean-360.tar.gz"),
    'TRAIN_OTHER_500': ("http://www.openslr.org/resources/12/train-other-500.tar.gz"),
    'DEV_CLEAN': "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    'DEV_OTHER': "http://www.openslr.org/resources/12/dev-other.tar.gz",
    'TEST_CLEAN': "http://www.openslr.org/resources/12/test-clean.tar.gz",
    'TEST_OTHER': "http://www.openslr.org/resources/12/test-other.tar.gz",

    # MiniLibriSpeech
    'DEV_CLEAN_2': "http://www.openslr.org/resources/31/dev-clean-2.tar.gz",
    'TRAIN_CLEAN_5': "http://www.openslr.org/resources/31/train-clean-5.tar.gz",
}

def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    Returns:
    """
    source = URLS[source]
    if not os.path.exists(destination):
        logging.info("{0} does not exist. Downloading ...".format(destination))
        urllib.request.urlretrieve(source, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        logging.info("Downloaded {0}.".format(destination))
    else:
        logging.info("Destination {0} exists. Skipping.".format(destination))
    return destination


def __extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info('Not extracting. Maybe already there?')


def __process_data(data_folder: str, dst_folder: str, manifest_file: str):
    """
    Converts flac to wav and build manifests's json in SpeechBrain format
    Args:
        data_folder: source with flac files
        dst_folder: where wav files will be stored
        manifest_file: where to store manifest
    Returns:
    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = []
    entries = {}

    for root, dirnames, filenames in os.walk(data_folder):
        for filename in fnmatch.filter(filenames, '*.trans.txt'):
            files.append((os.path.join(root, filename), root))

    for transcripts_file, root in tqdm(files):
        with open(transcripts_file, encoding="utf-8") as fin:
            for line in fin:
                id, text = line[: line.index(" ")], line[line.index(" ") + 1 :]
                transcript_text = text.lower().strip()

                # Convert FLAC file to WAV
                flac_file = os.path.join(root, id + ".flac")
                wav_file = os.path.join(dst_folder, id + ".wav")
                if not os.path.exists(wav_file):
                    Transformer().build(flac_file, wav_file)
                # check duration
                duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)

                entry = {}
                entry['file_path'] = os.path.abspath(wav_file)
                entry['words'] = transcript_text
                entry['spkID'] = id.split("-")[0]
                entry['length'] = float(duration)
                entries[id] = entry

    with open(manifest_file, 'w') as fout:
        json.dump(entries, fout, indent=4)

def get_librispeech(data_root, data_sets="all"):
    """
    Downloads LibriSpeech data and process it into WAV files, generating
    a manifest in JSON format

    Args:
        data_root: local folder
        data_sets: LibriSpeech dataset, default is all 
    Returns:
    """
    json_all = ""
    librispeech = os.path.join(data_root, "LibriSpeech")
    if data_sets == "all":
        data_sets = "dev_clean,dev_other,train_clean_100,train_clean_360,train_other_500,test_clean,test_other,dev_clean_2,train_clean_5"
        json_all = "all_train"

    for data_set in data_sets.split(','):
        logging.info("\n\nWorking on: {0}".format(data_set))
        filepath = os.path.join(data_root, data_set + ".tar.gz")
        logging.info("Getting {0}".format(data_set))
        __maybe_download_file(filepath, data_set.upper())
        logging.info("Extracting {0}".format(data_set))
        __extract_file(filepath, data_root)
        logging.info("Processing {0}".format(data_set))
        __process_data(
            os.path.join(librispeech, data_set.replace("_", "-")),
            os.path.join(librispeech, data_set.replace("_", "-")) +"-processed",
            os.path.join(librispeech, data_set + ".json")
        )

    # JSON for all 960h of training data 
    if json_all == "all_train":
        data_all = {}
        data_sets = "all_train"
        manifest_file = os.path.join(librispeech, data_sets + ".json")
        files = glob.glob(os.path.join(librispeech, "train*.json"))
        for file in files:
            j = json.load(open(file))
            data_all.update(j)
        with open(manifest_file, 'w') as fout:
            json.dump(data_all, fout, indent=4)

    logging.info('Done!')

def load_librispeech(data_root, data_sets, max_batch_length, 
    dataset_name="LibriSpeech") -> DataLoader:
    """
    Load audio signals from LibriSpeech dataset using SpeechBrain data pipeline 
    and dynamic batching. 

    Args:
        data_root: local folder
        data_sets: LibriSpeech dataset. all_train loads 960h of speech data 
        from all the training datasets
        with 960h of
    Returns:
        dataloader: pytorch dataloader with dynamic batching

    [1] https://colab.research.google.com/drive/1mypqbHDrusZaIbqPoiEGY-WIbnpMHa2I?usp=sharing#scrollTo=FrsjDadz0AP_
    [2] https://colab.research.google.com/drive/1AiVJZhZKwEI4nFGANKXEe-ffZFfvXKwH?usp=sharing
    """
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("signal")
    def audio_pipeline(file_path):
        config = EfficientSSLConfig()
        fft = STFT(
            sample_rate=config.sample_rate, win_length=config.win_length,
            hop_length=config.hop_length, n_fft=config.n_fft)
        compute_fbanks = Filterbank(
            n_mels=config.n_mels, log_mel=config.log_mel, n_fft=config.n_fft)
        
        signal = sb.dataio.dataio.read_audio(file_path)
        signal = signal.view(1, -1)
        signal = fft(signal)
        signal = spectral_magnitude(signal)
        signal = compute_fbanks(signal)
        signal = signal.squeeze()
        logging.debug('Pipeline signal shape {0}'.format(signal.shape))
        return signal
    
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("signal_noise")
    def audio_pipeline_numpy(file_path):
        """
        Add random noise from DNS 2021 challenge

        [1] https://github.com/microsoft/DNS-Challenge
        [2] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/examples/asr/conf/spiral/spiral_base_pretrain_ls960.py
        """
        rng = random.Random()
        config = EfficientSSLConfig()
        fft = STFT(
            sample_rate=config.sample_rate, win_length=config.win_length,
            hop_length=config.hop_length, n_fft=config.n_fft)
        signal_noise, _ = sf.read(file_path, dtype="float32")
        compute_fbanks = Filterbank(
            n_mels=config.n_mels, log_mel=config.log_mel, n_fft=config.n_fft)
        
        if rng.random() < config.noise_ratio: 
            dns_noises_csv = os.path.join(
                config.data_dir, "datasets", "noises.csv")
            noiseaug = RandomNoisePerturbation(
                manifest_path=[dns_noises_csv],
                min_snr_db=0.,
                max_snr_db=30.,
                ratio=config.noise_ratio,
                target_sr=config.sample_rate,
                data_dir='',
                cache_noise=False, # True significantly slows down training
            )
            signal_noise = noiseaug.perturb_np(signal_noise)

        signal_noise = torch.from_numpy(signal_noise)
        signal_noise = signal_noise.view(1, -1)
        signal_noise = fft(signal_noise)
        signal_noise = spectral_magnitude(signal_noise)
        signal_noise = compute_fbanks(signal_noise)
        signal_noise = signal_noise.squeeze()
        logging.debug(
            'Pipeline signal noise shape {0}'.format(signal_noise.shape))
        return signal_noise

    dataset = DynamicItemDataset.from_json(json_path = os.path.join(
        os.path.join(data_root, dataset_name), data_sets + ".json"))
    dataset.add_dynamic_item(audio_pipeline)
    sb.dataio.dataset.add_dynamic_item([dataset], audio_pipeline_numpy) 
    dataset.set_output_keys(
        ["id", "signal", "signal_noise", "words", "length"])

    dynamic_batcher = DynamicBatchSampler(
        dataset,
        max_batch_length=max_batch_length, 
        num_buckets=60,
        length_func=lambda x: x["length"],
        shuffle=False,
        batch_ordering="random",
        ) 

    dataloader = DataLoader(
        dataset, 
        batch_sampler=dynamic_batcher, 
        collate_fn=PaddedBatch,
        num_workers=4
        )
    
    return dataloader