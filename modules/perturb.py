# Software Name : essl
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

"""
Audio perturbations using specaug, noise, and random shifts

[1] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/parts/perturb.py
[2] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/parts/spectr_augment.py
[3] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/models/st2vec/st2vec_model.py

"""

import os
import logging
import random
import pandas
import librosa
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from typing import Optional, List, Any
from dataclasses import field, dataclass
from omegaconf import MISSING

from config.block import ShiftPerturbConfig, NoisePerturbConfig

class SpecAugment(nn.Module):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).
    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment.
        Can be a positive integer or a float value in the range [0, 1].
        If positive integer value, defines maximum number of time steps
        to be cut in one segment.
        If a float value, defines maximum percentage of timesteps that
        are cut adaptively.
    """

    def __init__(
        self, freq_masks=0, time_masks=0, freq_width=10, time_width=10, max_time_masks=20, gauss_mask_std=0.0, rng=None,
    ):
        super(SpecAugment, self).__init__()

        self._rng = random.Random() if rng is None else rng

        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.max_time_masks = max_time_masks

        self.freq_width = freq_width
        self.time_width = time_width

        self.gauss_mask_std = gauss_mask_std

        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError('If `time_width` is a float value, must be in range [0, 1]')

            self.adaptive_temporal_width = True

        if isinstance(time_masks, int):
            self.adaptive_time_mask = False
        else:
            if time_masks >= 1.0 or time_masks < 0.0:
                raise ValueError('If `time_width` is a float value, must be in range [0, 1]')

            self.adaptive_time_mask = True

    @torch.no_grad()
    def forward(self, x, length):
        B, D, T = x.shape

        for idx in range(B):
            for _ in range(self.freq_masks):
                x_left = self._rng.randint(0, D - self.freq_width)

                w = self._rng.randint(0, self.freq_width)

                x[idx, x_left : x_left + w, :] = 0.0

            if self.adaptive_temporal_width:
                time_width = max(1, int(length[idx] * self.time_width))
            else:
                time_width = self.time_width

            if self.adaptive_time_mask:
                time_masks = int(length[idx] * self.time_masks)
                time_masks = min(time_masks, self.max_time_masks)
            else:
                time_masks = self.time_masks

            for _ in range(time_masks):
                y_left = self._rng.randint(0, length[idx] - time_width)

                w = self._rng.randint(0, time_width)

                if self.gauss_mask_std == 0:
                    x[idx, :, y_left:y_left + w] = 0.0
                else:
                    x[idx, :, y_left:y_left + w] = torch.normal(mean=0, std=self.gauss_mask_std, size=(D, w)).to(x.device)

        return x
    
class RandomShift:
    def __init__(self, cfg: ShiftPerturbConfig):
        self.dist = cfg.dist
        if self.dist == 'uniform':
            assert isinstance(cfg.max, int) and isinstance(cfg.min, int)
            self.min = cfg.min
            self.max = cfg.max
        else:
            assert cfg.dist == 'rounded_normal'
            assert isinstance(cfg.mean, float) and isinstance(cfg.std, float)
            self.mean = cfg.mean
            self.std = cfg.std
        self.max_ratio = cfg.max_ratio
        assert isinstance(cfg.unit, int)
        self.unit = cfg.unit
        self.shift_prob = cfg.shift_prob
        self.truncate = cfg.truncate

    def shift(self, inputs, inputs_len, mask_emb):
        if np.random.random() >= self.shift_prob:
            return inputs, inputs_len, 0, 0, 0

        shift_num, shift_num_units, r_shift_num, r_shift_num_units = self.get_shift_num(inputs_len.min())

        if self.truncate and shift_num > 0 and r_shift_num > 0:
            r_shift_num = 0
            r_shift_num_units = 0

        orig_inputs_t = inputs.shape[1]

        if shift_num_units > 0:
            inputs = torch.nn.functional.pad(inputs, (0, 0, shift_num_units, 0))
            inputs[:, :shift_num_units] = mask_emb
            inputs_len = inputs_len + shift_num_units
        elif shift_num_units < 0:
            abs_shift_num_units = abs(shift_num_units)
            inputs = inputs[:, abs_shift_num_units:]
            inputs_len = inputs_len - abs_shift_num_units

        if r_shift_num_units > 0:
            inputs = torch.nn.functional.pad(inputs, (0, 0, 0, r_shift_num_units))
            shift_padding_mask = create_shift_padding_mask(inputs_len, inputs.shape[1], r_shift_num_units)
            inputs[shift_padding_mask] = mask_emb
            inputs_len = inputs_len + r_shift_num_units
        elif r_shift_num_units < 0:
            shift_padding_mask = create_shift_padding_mask(inputs_len, inputs.shape[1], r_shift_num_units)
            inputs[shift_padding_mask] = 0.0
            abs_shift_num_units = abs(r_shift_num_units)
            inputs_len = inputs_len - abs_shift_num_units
            inputs = inputs[:, :-abs_shift_num_units]

        inputs_t_diff = inputs.shape[1] - orig_inputs_t
        if self.truncate and inputs_t_diff > 0:
            truncated_r_shift_num = r_shift_num - int(inputs_t_diff / self.unit)
            assert truncated_r_shift_num == -shift_num
            inputs = inputs[:, :-inputs_t_diff]
            inputs_len = inputs_len - inputs_t_diff
        else:
            truncated_r_shift_num = r_shift_num

        return inputs, inputs_len, shift_num, r_shift_num, truncated_r_shift_num

    def get_shift_num(self, total_units_num):
        if self.dist == 'uniform':
            shift_num = np.random.randint(self.min, self.max + 1)
            r_shift_num = np.random.randint(self.min, self.max + 1)
        else:
            shift_num = np.random.normal(loc=self.mean, scale=self.std)
            shift_num = int(round(shift_num))
            r_shift_num = np.random.normal(loc=self.mean, scale=self.std)
            r_shift_num = int(round(r_shift_num))

        max_num = int(total_units_num * self.max_ratio / self.unit)
        if shift_num > max_num:
            if self.truncate:
                shift_num = max_num
        elif shift_num < -max_num:
            shift_num = -max_num

        if r_shift_num < 0:
            if shift_num > 0:
                r_shift_num = max(-max_num, r_shift_num)
            else:
                r_shift_num = max(-(max_num - abs(shift_num)), r_shift_num)

        return shift_num, shift_num * self.unit, r_shift_num, r_shift_num * self.unit

def create_shift_padding_mask(lengths, max_len, shift_num_units):
    positions = torch.arange(max_len, device=lengths.device)
    positions.expand(len(lengths), max_len)
    shift_audio_lengths = lengths + shift_num_units
    if shift_num_units > 0:
        padding_mask = (positions >= lengths.unsqueeze(1)) & (positions < shift_audio_lengths.unsqueeze(1))
    else:
        padding_mask = (positions >= shift_audio_lengths.unsqueeze(1)) & (positions < lengths.unsqueeze(1))
    return padding_mask

class Perturbation(object):
    def max_augmentation_length(self, length):
        return length

    def perturb(self, data):
        raise NotImplementedError

class RandomNoisePerturbation(Perturbation):
    """
    Perturbation that adds noise to input audio.
    Args:
        manifest_path (str): Manifest file with paths to noise files
        min_snr_db (float): Minimum SNR of audio after noise is added
        max_snr_db (float): Maximum SNR of audio after noise is added
        max_gain_db (float): Maximum gain that can be applied on the noise sample
        audio_tar_filepaths (list) : Tar files, if noise audio files are tarred
        shuffle_n (int): Shuffle parameter for shuffling buffered files from the tar files
        orig_sr (int): Original sampling rate of the noise files
        rng: Random number generator
    """

    def __init__(
        self,
        manifest_path=None,
        min_snr_db=0,
        max_snr_db=30,
        max_gain_db=300.0,
        ratio=0.5,
        rng=None,
        target_sr=16000,
        data_dir='',
        cache_noise=False
    ):

        self._rng = random.Random() if rng is None else rng
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._max_gain_db = max_gain_db
        self.ratio = ratio

        self.data_dir = data_dir
        self.target_sr = target_sr
        manifest, self._noise_weights = self.read_manifest(manifest_path)
        self.noise_files = manifest['wav_filename'].tolist()
        self._cache = {}
        self.cache_noise = cache_noise

    def read_manifest(self, manifest_fps):
        manifest_files = []
        for fp in manifest_fps:
            manifest_files.append(pandas.read_csv(fp, encoding='utf-8'))
        manifest = pandas.concat(manifest_files)

        orig_noise_num = len(manifest)
        wav_header_size = 44
        # only use noise with duration longer than 1 second
        manifest = manifest[manifest['wav_filesize'] > (1 * 16000 * 2 + wav_header_size)]
        logging.debug('filter noise less than 1s: from {} to {} samples'.format(orig_noise_num, len(manifest)))
        wav_data_size = manifest['wav_filesize'].values - wav_header_size
        logging.debug('noise duration sum: {}h'.format(wav_data_size.sum() / (16000 * 2) / 60 / 60))
        noise_weights = wav_data_size / wav_data_size.sum()
        return manifest, noise_weights.tolist()

    @torch.no_grad()
    def perturb_tensor(self, signal):
        """
        https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/parts/perturb.py
        """
        if self._rng.random() < self.ratio:
            B, T = signal.shape
            num_samples = B * T
            signal = signal.view(-1)
            signal_np = signal.cpu().detach().numpy()

            noises = self.get_noises(num_samples)
            snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
            data_rms = rms_db(signal_np)

            noise_sequence = np.array([])
            for noise_i in noises:
                noise_gain_db = min(
                    data_rms - rms_db(noise_i) - snr_db, self._max_gain_db)
                # adjust gain for snr purposes and superimpose
                noise_i = gain_db(noise_i, noise_gain_db)
                noise_sequence = np.append(noise_sequence, noise_i)
                
            noise_sequence = torch.tensor(noise_sequence, device=signal.device)
            assert noise_sequence.shape[0] == signal.shape[0]
            signal = torch.add(signal, noise_sequence)
            signal = signal.view(B, T)
            signal = signal.float()
            
        return signal
    
    def perturb_np(self, signal_np):
        """
        https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/parts/perturb.py
        """
        num_samples = signal_np.size
        noises = self.get_noises(num_samples)
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        data_rms = rms_db(signal_np)

        noises = self.get_noises(num_samples)
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        data_rms = rms_db(signal_np)

        start_index = 0
        for noise_i in noises:
            noise_gain_db = min(
                data_rms - rms_db(noise_i) - snr_db, self._max_gain_db)
            noise_i = gain_db(noise_i, noise_gain_db)

            end_index = start_index + noise_i.shape[0]
            signal_np[start_index:end_index] += noise_i
            start_index = end_index
        assert end_index == num_samples
            
        return signal_np
    
    def perturb(self, data):
        if self._rng.random() < self.ratio:
            noises = self.get_noises(data.num_samples)
            self.perturb_with_input_noise(data, noises)

    def perturb_with_input_noise(self, data, noises):
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        data_rms = rms_db(data._samples)

        start_index = 0
        for noise_i in noises:
            noise_gain_db = min(data_rms - rms_db(noise_i) - snr_db, self._max_gain_db)
            # logging.debug("noise: %s %s %s", snr_db, noise_gain_db, noise_record.audio_file)

            # adjust gain for snr purposes and superimpose
            noise_i = gain_db(noise_i, noise_gain_db)

            end_index = start_index + noise_i.shape[0]
            data._samples[start_index:end_index] += noise_i
            start_index = end_index
        assert end_index == data.num_samples

    def get_noises(self, num_samples):
        left_noise_samples = num_samples
        noise_data = []
        while left_noise_samples > 0:
            noise = self.read_one_noise()
            if noise.shape[0] > left_noise_samples:
                start_pos = self._rng.randrange(0, noise.shape[0] - left_noise_samples + 1)
                noise = noise[start_pos:start_pos + left_noise_samples]
            left_noise_samples -= noise.shape[0]
            noise_data.append(noise)
        assert left_noise_samples == 0
        return noise_data

    def read_one_noise(self):
        fp = self._rng.choices(self.noise_files, weights=self._noise_weights)[0]
        fp = os.path.join(self.data_dir, fp)

        cached_noise = self._cache.get(fp, None)
        if cached_noise is None:
            cached_noise = AudioSegment.from_file(fp, target_sr=self.target_sr)._samples
            if self.cache_noise:
                self._cache[fp] = cached_noise
        return cached_noise.copy()

def rms_db(samples):
    mean_square = np.mean(samples ** 2)
    if mean_square == 0:
        return -np.inf
    else:
        return 10 * np.log10(mean_square)

def gain_db(samples, gain):
    return samples * (10.0 ** (gain / 20.0))

def read_one_audiosegment(
    manifest, target_sr, rng, tarred_audio=False, audio_dataset=None):

    if tarred_audio:
        if audio_dataset is None:
            raise TypeError("Expected augmentation dataset but got None")
        audio_file, file_id = next(audio_dataset)
        manifest_idx = manifest.mapping[file_id]
        manifest_entry = manifest[manifest_idx]

        offset = 0 if manifest_entry.offset is None else manifest_entry.offset
        duration = 0 if manifest_entry.duration is None else manifest_entry.duration

    else:
        audio_record = rng.sample(manifest.data, 1)[0]
        audio_file = audio_record.audio_file
        offset = 0 if audio_record.offset is None else audio_record.offset
        duration = 0 if audio_record.duration is None else audio_record.duration

    return AudioSegment.from_file(audio_file, target_sr=target_sr, offset=offset, duration=duration)

class AudioSegment(object):
    """
    Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.

    [1] https://github.com/huawei-noah/Speech-Backbones/blob/main/SPIRAL/nemo/collections/asr/parts/segment.py
    """

    def __init__(self, samples, sample_rate, target_sr=None, trim=False, trim_db=60, orig_sr=None):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)
        if target_sr is not None and target_sr != sample_rate:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
            sample_rate = target_sr
        if trim:
            samples, _ = librosa.effects.trim(samples, trim_db)
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

        self._orig_sr = orig_sr if orig_sr is not None else sample_rate

    def __eq__(self, other):
        """Return whether two objects are equal."""
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    def __str__(self):
        """Return human-readable representation of segment."""
        return "%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, rms=%.2fdB" % (
            type(self),
            self.num_samples,
            self.sample_rate,
            self.duration,
            self.rms_db,
        )

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= 1.0 / 2 ** (bits - 1)
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    @classmethod
    def from_file(
        cls, audio_file, target_sr=None, int_values=False, offset=0, duration=0, trim=False, orig_sr=None,
    ):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param audio_file: path of file to load
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :return: numpy array of samples
        """
        with sf.SoundFile(audio_file, 'r') as f:
            dtype = 'int32' if int_values else 'float32'
            sample_rate = f.samplerate
            if offset > 0:
                f.seek(int(offset * sample_rate))
            if duration > 0:
                samples = f.read(int(duration * sample_rate), dtype=dtype)
            else:
                samples = f.read(dtype=dtype)

        samples = samples.transpose()
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim, orig_sr=orig_sr)

    @classmethod
    def segment_from_file(cls, audio_file, target_sr=None, n_segments=0, trim=False, orig_sr=None):
        """Grabs n_segments number of samples from audio_file randomly from the
        file as opposed to at a specified offset.
        Note that audio_file can be either the file path, or a file-like object.
        """
        with sf.SoundFile(audio_file, 'r') as f:
            sample_rate = f.samplerate
            if n_segments > 0 and len(f) > n_segments:
                max_audio_start = len(f) - n_segments
                audio_start = random.randint(0, max_audio_start)
                f.seek(audio_start)
                samples = f.read(n_segments, dtype='float32')
            else:
                samples = f.read(dtype='float32')

        samples = samples.transpose()
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim, orig_sr=orig_sr)

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_samples(self):
        return self._samples.shape[0]

    @property
    def duration(self):
        return self._samples.shape[0] / float(self._sample_rate)

    @property
    def rms_db(self):
        mean_square = np.mean(self._samples ** 2)
        return 10 * np.log10(mean_square)

    @property
    def orig_sr(self):
        return self._orig_sr

    def gain_db(self, gain):
        self._samples *= 10.0 ** (gain / 20.0)

    def pad(self, pad_size, symmetric=False):
        """Add zero padding to the sample. The pad size is given in number
        of samples.
        If symmetric=True, `pad_size` will be added to both sides. If false,
        `pad_size`
        zeros will be added only to the end.
        """
        self._samples = np.pad(self._samples, (pad_size if symmetric else 0, pad_size), mode='constant',)

    def subsegment(self, start_time=None, end_time=None):
        """Cut the AudioSegment between given boundaries.
        Note that this is an in-place transformation.
        :param start_time: Beginning of subsegment in seconds.
        :type start_time: float
        :param end_time: End of subsegment in seconds.
        :type end_time: float
        :raise ValueError: If start_time or end_time is incorrectly set,
        e.g. out
                           of bounds in time.
        """
        start_time = 0.0 if start_time is None else start_time
        end_time = self.duration if end_time is None else end_time
        if start_time < 0.0:
            start_time = self.duration + start_time
        if end_time < 0.0:
            end_time = self.duration + end_time
        if start_time < 0.0:
            raise ValueError("The slice start position (%f s) is out of bounds." % start_time)
        if end_time < 0.0:
            raise ValueError("The slice end position (%f s) is out of bounds." % end_time)
        if start_time > end_time:
            raise ValueError(
                "The slice start position (%f s) is later than the end position (%f s)." % (start_time, end_time)
            )
        if end_time > self.duration:
            raise ValueError("The slice end position (%f s) is out of bounds (> %f s)" % (end_time, self.duration))
        start_sample = int(round(start_time * self._sample_rate))
        end_sample = int(round(end_time * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]