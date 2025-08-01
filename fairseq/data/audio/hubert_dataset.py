# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import itertools
import logging
import os
import random
import sys
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler

from fairseq.data import data_utils
from fairseq.data.audio.audio_utils import parse_path, read_from_stored_zip
from fairseq.data.fairseq_dataset import FairseqDataset

logger = logging.getLogger(__name__)


def load_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


class HubertDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
    ):
        self.audio_root, self.audio_names, inds, tot, self.sizes = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert label_processors is None or len(label_processors) == self.num_labels
        for label_path, label_rate in zip(label_paths, self.label_rates):
            verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot
            )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        _path, slice_ptr = parse_path(wav_path)
        if len(slice_ptr) == 0:
            wav, cur_sample_rate = sf.read(_path)
        else:
            assert _path.endswith(".zip")
            data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            f = io.BytesIO(data)
            wav, cur_sample_rate = sf.read(f)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav = self.get_audio(index)
        labels = self.get_labels(index)
        return {"id": index, "source": wav, "label_list": labels}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        net_input = {"source": collated_audios, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(self, targets, audio_size, audio_starts, label_rate, pad):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s : s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1.0:
                targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav


class HubertMTLDataset(HubertDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        label_paths,
        label_rates,
        ssl_num_samples,
        pad_list,
        eos_list,
        label_processors=None,
        max_keep_sample_size=None,
        min_keep_sample_size=None,
        max_sample_size=None,
        shuffle=True,
        pad_audio=False,
        normalize=False,
        store_labels=True,
        random_crop=False,
        single_target=False,
    ):
        self.audio_root, self.audio_names, inds, tot, self.sizes = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )
        self.store_labels = store_labels
        self.ssl_num_samples = ssl_num_samples

        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert label_processors is None or len(label_processors) == self.num_labels
        for label_path, label_rate in zip(label_paths, self.label_rates):
            self.verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot
            )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

    def __getitem__(self, index):
        wav = self.get_audio(index)
        labels = self.get_labels(index)
        is_item_annotated = index >= self.ssl_num_samples
        return {
            "id": index,
            "source": wav,
            "label_list": labels,
            "is_item_annotated": is_item_annotated,
        }

    # Changed the net input so that I can add whether items are annotated or not
    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        is_item_annotated = [s["is_item_annotated"] for s in samples]
        assert (
            is_item_annotated[0] == False
            and sorted(is_item_annotated) == is_item_annotated
        )  # If the ssl and sl items are not sequential, then we should rearrange them and their corresponding audios so that it is ordered

        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        # Labels are different for the ssl and sl task, only collate the labels for the ssl task (whose length depend on the audio)
        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts, is_item_annotated
        )

        net_input = {
            "source": collated_audios,
            "padding_mask": padding_mask,
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "is_item_annotated": torch.BoolTensor(is_item_annotated),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_label(
        self,
        targets_by_label,
        audio_size,
        audio_starts,
        is_item_annotated,
    ):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1.0:
                targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets,
                    audio_size,
                    audio_starts,
                    label_rate,
                    pad,
                    is_item_annotated,
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    # Assumption that targets are ordered with ssl samples first and sl sampels afterwards
    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad, is_item_annotated
    ):
        assert label_rate > 0

        targets_ssl = [
            targets[i] for i in range(len(targets)) if not is_item_annotated[i]
        ]  # TODO: find faster method
        audio_starts_ssl = [
            audio_starts[i]
            for i in range(len(audio_starts))
            if not is_item_annotated[i]
        ]

        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts_ssl]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets_ssl, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets_ssl = [
            t[s : s + frm_size] for t, s in zip(targets_ssl, frm_starts)
        ]  # this crops the labels to the minimum size
        targets[0 : len(targets_ssl)] = targets_ssl
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    # modified to deleted the warnings if the sample comes from the supervised pool
    def verify_label_lengths(
        self,
        audio_sizes,
        audio_rate,
        label_path,
        label_rate,
        inds,
        tot,
        tol=0.1,  # tolerance in seconds
    ):
        if label_rate < 0:
            logger.info(f"{label_path} is sequence label. skipped")
            return

        with open(label_path) as f:
            lengths = [len(line.rstrip().split()) for line in f]
            assert len(lengths) == tot
            lengths = [lengths[i] for i in inds]
        num_invalid = 0
        for i, ind in enumerate(inds):
            dur_from_audio = audio_sizes[i] / audio_rate
            dur_from_label = lengths[i] / label_rate
            if i < self.ssl_num_samples and abs(dur_from_audio - dur_from_label) > tol:
                logger.warning(
                    (
                        f"audio and label duration differ too much "
                        f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                        f"in line {ind+1} of {label_path}. Check if `label_rate` "
                        f"is correctly set (currently {label_rate}). "
                        f"num. of samples = {audio_sizes[i]}; "
                        f"label length = {lengths[i]}"
                    )
                )
                num_invalid += 1
        if num_invalid > 0:
            logger.warning(
                f"total {num_invalid} (audio, label) pairs with mismatched lengths"
            )


# TODO: adapt this code for distributed sampling?
class UnevenBatchSampler(BatchSampler):
    def __init__(
        self,
        size_dataset: int,
        batch_size: int,
        num_samples_unlabelled_dataset: int,
        supervised_sampling_ratio: float = 0.5,
        drop_last=False,
    ):
        """
        Args:
            dataset (Dataset): The dataset to sample from. It should be composed of all the unlabelled data followed by the labelled data
            batch_size (int): Total batch size.
            num_samples_unlabelled_dataset (int): Tot number of samples of the unlabelled data. Samples after this values belong to the labelled data
            supervised_sampling_ratio (float): How much data from the supervised part should be included in the final batch
            drop_last (bool): Whether to drop the last incomplete batch.
        """
        assert (
            0 < supervised_sampling_ratio < 1
        ), "supervised_sampling_ratio must be between 0 and 1 (exclusive)."

        self.batch_size = batch_size
        self.ssl_all_indices = [i for i in range(num_samples_unlabelled_dataset)]
        self.drop_last = drop_last

        all_indices = set(range(size_dataset))
        self.supervised_all_indices = list(all_indices - set(self.ssl_all_indices))
        self.supervised_sampling_ratio = supervised_sampling_ratio

        if len(self.ssl_all_indices) == 0 or len(self.supervised_all_indices) == 0:
            raise ValueError(
                "indices_ssl_task and indices_supervised_task must both be non-empty."
            )

        # Compute batch component sizes
        self.supervised_size = round(self.supervised_sampling_ratio * self.batch_size)
        self.ssl_size = self.batch_size - self.supervised_size

        # Estimate number of batches available. TODO: change that so that the num of batches doesn't depend on the supervised data?
        # It could depend on the unsupervised data, but then I need to implement this in the sampler
        self.num_batches = min(
            len(self.ssl_all_indices) // self.ssl_size,
            len(self.supervised_all_indices) // self.supervised_size,
        )

    def __iter__(self):
        ssl_samples = random.sample(self.ssl_all_indices, len(self.ssl_all_indices))
        supervised_samples = random.sample(
            self.supervised_all_indices, len(self.supervised_all_indices)
        )

        for i in range(self.num_batches):
            ssl_start = i * self.ssl_size
            supervised_start = i * self.supervised_size
            batch = (
                ssl_samples[ssl_start : ssl_start + self.ssl_size]
                + supervised_samples[
                    supervised_start : supervised_start + self.supervised_size
                ]
            )
            yield batch

    def __len__(self):
        return self.num_batches if self.drop_last else self.num_batches + 1
