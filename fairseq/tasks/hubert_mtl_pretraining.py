# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from fairseq.data import FairseqDataset, data_utils, iterators
from fairseq.data.audio.hubert_dataset import HubertMTLDataset, UnevenBatchSampler
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
    LabelEncoder,
)

logger = logging.getLogger(__name__)


@dataclass
class HubertMTLPretrainingConfig(HubertPretrainingConfig):
    # we want to get the number of ssl examples for the dataset to know where the border beterrn the ssl and the supervised data is
    ssl_data: str = field(
        default=MISSING,
        metadata={"help": "path to data directory containing the the ssl data"},
    )


@register_task("hubert_mtl_pretraining", dataclass=HubertMTLPretrainingConfig)
class HubertMTLPretrainingTask(HubertPretrainingTask):

    cfg: HubertMTLPretrainingConfig

    def __init__(
        self,
        cfg: HubertMTLPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

    @classmethod
    def setup_task(
        cls, cfg: HubertMTLPretrainingConfig, **kwargs
    ) -> "HubertMTLPretrainingTask":
        return cls(cfg)

    # TODO: check and test that!!!
    def _get_size_ssl_dataset(self, split: str) -> int:
        with open(f"{self.cfg.ssl_data}/{split}.tsv") as f:
            line_count = sum(1 for _ in f)
            size_ssl_dataset = line_count - 1  # header
        return size_ssl_dataset

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        paths = [f"{self.cfg.label_dir}/{split}.{l}" for l in self.cfg.labels]

        # hubert v1: pad_audio=True, random_crop=False;
        self.datasets[split] = HubertMTLDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            ssl_num_samples=self._get_size_ssl_dataset(split),
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=False,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
        )

    # In order to control the indices of each batch. I just modified the batch sampler in this
    # method
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = (
            not disable_iterator_cache
            and not update_epoch_batch_itr
            and self.can_reuse_epoch_itr(dataset)
        )
        logger.info(f"can_reuse_epoch_itr = {can_reuse_epoch_itr}")
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        def make_batches(dataset, epoch):
            logger.info(f"creating new batches for epoch {epoch}")

            # # get indices ordered by example size
            # with data_utils.numpy_seed(seed + epoch):
            #     indices = dataset.ordered_indices()

            # # filter examples that are too large
            # if max_positions is not None:
            #     indices = self.filter_indices_by_size(
            #         indices, dataset, max_positions, ignore_invalid_inputs
            #     )

            # create mini-batches with given size constraints

            # TODO: see if this is what was expected
            batches = UnevenBatchSampler(
                len(dataset),
                batch_size=64,  # TODO: use max_tokens somehow instead or define the batch size in the config
                num_samples_unlabelled_dataset=dataset.ssl_num_samples,
            )
            return batches

        reuse_dataloader = getattr(self.cfg, "reuse_dataloader", True)
        persistent_workers = getattr(self.cfg, "persistent_workers", True)
        rebuild_batches = getattr(self.cfg, "rebuild_batches", False)
        logger.info(f"reuse_dataloader = {reuse_dataloader}")
        logger.info(f"rebuild_batches = {rebuild_batches}")

        if rebuild_batches:
            logger.info("batches will be rebuilt for each epoch")
            batch_sampler = make_batches
        else:
            batch_sampler = make_batches(dataset, epoch)

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
            grouped_shuffling=grouped_shuffling,
            reuse_dataloader=reuse_dataloader,
            persistent_workers=persistent_workers,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter
