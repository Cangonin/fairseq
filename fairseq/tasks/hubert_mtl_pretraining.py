# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import sys
from typing import List, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq.data import HubertDataset
from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig, HubertPretrainingTask, LabelEncoder
from fairseq.tasks import register_task
from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclass
class HubertMTLPretrainingConfig(HubertPretrainingConfig):
    data_supervised: str = field(default=MISSING, metadata={"help": "path to supervised data directory"})
    
    labels_dir_supervised: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "labels directory for the supervised task"
            )
        },
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

    
    # TODO: only overwrite the load_dataset function with a supervised=False/true kwarg to avoid boilerplate code?
    def load_supervised_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data_supervised}/{split}.tsv"
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts] # TODO: change that one
        paths = [f"{self.cfg.labels_dir_supervised}/{split}.{l}" for l in self.cfg.labels] # TODO: change that one to align with the unsupervised labels paths

        # hubert v1: pad_audio=True, random_crop=False;
        self.datasets[split] = HubertDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
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