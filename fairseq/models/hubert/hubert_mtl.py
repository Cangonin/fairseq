# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import II

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.hubert import HubertConfig
from fairseq.models.hubert.hubert import HubertModel
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    LAYER_TYPE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.hubert_mtl_pretraining import (
    HubertMTLPretrainingConfig,
    HubertMTLPretrainingTask,
)

logger = logging.getLogger(__name__)


# TODO: find the number of supervised classes with another way
@dataclass
class HubertMTLConfig(HubertConfig):
    # balance between the supervised and self-supervised tasks
    proportion_supervised_data: float = field(
        default=0.5,
        metadata={"help": "the proportion of supervised data in each training batch"},
    )
    num_classes_supervised: int = field(
        default=45,
        metadata={"help": "the number of classes included in the supervised task"},
    )


@register_model("hubert_mtl", dataclass=HubertMTLConfig)
class HubertMTLModel(HubertModel):
    def __init__(
        self,
        cfg: HubertMTLConfig,
        task_cfg: HubertMTLPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__(cfg, task_cfg, dictionaries)

        self.final_proj_supervised = nn.Linear(
            cfg.encoder_embed_dim, cfg.num_classes_supervised
        )

    @classmethod
    def build_model(cls, cfg: HubertMTLConfig, task: HubertMTLPretrainingTask):
        """Build a new model instance."""

        model = HubertMTLModel(cfg, task.cfg, task.dictionaries)
        return model

    def forward(
        self,
        source: torch.Tensor,
        is_item_annotated: torch.BoolTensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        features = self.forward_features(source)
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        # supervised part (will give an output for everything, even for the parts without labels)
        # only predict the supervised part of the batch

        x_mean = torch.mean(
            x[is_item_annotated], dim=1
        )  # TODO: should we max pool instead? I guess mean is better because the vocalisations are longer than 25 ms?
        logits_supervised = self.final_proj_supervised(x_mean)

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        # do not forward the supervised part of the batch with the unlabelled part
        x = x[~is_item_annotated]
        padding_mask = padding_mask[~is_item_annotated]
        mask_indices = mask_indices[~is_item_annotated]
        target_list = [
            target_list[i][~is_item_annotated] for i in range(len(target_list))
        ]

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
            "logits_supervised": logits_supervised,
        }
        return result

    def get_supervised_logits(self, net_output):
        return net_output["logits_supervised"]

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
        self.final_proj_supervised = None
