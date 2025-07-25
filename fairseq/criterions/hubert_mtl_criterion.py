# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.hubert_criterion import HubertCriterion, HubertCriterionConfig
from fairseq.logging import metrics


# TODO: constrain the weight of the unsupervised task
# TODO: see if I can put the supervised_task_weight in the loss_weights part instead?
# If I change the proportion of the supervised data, I will have to account for it in the loss, since it will change the magnitude of the loss.
# Should I divide by the number of examples when computing the loss function to circumvent this problem? Or I could keep the proprotion of
# supervised samples constant, but only vary the weight
@dataclass
class HubertMTLCriterionConfig(HubertCriterionConfig):
    # TODO: I have defined this in two places, choose one.
    supervised_task_weight: float = field(
        default=0.1,
        metadata={"help": "weight for the supervised task"},
    )


@register_criterion("hubert_mtl", dataclass=HubertMTLCriterionConfig)
class HubertMTLCriterion(HubertCriterion):
    def __init__(
        self,
        task,
        pred_masked_weight,
        pred_nomask_weight,
        supervised_task_weight,
        loss_weights=None,
        log_keys=None,
    ):
        super().__init__(
            task, pred_masked_weight, pred_nomask_weight, loss_weights, log_keys
        )
        self.supervised_task_weight = supervised_task_weight
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(target_list=sample["target_list"], **sample["net_input"])
        loss = 0.0
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"
        is_item_annotated = sample["is_item_annotated"]

        loss_m_list = []
        logp_m_list = model.get_logits(net_output, True)
        # logp_m_list = model.get_logits(net_output, True)[
        #     ~is_item_annotated
        # ]  # TODO: is this correct??
        targ_m_list = model.get_targets(
            net_output, True
        )  # Those are only 0??? I don't get why
        # targ_m_list = model.get_targets(net_output, True)[~is_item_annotated]
        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            loss_m = F.cross_entropy(
                logp_m, targ_m, reduction=reduction
            )  # should I add the mask here? Did I get the reduction correctly?
            loss_m_list.append(loss_m)
            logging_output[f"loss_m_{i}"] = loss_m.detach().item()
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size += targ_m_list[
                0
            ].numel()  # TODO: what does the [0] correspond to exactly?

        loss_u_list = []
        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = model.get_targets(net_output, False)

        # logp_u_list = model.get_logits(net_output, False)[~is_item_annotated]
        # targ_u_list = model.get_targets(net_output, False)[~is_item_annotated]
        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
            loss_u_list.append(loss_u)
            logging_output[f"loss_u_{i}"] = loss_u.detach().item()
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size += targ_u_list[0].numel()

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(
                net_output
            )  # TODO: implement loss there instead?
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    logging_output[f"loss_{n}"] = p.item()

        # def compute_supervised_loss(
        #     mask_supervised: torch.BoolTensor,
        #     logits: torch.Tensor,
        #     labels: torch.Tensor,
        #     reduction: Optional[str],
        # ) -> torch.Tensor:
        #     labels = labels[mask_supervised]
        #     logits = logits[mask_supervised]
        #     masked_loss = F.binary_cross_entropy_with_logits(
        #         logits, target=labels, reduction=reduction
        #     )
        #     return masked_loss

        # supervised_loss = compute_supervised_loss(
        #     mask_supervised=is_item_annotated,
        #     logits=model.get_supervised_logits(net_output),
        #     labels=sample["supervised_labels"],
        # )

        # # Weighted ssl and sl loss
        # ssl_task_weight = 1 - self.supervised_task_weight
        # loss = ssl_task_weight * loss + self.supervised_task_weight * supervised_loss

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        def compute_correct(logits):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == 0
                min = logits.argmin(-1) == 0
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = max.numel()
                return corr, count

        with torch.no_grad():
            for i, logp_m in enumerate(logp_m_list):
                corr_m, count_m = compute_correct(logp_m)
                logging_output[f"correct_m_{i}"] = corr_m
                logging_output[f"count_m_{i}"] = count_m

            for i, logp_u in enumerate(logp_u_list):
                corr_u, count_u = compute_correct(logp_u)
                logging_output[f"correct_u_{i}"] = corr_u
                logging_output[f"count_u_{i}"] = count_u

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
