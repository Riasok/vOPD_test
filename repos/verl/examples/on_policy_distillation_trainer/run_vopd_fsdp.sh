#!/usr/bin/env bash
# Run from the verl repository root. Inherits model/data/resource environment
# variables from run_qwen3_8b_fsdp.sh; pass Hydra overrides as additional arguments.
set -euo pipefail
mode=${VOPD_MODE:-vopd_topk}
k=${VOPD_TOPK:-20}
if [[ "$mode" != vopd_topk && "$mode" != vopd_full ]]; then
    echo 'VOPD_MODE must be vopd_topk or vopd_full' >&2
    exit 1
fi
rollout_k=0
if [[ "$mode" == vopd_topk ]]; then
    rollout_k=$k
fi
DISTILLATION_LOSS_MODE="$mode" DISTILLATION_TOPK="$k" USE_POLICY_GRADIENT=True \
bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_fused_kernels=False \
    actor_rollout_ref.actor.pad_to_length=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.logprobs_mode=processed_logprobs \
    actor_rollout_ref.rollout.topk_log_probs="$rollout_k" \
    +distillation.teacher_models.teacher_model.inference.logprobs_mode=raw_logprobs \
    distillation.distillation_loss.loss_max_clamp=null \
    distillation.distillation_loss.log_prob_min_clamp=null \
    "$@"
