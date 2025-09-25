set -x

MODEL_PATH=./Qwen2.5-VL-3B-Instruct  # replace it with your local file path
# We also provide the checkpoint after math cold-start training: zhehuderek/praxis_vlm_7b_decisionmaking / zhehuderek/praxis_vlm_3b_decisionmaking


SYSTEM_PROMPT="""You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags, and the final answer MUST BE enclosed within <answer> and </answer> tags."""

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=zhehuderek/textual_decisionmaking_data@train \
    data.val_files=zhehuderek/textual_decisionmaking_data@test \
    data.image_key=images \
    data.prompt_key=question \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=praxis_vlm_3b_textual_training \
    worker.reward.compute_score=mcq \
    worker.rollout.limit_images=1 \
    trainer.n_gpus_per_node=2
