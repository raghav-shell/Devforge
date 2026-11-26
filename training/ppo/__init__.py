"""Custom recurrent PPO engine (owner: rl-training-engineer). Training-only; never imported by submission/.

Modules: gae (compute_gae), rollout (RolloutBuffer), loss (ppo_loss, ReturnNormalizer), trainer
(PPOTrainer), checkpoint (ckpt-1), export (export-1), logger (JSONL + TensorBoard), adapters (action API
over doom_agent.models), testing (fake policy / vec env test doubles). See docs/TRAINING.md.
"""
