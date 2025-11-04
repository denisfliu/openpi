# Training and Serving Commands

## Training

To train a model on the penguin_grasp dataset:

```bash
uv run python scripts/train.py pi0_penguin_grasp_low_mem_finetune \
  --exp-name EXP_NAME \
  --num-train-steps NUM_STEPS \
  --batch-size BATCH_SIZE \
  --log-interval LOG_INTERVAL \
  --save-interval SAVE_INTERVAL
```

### Example

```bash
uv run python scripts/train.py pi0_penguin_grasp_low_mem_finetune \
  --exp-name my_experiment \
  --num-train-steps 30000 \
  --batch-size 2 \
  --log-interval 50 \
  --save-interval 5000
```

## Serving Policy

To serve a trained policy for inference:

```bash
uv run python scripts/serve_policy.py --port 8000 \
  policy:checkpoint \
  --policy.config CONFIG_NAME \
  --policy.dir CHECKPOINT_PATH
```

### Example

```bash
uv run python scripts/serve_policy.py --port 8000 \
  policy:checkpoint \
  --policy.config pi0_penguin_grasp_low_mem_finetune \
  --policy.dir checkpoints/pi0_penguin_grasp_low_mem_finetune/my_experiment/29999
```

## Notes

- Checkpoints are saved in `checkpoints/CONFIG_NAME/EXP_NAME/STEP_NUMBER/`
- The serve command requires `--port` to come before the `policy:checkpoint` subcommand
- Training outputs are logged to wandb

