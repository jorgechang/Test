# R2Dreamer ANYmal v13.45

This private repository stores the complete v13.45 experiment bundle for transfer to OSCAR.

Because the GitHub connector cannot upload a binary ZIP directly, the original ZIP is stored losslessly as Base64 chunks in `bundle_parts/`. `restore_bundle.sh` reconstructs the original ZIP and verifies its SHA-256 before use.

## OSCAR setup

```bash
cd ~/path/to/VPT/experiments
git clone git@github.com:jorgechang/Test.git
cd Test

bash restore_bundle.sh
unzip r2dreamer_isaaclab_anymal_v13_45_simple_runs_three_checkpoints_proprio_complete.zip

cd r2dreamer_isaaclab_anymal_v13_45_simple_runs_three_checkpoints_proprio_complete
```

Expected SHA-256:

```text
553db463a3be0149b90a89c319df22b31adcedefd870607c1464e34ed4ae0aff
```

The bundle contains both launch modes:

- `./run_anymal_headless.sh` — RGB only
- `./run_anymal_proprio_headless.sh` — RGB + 33-D proprioception

It also keeps the three checkpoint outputs: `latest.pt`, `best_success.pt`, and `best_reward.pt`.
