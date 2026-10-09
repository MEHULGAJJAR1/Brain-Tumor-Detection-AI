# Model artifact setup

This project does **not** include trained ResNet50 weights. The supplied reference material described a checkpoint, but no `.pt`/`.pth` artifact or source training notebook was attached. The web app therefore starts in an explicit **model not configured** state and will never invent a prediction.

To create a compatible checkpoint, see the dataset preparation and training commands in the project `README.md`. The training script writes `models/brain_tumor_resnet50.pt` with the architecture, class order, and preprocessing metadata expected by `backend/ml/inference.py`.

You may point the app to another compatible artifact with `MODEL_PATH`. Checkpoints must use `architecture: resnet50_mlp_v1`, include a PyTorch `state_dict`, and include all three `class_names`: `glioma`, `meningioma`, and `pituitary`. Arbitrary model files are not loaded as executable pickle objects; the app uses PyTorch's restricted `weights_only` loader when available.

A checkpoint from a different training pipeline may have a different class order, head, resize policy, or normalization. Do not use it unless these have been verified. Even a compatible checkpoint is research software, not a medical device.
