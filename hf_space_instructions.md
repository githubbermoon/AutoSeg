# Deploying to Hugging Face Spaces

This project is ready for deployment on Hugging Face Spaces.

## Prerequisites
1.  Hugging Face Account.
2.  Weights & Biases Account (optional, for logging).

## Steps

1.  **Create a New Space**:
    - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
    - Name: `terrain-safety-analyzer`.
    - SDK: **Gradio**.
    - Hardware: **CPU Basic** (Free) works with SegFormer B0.

2.  **Upload Files**:
    Upload the following files to the Space:
    - `app.py`
    - `model_utils.py`
    - `requirements.txt`
    - `README.md` (optional, Space has its own)

3.  **Secrets (Optional)**:
    - If using W&B, go to **Settings > Variables and secrets**.
    - Add `WANDB_API_KEY`.

4.  **Launch**:
    - The Space will build and launch automatically.
    - If it fails on "Out of Memory", switch to a smaller model (B0) in `app.py` as default.

## Note on Models
This app downloads `nvidia/segformer-b0-finetuned-ade-512-512` at startup. This adds a small delay on the first cold boot.
