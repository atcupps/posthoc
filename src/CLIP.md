(By Andrew)

I've modified `utils/extras.py` to assume there is already an OpenCLIP instance with weights in the folder `model_ckpts` (which doesn't exist until you make it). So to run the program without downloading the weights at runtime, you need to:

1. On the login node (where internet is fast), make a new folder `mkdir model_ckpts` and then `cd model_ckpts`.

2. Download the model weights; for instance, for ViT-B-32 with laion400m_e32, run ` wget https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt -O ViT-B-32-quickgelu_laion400m_e32.pt`. This depends on which corpus_config you are using to run.

3. Move the entire `model_ckpts` with the `.pt` file in it into the base `POC` folder.
