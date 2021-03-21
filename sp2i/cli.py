import logging
import torch
from dalle_pytorch import OpenAIDiscreteVAE, DALLE

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

vae = OpenAIDiscreteVAE()  # loads pretrained OpenAI VAE

dalle = DALLE(
    dim=1024,
    vae=vae,  # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens=10000,  # vocab size for text
    text_seq_len=256,  # text sequence length
    depth=1,  # should aim to be 64
    heads=16,  # attention heads
    dim_head=64,  # attention head dimension
    attn_dropout=0.1,  # attention dropout
    ff_dropout=0.1,  # feedforward dropout
)


def generate_images(text):
    logging.info("Generating images...")
    images = dalle.generate_images(text)
    print(type(images))


    # text = torch.randint(0, 10000, (4, 256))
    # images = torch.randn(4, 3, 256, 256)
    # mask = torch.ones_like(text).bool()

    # loss = dalle(text, images, mask = mask, return_loss = True)
    # loss.backward()
