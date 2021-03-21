import logging
import torch
import numpy as np
from dalle_pytorch import OpenAIDiscreteVAE, DALLE

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

vae = OpenAIDiscreteVAE()  # loads pretrained OpenAI VAE


def generate_image_code(dalle, text, mask):
    vae, text_seq_len, image_seq_len, num_text_tokens = (
        dalle.vae,
        dalle.text_seq_len,
        dalle.image_seq_len,
        dalle.num_text_tokens,
    )
    total_len = text_seq_len + image_seq_len
    out = text

    for cur_len in range(text.shape[1], total_len):
        is_image = cur_len >= text_seq_len

        text, image = out[:, :text_seq_len], out[:, text_seq_len:]

        logits = dalle(text, image, mask=mask)[:, -1, :]
        chosen = torch.argmax(logits, dim=1, keepdim=True)
        chosen -= (
            num_text_tokens if is_image else 0
        )  # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
        out = torch.cat((out, chosen), dim=-1)

        if out.shape[1] <= text_seq_len:
            mask = F.pad(mask, (0, 1), value=True)

    img_seq = out[:, -image_seq_len:]
    return img_seq


def generate_images(captions):
    print(sum(captions, start=[]))
    all_words = list(sorted(frozenset(sum(captions, start=[]))))
    word_tokens = dict(zip(all_words, range(1, len(all_words) + 1)))
    caption_tokens = [[word_tokens[w] for w in c] for c in captions]

    logging.info(f"{all_words =}")
    logging.info(f"{word_tokens =}")
    logging.info(f"{caption_tokens =}")

    longest_caption = max(len(c) for c in captions)
    captions_array = np.zeros((len(caption_tokens), longest_caption), dtype=np.int64)
    for i in range(len(caption_tokens)):
        captions_array[i, : len(caption_tokens[i])] = caption_tokens[i]

    # captions_array = torch.from_numpy(captions_array).cuda()
    captions_array = torch.from_numpy(captions_array)
    captions_mask = captions_array != 0

    logging.info(f"{captions_array = }")

    dalle = DALLE(
        dim=1024,
        vae=vae,  # automatically infer (1) image sequence length and (2) number of image tokens
        num_text_tokens=len(word_tokens) + 1,  # vocab size for text
        text_seq_len=longest_caption,  # text sequence length
        depth=12,  # should aim to be 64
        heads=16,  # attention heads
        dim_head=64,  # attention head dimension
        attn_dropout=0.1,  # attention dropout
        ff_dropout=0.1,  # feedforward dropout
    )
    # .cuda

    generated_image_codes = []
    with torch.no_grad():
        for i in range(0, len(captions), 128):
            generated = generate_image_code(
                dalle,
                captions_array[i : i + 128, ...],
                mask=captions_mask[i : i + 128, ...],
            )
            generated_image_codes.append(generated)

    generated_image_codes = torch.cat(generated_image_codes, axis=0)

    with torch.no_grad():
        generated_images = vae.decode(generated_image_codes)
        logging.info(f"{generated_images = }")
