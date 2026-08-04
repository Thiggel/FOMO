"""Check whether SD3 empty-prompt guidance changes fixed-seed img2img output."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusion3Img2ImgPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--guidance", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.5])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.image).convert("RGB")
    width = max(16, source.width // 16 * 16)
    height = max(16, source.height // 16 * 16)
    source = source.resize((width, height), Image.Resampling.BICUBIC)

    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    arrays = {}
    rows = []
    for guidance in args.guidance:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        image = pipe(
            prompt="",
            image=source,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=guidance,
            generator=generator,
            height=height,
            width=width,
        ).images[0]
        path = output / f"guidance_{guidance:g}.png"
        image.save(path)
        array = np.asarray(image, dtype=np.float32) / 255.0
        arrays[guidance] = array
        rows.append(
            {
                "guidance": guidance,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    reference = arrays[args.guidance[0]]
    for row in rows:
        difference = arrays[row["guidance"]] - reference
        mse = float(np.mean(difference**2))
        row.update(
            {
                "reference_guidance": args.guidance[0],
                "pixel_mae": float(np.mean(np.abs(difference))),
                "pixel_max_abs": float(np.max(np.abs(difference))),
                "psnr": math.inf if mse == 0 else float(-10.0 * math.log10(mse)),
            }
        )

    payload = {
        "image": str(Path(args.image).resolve()),
        "prompt": "",
        "negative_prompt": "pipeline default",
        "steps": args.steps,
        "strength": args.strength,
        "seed": args.seed,
        "height": height,
        "width": width,
        "results": rows,
    }
    (output / "guidance_check.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
