import os, json, argparse, torch
from datasets import load_dataset
from diffusers import PixArtSigmaPipeline

def run(ckpt_dir, out_dir, num, batch, steps, gscale, make_json):
    # 1) load pipeline
    pipe = PixArtSigmaPipeline.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ModuleNotFoundError:
        print("--  xformers not installed, continuing without it.")

    # 2) load only 'num' examples from the TextVisionBlend split
    ds = load_dataset("CSU-JPG/TextAtlasEval", "textvisionblend", split="train")
    ds = ds.select(range(num))

    # 3) make output dirs
    preds = os.path.join(out_dir, "preds")
    gt    = os.path.join(out_dir, "gt")
    os.makedirs(preds, exist_ok=True)
    os.makedirs(gt,    exist_ok=True)

    # 4) optionally collect JSON metadata
    meta = [] if make_json else None

    # 5) generate
    for i, rec in enumerate(ds):
        img = pipe(
            [rec["annotation"]],
            num_inference_steps=steps,
            guidance_scale=gscale
        ).images[0]

        fn_pred = rec["image_path"]
        fn_gt   = rec["image_path"]           # e.g. "f0c9…png"

        img.save(os.path.join(preds, fn_pred))
        rec["image"].save(os.path.join(gt, fn_gt))

        if make_json:
            meta.append({
                "image_path":          os.path.join(preds, fn_pred),
                "original_image_path": os.path.join(gt,   fn_gt),
                "prompt":              rec["annotation"],
                "raw_text":            rec.get("raw_text", ""),
            })
        print(f"-- saved {fn_pred}")

    # 6) write JSON if requested
    if make_json:
        with open(os.path.join(out_dir, "eval_demo.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\n→ JSON metadata written to {out_dir}/eval_demo.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    required=True,
                   help="path to a Diffusers-style PixArtSigma folder")
    p.add_argument("--out_dir", default="demo_small")
    p.add_argument("-n","--num",     type=int,   default=3, help="how many images to sample")
    p.add_argument("--batch",        type=int,   default=1)
    p.add_argument("--steps",        type=int,   default=25)
    p.add_argument("--guidance",     type=float, default=5.0)
    p.add_argument("--make_json",    action="store_true",
                   help="also dump a eval_demo.json for local cal_*.py runs")
    args = p.parse_args()

    run(args.ckpt, args.out_dir, args.num, args.batch,
        args.steps, args.guidance, args.make_json)

#---------------------------------------------- How to call it locally ------------------------------------------------

# # first convert your first_epoch.pth → diffusers folder (in PixArt-Sigma Project) -> move transformers to textatlas-proj):
# python tools/convert_pixart_to_diffusers.py ^
#   --orig_ckpt_path output/pretrained_models/{pth file}.pth ^
#   --dump_path      output/pretrained_models/pixart512_ft ^
#   --only_transformer=False ^
#   --image_size     512 ^
#   --version        sigma
#
# then on your laptop generate 3 images + JSON:
# python generate_for_eval.py ^
#   --ckpt      ./pixart512_uft ^
#   --out_dir   demo_uft_final ^
#   -n 3 ^
#   --make_json
#
# # now you have demo_ft/preds/*.png, demo_ft/gt/*.png and demo_ft/eval_demo.json
# # run the cal_*.py scripts by hand:
#
# python TextAtlas/evaluation/cal_fid.py        --json_file demo_ft/eval_demo.json --save_path demo_ft/fid.json
# python TextAtlas/evaluation/cal_clip_score.py --json_file demo_ft/eval_demo.json --save_path demo_ft/clip
# python TextAtlas/evaluation/cal_ocr.py        --json_file demo_ft/eval_demo.json --save_path demo_ft/ocr.json
# python TextAtlas/evaluation/cal_ocr_cer.py    --ocr_result_path demo_ft/ocr.json --save_path demo_ft/cer.json


#---------------------------------------------- How to use for the full 1000‐sample run -------------------------------

# # 1) generate all 1000 images (no JSON needed):
# python generate_for_eval.py ^
#   --ckpt    ./pixart512_ft ^
#   --out_dir /big/output/exp512_clean ^
#   -n 1000
#
# # 2) call the *official* eval_script.py in HF‐mode:
# python eval_script.py \
#   --dataset_type textvisionblend \
#   --image_save_dir /big/output/exp512_clean/preds \
#   --output_dir     /big/output/exp512_clean/eval \
#   --cal_fid --cal_clip --cal_ocr