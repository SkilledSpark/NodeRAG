import os
import re
import math
import json
import argparse
import sys
import fitz
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from eval.eval_score import eval_score, eval_acc_and_f1, show_results
from eval.extract_answer import extract_answer


def load_model(model_name, cache_path):
    if model_name == '4khd':
        from models.internlm_xc2_4khd import init_model, get_response_concat
    elif model_name == 'internvl':
        from models.internvl_chat import init_model, get_response_concat
    elif model_name == 'minicpm_llama3':
        from models.minicpm_llama3 import init_model, get_response_concat
    else:
        raise NotImplementedError
    model = init_model(cache_path)
    return model, get_response_concat


def extract_images(sample, document_path, max_pages=1000, resolution=144):
    image_list = list()
    doc_name = re.sub(r"\.pdf$", "", sample["doc_id"]).split("/")[-1]
    # Get script directory for tmp folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(script_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    with fitz.open(os.path.join(document_path, sample["doc_id"])) as pdf:
        for index, page in enumerate(pdf[:max_pages]):
            img_path = os.path.join(tmp_dir, f"{doc_name}_{index+1}.png")
            if not os.path.exists(img_path):
                im = page.get_pixmap(dpi=resolution)
                im.save(img_path)
            image_list.append(img_path)

    return image_list


def concat_images(image_list, concat_num=1, column_num=3):
    interval = max(math.ceil(len(image_list) / concat_num), 1)
    concatenated_image_list = list()

    for i in range(0, len(image_list), interval):
        image_path = "_".join(image_list[0].split("_")[:-1]) + "_concat{}_{}.jpg".format(concat_num, i//interval)
        if not os.path.exists(image_path):
            images_this_batch = [
                Image.open(filename) for filename in image_list[i:i + interval]
            ]
            if column_num==1:
                total_height = images_this_batch[0].height*len(images_this_batch)
            else:
                total_height = images_this_batch[0].height*((len(images_this_batch)-1)//column_num+1)

            concatenated_image = Image.new('RGB', (images_this_batch[0].width*column_num, total_height), 'white')
            x_offset, y_offset = 0, 0
            for cnt, image in enumerate(images_this_batch):
                concatenated_image.paste(image, (x_offset, y_offset))
                x_offset += image.width
                if (cnt+1)%column_num==0:
                    y_offset += image.height
                    x_offset = 0
            concatenated_image.save(image_path)
        concatenated_image_list.append(image_path)

    return concatenated_image_list


def load_questions(args):
    print("=" * 80)
    print("MMLongBench Evaluation - LVLM Mode")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Input: {args.input_path}")
    print(f"Documents: {args.document_path}")
    print(f"Output: {args.output_path}")
    print(f"Max Pages: {args.max_pages}")
    print(f"Resolution: {args.resolution} DPI")
    print(f"Concat Num: {args.concat_num}")
    print("=" * 80)
    
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            samples = json.load(f)
        print(f"✓ Resuming from existing results: {args.output_path}")
        print(f"  Found {len(samples)} samples, {sum(1 for s in samples if 'score' in s)} already completed")
    else:
        with open(args.input_path, 'r') as f:
            samples = json.load(f)
        print(f"✓ Loaded {len(samples)} samples from: {args.input_path}")
        
    # load evaluation prompt
    print(f"\nLoading evaluation prompt from: {args.extractor_prompt_path}")
    with open(args.extractor_prompt_path) as f:
        prompt = f.read()
    print("✓ Evaluation prompt loaded")

    print(f"\nInitializing model: {args.model_name}")
    model, get_response_concat = load_model(args.model_name, args.model_cached_path)
    print(f"✓ Model initialized")
    
    print(f"\nStarting evaluation...")
    print("-" * 80)
    
    # Disable tqdm for better visibility of our custom output
    for idx, sample in enumerate(samples, 1):
        if "score" in sample:
            score = sample["score"]
            print(f"[Sample {idx}/{len(samples)}] Already completed (score: {score:.2f})")
        else:
            print(f"\n[Sample {idx}/{len(samples)}]")
            print(f"  Document: {sample['doc_id']}")
            print(f"  Question: {sample['question'][:100]}..." if len(sample['question']) > 100 else f"  Question: {sample['question']}")
            
            print(f"  Extracting images from PDF...")
            sys.stdout.flush()  # Force output to appear immediately
            image_list = extract_images(sample, document_path=args.document_path, max_pages=args.max_pages, resolution=args.resolution)
            print(f"  ✓ Extracted {len(image_list)} images")
            sys.stdout.flush()
            
            print(f"  Concatenating images (concat_num={args.concat_num})...")
            sys.stdout.flush()
            concat_image_list = concat_images(image_list, concat_num=args.concat_num)
            print(f"  ✓ Created {len(concat_image_list)} concatenated images")
            sys.stdout.flush()
            
            print(f"  Generating response with {args.model_name}...")
            sys.stdout.flush()
            response = get_response_concat(model, sample["question"], concat_image_list, max_new_tokens=args.max_tokens, temperature=args.temperature)

            if response == 'Failed':
                print(f"  ⚠ Response failed, retrying with smaller concat_num...")
                tmp_concat_num = args.concat_num - 1
                while response == 'Failed' and tmp_concat_num > 0:
                    print(f"    Retrying with concat_num={tmp_concat_num}...")
                    concat_image_list = concat_images(image_list, concat_num=tmp_concat_num)
                    response = get_response_concat(model, sample["question"], concat_image_list, max_new_tokens=args.max_tokens, temperature=args.temperature)
                    tmp_concat_num -= 1
                
                if response != 'Failed':
                    print(f"  ✓ Got response after retry")
                else:
                    print(f"  ✗ All retries failed")
            else:
                print(f"  ✓ Got response from model")

            sample["response"] = response
            print(f"  Extracting answer...")
            extracted_res = extract_answer(sample["question"], response, prompt)
            sample["extracted_res"] = extracted_res
            try:
                pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
                score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
            except Exception as e:
                pred_ans = "Failed to extract"
                score = 0.0
                print(f"  ✗ Failed to extract answer: {str(e)[:100]}")
            sample["pred"] = pred_ans
            sample["score"] = score
            print(f"  ✓ Score: {score:.2f}")

        acc, f1 = eval_acc_and_f1(samples)
        print("\n" + "=" * 80)
        print(f"Progress: {samples.index(sample) + 1}/{len(samples)} samples completed")
        print(f"Current Sample Results:")
        print(f"  Question: {sample['question'][:80]}...")
        print(f"  Ground Truth: {sample['answer']}")
        print(f"  Prediction: {sample['pred']}")
        print(f"  Score: {sample['score']:.2f}")
        print(f"\nOverall Metrics:")
        print(f"  Average Accuracy: {acc:.4f}")
        print(f"  Average F1: {f1:.4f}")
        print("=" * 80)
        
        with open(args.output_path, 'w') as f:
            json.dump(samples, f)
        print(f"✓ Saved results to: {args.output_path}\n")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    txt_path = re.sub(r"\.json$", ".txt", args.output_path)
    show_results(samples, show_path=txt_path)
    print(f"\n✓ Final results saved to: {args.output_path}")
    print(f"✓ Detailed report saved to: {txt_path}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input_path", type=str, default=os.path.join(script_dir, "data", "samples.json"))
    parser.add_argument("--document_path", type=str, default=os.path.join(script_dir, "data", "documents"))
    parser.add_argument("--extractor_prompt_path", type=str, default=os.path.join(script_dir, "eval", "prompt_for_answer_extraction.md"))
    parser.add_argument("--model_name", type=str, default="internvl", choices=["internvl", "4khd", "minicpm_llama3"])
    parser.add_argument("--model_cached_path", type=str, default=None)
    parser.add_argument("--max_pages", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=144)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    
    # Set output path relative to script directory
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    args.output_path = os.path.join(results_dir, f'res_{args.model_name}.json')
    args.concat_num = 1 if args.model_name in ['minicpm_llama3'] else 5
    load_questions(args)