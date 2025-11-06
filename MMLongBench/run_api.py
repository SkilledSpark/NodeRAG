import os
import re
import json
import base64
import argparse
import sys
import fitz

from PIL import Image
from uuid import uuid4

from eval.extract_answer import extract_answer
from eval.eval_score import eval_score, eval_acc_and_f1, show_results


cached_image_list = dict()


def encode_image_to_base64(img):
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    tmp = os.path.join('/tmp', str(uuid4()) + '.jpg')
    img.save(tmp)
    with open(tmp, 'rb') as image_file:
        image_data = image_file.read()
    ret = base64.b64encode(image_data).decode('utf-8')
    os.remove(tmp)
    return ret


def process_sample_gpt(sample, args):
    question = sample["question"]
    doc_name = re.sub(r"\.pdf$", "", sample["doc_id"]).split("/")[-1]
    
    # Get script directory for tmp folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(script_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    image_list = list()
    with fitz.open(os.path.join(args.document_path, sample["doc_id"])) as pdf:
        for index, page in enumerate(pdf[:args.max_pages]):
            img_path = os.path.join(tmp_dir, f"{doc_name}_{index+1}.png")
            if not os.path.exists(img_path):
                image = page.get_pixmap(dpi=args.resolution)
                image.save(img_path)
            image = Image.open(img_path)
            encoded_image = encode_image_to_base64(image)
            image_list.append(encoded_image)

    content = list()
    content.append(
        {
            "type": "text",
            "text": question,
        }
    )
    for encoded_image in image_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages


def process_sample_gemini(sample, args, mode):
    question = sample["question"]
    doc_name = re.sub(r"\.pdf$", "", sample["doc_id"]).split("/")[-1]
    
    # Get script directory for tmp folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(script_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    image_list = list()
    with fitz.open(os.path.join(args.document_path, sample["doc_id"])) as pdf:
        if mode=="png":
            for index, page in enumerate(pdf[:args.max_pages]):
                img_path = os.path.join(tmp_dir, f"{doc_name}_{index+1}.png")
                if not os.path.exists(img_path):
                    im = page.get_pixmap(dpi=args.resolution)
                    im.save(img_path)
                image_list.append(Image.open(img_path))
        else:
            if sample["doc_id"] in cached_image_list:
                image_list = cached_image_list[sample["doc_id"]]
            else:
                for index, page in enumerate(pdf[:args.max_pages]):
                    img_path = os.path.join(tmp_dir, f"{doc_name}_{index+1}.png")
                    if not os.path.exists(img_path):
                        im = page.get_pixmap(dpi=args.resolution)
                        im.save(img_path)
                    image_list.append(genai.upload_file(img_path))
                cached_image_list[sample["doc_id"]] = image_list
    
    return [question] + image_list


def process_sample(sample, args, mode="png"):
    if "gpt-4" in args.model_name:
        return process_sample_gpt(sample, args)
    elif "gemini-1.5" in args.model_name:
        return process_sample_gemini(sample, args, mode)
    else:
        raise AssertionError()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input_path", type=str, default=os.path.join(script_dir, "data", "samples.json"))
    parser.add_argument("--document_path", type=str, default=os.path.join(script_dir, "data", "documents"))
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_pages", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=144)
    parser.add_argument("--max_try", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--extractor_prompt_path", type=str, default=os.path.join(script_dir, "eval", "prompt_for_answer_extraction.md"))
    args = parser.parse_args()

    # Set output path relative to script directory
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    args.output_path = os.path.join(results_dir, f'res_{args.model_name}.json')

    print("=" * 80)
    print(f"MMLongBench Evaluation - API Mode")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Input: {args.input_path}")
    print(f"Documents: {args.document_path}")
    print(f"Output: {args.output_path}")
    print(f"Max Pages: {args.max_pages}")
    print(f"Resolution: {args.resolution} DPI")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print("=" * 80)

    if "gpt-4" in args.model_name:
        from openai import OpenAI
        client = OpenAI()
        print(f"✓ Initialized OpenAI client for {args.model_name}")
    elif "gemini-1.5" in args.model_name:
        import google.generativeai as genai
        client = genai.GenerativeModel(args.model_name)
        config = genai.types.GenerationConfig(max_output_tokens=args.max_tokens, temperature=args.temperature)
        print(f"✓ Initialized Gemini client for {args.model_name}")
    else:
        raise AssertionError()

    print(f"\nLoading evaluation prompt from: {args.extractor_prompt_path}")
    with open(args.extractor_prompt_path) as f:
        prompt = f.read()
    print("✓ Evaluation prompt loaded")
    
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            samples = json.load(f)
        print(f"✓ Resuming from existing results: {args.output_path}")
        print(f"  Found {len(samples)} samples, {sum(1 for s in samples if 'score' in s)} already completed")
    else:
        with open(args.input_path, 'r') as f:
            samples = json.load(f)
        print(f"✓ Loaded {len(samples)} samples from: {args.input_path}")
    
    print(f"\nStarting evaluation...")
    print("-" * 80)

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
            
            messages = process_sample(sample, args)
            print(f"  Processing {args.max_pages} pages at {args.resolution} DPI...")
            sys.stdout.flush()  # Force output to appear immediately
            
            try_cnt = 0
            is_success = False
            while True:
                try:
                    if "gpt-4" in args.model_name:
                        response = client.chat.completions.create(
                            model=args.model_name,
                            messages=messages,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature
                        )
                        response = response.choices[0].message.content
                    elif "gemini-1.5" in args.model_name:
                        try:
                            response = client.generate_content(messages, generation_config=config)
                        except Exception as e:
                            print(f"  ⚠ Payload oversize! Using File API instead. Error: {e}")
                            messages = process_sample(sample, args, mode="file")
                            response = client.generate_content(messages, generation_config=config)
                        response.resolve()
                        response = response.text.strip()
                    else:
                        pass
                    is_success = True
                    print(f"  ✓ Got response from {args.model_name}")
                    sys.stdout.flush()
                except Exception as e:
                    try_cnt += 1
                    response = "Failed"
                    print(f"  ✗ API call failed (attempt {try_cnt}/{args.max_try}): {str(e)[:100]}")
                    sys.stdout.flush()
                if is_success or try_cnt>args.max_try:
                    break
                
            sample["response"] = response
            extracted_res = extract_answer(sample["question"], response, prompt)
            sample["extracted_res"] = extracted_res
            # try:
            print(f"  Extracting answer...")
            sys.stdout.flush()
            pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
            score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
            # except:
            #     pred_ans = "Failed to extract"
            #     score = 0.0
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