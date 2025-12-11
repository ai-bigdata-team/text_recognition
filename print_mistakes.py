import os
import json 
import argparse

'''Example
python print_mistakes.py \
    --input-file /home/pavt1024/deeplearning/text_recognition/logs/training_mcocr/predictions--iter=2400.jsonl
'''

def get_mistakes(filepath):
    mistakes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines: 
        if line.strip():
            sample = json.loads(line.strip())
            actual = sample.get('actual')
            pred = sample.get('prediction')
            if actual != pred:
                mistakes.append(sample)
    print(f"Number of lines: {len(lines)}\nTotal mistakes: {len(mistakes)}")
    return mistakes

def save_mistakes(mistakes, output_file):
    with open(output_file, 'w', encoding='utf-8') as f: 
        for mistake in mistakes:
            f.write(json.dumps(mistake, ensure_ascii=False)+"\n")
    print(f"Save mistakes successfully, total samples: {len(mistakes)}")

def get_args():
    parser = argparse.ArgumentParser(description="Code to extract mistakes from predictions of model")
    parser.add_argument("--input-file", required=True, help="Input file, having 'actual' and 'prediction' label")
    parser.add_argument("--output-file", default="unknown", help="file path to save mistakes")
    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = get_args()
    mistakes = get_mistakes(args.input_file)
    if args.output_file == "unknown":
        args.output_file = args.input_file.replace(".jsonl", "--mistakes.jsonl")
    save_mistakes(mistakes, args.output_file)