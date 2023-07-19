import argparse
import json


def get_args():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    return parser.parse_args()


def dialog_combine_all(input_path, output_path):
    data = json.load(open(input_path))['dialogs']
    out = {}

    for d in data:
        img_id = d['image_id']
        if img_id not in out:
            out[img_id] = {
                "image_id": img_id,
                "dialog": []
            }
        out[img_id]['dialog'].append({
            'question': d['dialog'][0]['question'],
            'answer': d['dialog'][0]['answer']
        })
    out = list(out.values())
    out = {'dialogs': out}
    
    with open(output_path, 'w') as outf:
        json.dump(out, outf, indent=2)

if __name__ == '__main__':
    args = get_args()
    dialog_combine_all(args.input_path, args.output_path)