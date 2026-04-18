import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()

cur_result = {}

for line in open(args.src):
    data = json.loads(line)
    qid = str(data["question_id"]).strip()
    if not qid.startswith("v1_"):
        qid = f"v1_{qid}"
    cur_result[qid] = data["text"]

with open(args.dst, "w") as f:
    json.dump(cur_result, f, indent=2)
