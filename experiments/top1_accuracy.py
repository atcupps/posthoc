import json, re, argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv_file", required=True)
parser.add_argument("--topk_json", required=True)
parser.add_argument("--test_txt", required=True)
parser.add_argument("--labels_json", required=True)
args = parser.parse_args()

def norm(path):
    return "/".join(path.replace("\\", "/").split("/")[-2:])

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_first(answer):
    match = re.search(r"1st:\s*(\d+)", str(answer))
    return int(match.group(1)) if match else None

def get_candidates(prompt):
    pattern = r"Candidate\s+(\d+):\s+(.+?)\s+\((.+?)\)"
    matches = re.findall(pattern, str(prompt))

    candidates = {}

    for num, common, sci in matches:
        candidates[int(num)] = [common.strip(), sci.strip()]

    return candidates

labels = load_json(args.labels_json)
name_to_label = {}

for id, info in labels.items():
    names = [info.get("name", ""), info.get("most_common_name", "")]
    names += list(info.get("alternates", {}).keys())

    for name in names:
        if name:
            name_to_label[name.lower().strip()] = int(id)

true_labels = {}

with open(args.test_txt) as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) >= 2:
            true_labels[norm(parts[0])] = int(parts[1])

topk_data = load_json(args.topk_json)

baseline_preds = {}
for item in topk_data.values():
    baseline_preds[norm(item["image_path"])] = int(item["pred"])

rows = []
before = 0
after = 0
total = 0
none_count = 0

df = pd.read_csv(args.csv_file)

for i, row in df.iterrows():
    image_path = norm(row["image_path"])

    if image_path not in true_labels or image_path not in baseline_preds:
        continue

    true_lbl = true_labels[image_path]
    before_pred = baseline_preds[image_path]

    candidates = get_candidates(row["prompt"])
    ranked = get_first(row["answer"])

    after_pred = -1
    after_species_name = "None of the candidates"

    if ranked in candidates:
        for name in candidates[ranked]:
            after_pred = name_to_label.get(name.lower().strip(), -1)

            if after_pred != -1:
                after_species_name = name
                break
    else:
        none_count += 1

    before += before_pred == true_lbl
    after += after_pred == true_lbl
    total += 1

    rows.append({
        "image_path": image_path,
        "true_label": true_lbl,
        "before_pred": before_pred,
        "after_pred": after_pred,
        "after_species_name": after_species_name,
        "before_correct": before_pred == true_lbl,
        "after_correct": after_pred == true_lbl
    })
before_acc = before / total * 100
after_acc = after / total * 100

results = (
    f"Total evaluated: {total}\n"
    f"Top-1 accuracy before reranking: {before_acc:.2f}%\n"
    f"Top-1 accuracy after reranking:  {after_acc:.2f}%\n"
    f"Change: {after_acc - before_acc:+.2f}%\n"
    f"Number of 'None' outputs: {none_count}\n"
)

print(results)

with open("accuracy_results.txt", "w") as f:
    f.write(results)

pd.DataFrame(rows).to_csv("top1_before_after_comparison.csv", index=False)
print("Finished")