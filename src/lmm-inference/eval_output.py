# NOTE: This file was made with assitance from GEMINI
"""
Evaluate the accuracy of LMM inference output CSV against ground truth.

Supports both:
  - Direct species name answers (e.g., "Most Likely: [Name]")
  - Ranking-based answers (e.g., "1st: 3") which need topk-json to resolve

Usage:
    python eval_output.py \
        --output-csv /path/to/output.csv \
        --test-list ../data/semi-aves/test.txt \
        --taxonomy-json ../data/semi-aves/semi-aves_labels.json \
        --topk-json /path/to/topk_predictions.json \
        --image-dir semi-aves
"""

import csv
import json
import os
import re
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict


def normalize(s: str) -> str:
    """Normalize a name string for fuzzy matching."""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = re.sub(r"[''`]", "'", s)
    s = re.sub(r"\s+", " ", s)
    return s


def build_name_to_id(taxonomy_path: Path):
    """Build lookup dicts: common_name->id, scientific_name->id."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    common2id = {}
    sci2id = {}
    id2common = {}
    id2sci = {}

    for cid_str, rec in taxonomy.items():
        cid = int(cid_str)
        common = rec.get("most_common_name", "").strip()
        sci = rec.get("name", "").strip()
        if common:
            common2id[normalize(common)] = cid
            id2common[cid] = common
        if sci:
            sci2id[normalize(sci)] = cid
            id2sci[cid] = sci
        # Also index alternate names
        for alt_name in rec.get("alternates", {}).keys():
            alt_norm = normalize(alt_name)
            if alt_norm not in common2id:
                common2id[alt_norm] = cid
            if alt_norm not in sci2id:
                sci2id[alt_norm] = cid

    return common2id, sci2id, id2common, id2sci


def load_ground_truth(test_list_path: Path):
    """Load ground truth: image_path -> class_id from test.txt."""
    gt = {}
    with open(test_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                img_path = parts[0]
                label = int(parts[1])
                gt[img_path] = label
                # Also store just the filename
                gt[Path(img_path).name] = label
    return gt


def _variants_for_path_str(path_str, image_dir):
    """Generate path variants for matching against topk index."""
    t = path_str.strip().split()[0].split(",")[0]
    p = Path(t)
    variants = set()
    variants.add(t)
    if not p.is_absolute():
        try:
            variants.add(str((image_dir / p).resolve()))
        except Exception:
            variants.add(str(image_dir / p))
    else:
        try:
            variants.add(str(p.resolve()))
        except Exception:
            variants.add(str(p))
    try:
        abs_p = (image_dir / p) if not p.is_absolute() else p
        abs_p = abs_p.resolve()
        variants.add(str(abs_p.relative_to(image_dir.resolve())))
    except Exception:
        pass
    try:
        parts = list(Path(t).parts)
        imgdir_name = Path(image_dir).name
        if parts and parts[0] == imgdir_name:
            variants.add(str(Path(*parts[1:])))
    except Exception:
        pass
    variants.add(Path(t).name)
    return list(variants)


def build_topk_index(topk_json_path, image_dir):
    """Build topk index matching the inference script's logic."""
    with open(topk_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index = {}
    if not isinstance(data, dict):
        raise ValueError("topk_json must be a dict")

    looks_like_layout_B = False
    if data and all(k.isdigit() for k in list(data.keys())[:5]):
        for k in list(data.keys())[:5]:
            v = data[k]
            looks_like_layout_B = isinstance(v, dict) and "image_path" in v
            if not looks_like_layout_B:
                break

    if looks_like_layout_B:
        for _, rec in data.items():
            imgp = rec.get("image_path")
            if not imgp:
                continue
            for v in _variants_for_path_str(imgp, image_dir):
                index.setdefault(v, rec)
    else:
        for k, rec in data.items():
            for v in _variants_for_path_str(k, image_dir):
                index.setdefault(v, rec)
    return index


def parse_ranking_answer(answer, topk_cls):
    """Parse a ranking-format answer to get the predicted class ID.
    
    The answer format is:
        Ranking:
        1st: 3
        2nd: 1
        ...
    Where the number refers to the candidate index (1-based).
    """
    answer = answer.replace("\\n", "\n")

    # Try to find "1st: X" pattern
    match = re.search(r"1st:\s*(\d+)", answer)
    if match:
        candidate_idx = int(match.group(1))
        # candidate_idx is 1-based, topk_cls is 0-indexed list
        if 1 <= candidate_idx <= len(topk_cls):
            return int(topk_cls[candidate_idx - 1])

    # Fallback: try "Ranking:" followed by first number
    match = re.search(r"Ranking:.*?(\d+)", answer, re.DOTALL)
    if match:
        candidate_idx = int(match.group(1))
        if 1 <= candidate_idx <= len(topk_cls):
            return int(topk_cls[candidate_idx - 1])

    return -1


def parse_name_answer(answer, common2id, sci2id):
    """Parse a name-format answer (e.g., Most Likely: [Name (Sci)])."""
    answer = answer.replace("\\n", "\n")

    # Try to match "Most Likely: Common Name (Scientific Name)"
    match = re.search(
        r"Most\s+Likely:\s*\[?\s*(.+?)(?:\s*\(([^)]+)\))?\s*\]?",
        answer, re.IGNORECASE
    )
    if match:
        common_name = match.group(1).strip().rstrip("]")
        sci_name = match.group(2).strip() if match.group(2) else None

        if sci_name:
            norm_sci = normalize(sci_name)
            if norm_sci in sci2id:
                return sci2id[norm_sci]
            if norm_sci in common2id:
                return common2id[norm_sci]

        norm_common = normalize(common_name)
        if norm_common in common2id:
            return common2id[norm_common]
        if norm_common in sci2id:
            return sci2id[norm_common]

    # Fallback: try to find any known species name in the answer
    norm_answer = normalize(answer)
    all_names = list(common2id.keys()) + list(sci2id.keys())
    all_names.sort(key=len, reverse=True)
    for name in all_names:
        if name in norm_answer:
            if name in common2id:
                return common2id[name]
            return sci2id[name]

    return -1


def detect_answer_format(output_csv):
    """Auto-detect whether answers are ranking-format or name-format."""
    ranking_count = 0
    name_count = 0
    total = 0

    with open(output_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answer = row.get("answer", "")
            total += 1
            if total > 50:
                break
            if re.search(r"1st:", answer):
                ranking_count += 1
            if re.search(r"Most\s+Likely:", answer, re.IGNORECASE):
                name_count += 1

    if ranking_count > name_count:
        return "ranking"
    return "name"


def main():
    parser = argparse.ArgumentParser(description="Evaluate LMM output accuracy")
    parser.add_argument("--output-csv", required=True, help="Output CSV from run_inference_local_hf.py")
    parser.add_argument("--test-list", required=True, help="Ground truth test list (e.g., data/semi-aves/test.txt)")
    parser.add_argument("--taxonomy-json", required=True, help="Taxonomy JSON (e.g., data/semi-aves/semi-aves_labels.json)")
    parser.add_argument("--topk-json", default=None, help="Top-k predictions JSON (required for ranking-format answers)")
    parser.add_argument("--image-dir", default=None, help="Image directory name (e.g., semi-aves)")
    parser.add_argument("--config-yaml", default=None, help="config.yml to resolve image-dir path")
    parser.add_argument("--verbose", action="store_true", help="Print per-image results")
    parser.add_argument("--format", choices=["auto", "ranking", "name"], default="auto",
                        help="Answer format: 'ranking' (1st: X), 'name' (Most Likely: ...), or 'auto'")
    args = parser.parse_args()

    # Load taxonomy
    common2id, sci2id, id2common, id2sci = build_name_to_id(Path(args.taxonomy_json))
    print(f"Loaded {len(id2common)} classes from taxonomy")

    # Load ground truth
    gt = load_ground_truth(Path(args.test_list))
    n_unique = len(set(gt.values()))  # unique labels
    print(f"Loaded {len(gt) // 2} ground truth entries ({n_unique} classes)")

    # Detect format
    fmt = args.format
    if fmt == "auto":
        fmt = detect_answer_format(args.output_csv)
        print(f"Auto-detected answer format: {fmt}")

    # Load topk index if needed
    topk_index = {}
    image_dir = Path(".")
    if fmt == "ranking":
        if not args.topk_json:
            print("ERROR: --topk-json is required for ranking-format answers")
            return
        # Resolve image_dir
        if args.image_dir:
            if args.config_yaml and os.path.exists(args.config_yaml):
                import yaml
                with open(args.config_yaml) as f:
                    cfg = yaml.safe_load(f)
                dataset_root = cfg.get("dataset_path", "")
                image_dir = Path(os.path.join(dataset_root, args.image_dir))
            else:
                image_dir = Path(args.image_dir)
        topk_index = build_topk_index(Path(args.topk_json), image_dir)
        print(f"Loaded {len(topk_index)} topk entries")

    # Parse output CSV
    total = 0
    correct = 0
    no_match = 0
    no_gt = 0
    no_topk = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    with open(args.output_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row["image_path"]
            answer = row.get("answer", "")

            # Find ground truth
            true_label = gt.get(img_path)
            if true_label is None:
                true_label = gt.get(Path(img_path).name)
            if true_label is None:
                no_gt += 1
                if args.verbose:
                    print(f"  NO GT: {img_path}")
                continue

            # Parse predicted class
            if fmt == "ranking":
                # Look up topk candidates for this image
                rec = None
                for v in _variants_for_path_str(img_path, image_dir):
                    rec = topk_index.get(v)
                    if rec:
                        break
                if not rec:
                    no_topk += 1
                    if args.verbose:
                        print(f"  NO TOPK: {img_path}")
                    continue
                topk_cls = rec.get("topk_cls", [])
                pred_label = parse_ranking_answer(answer, topk_cls)
            else:
                pred_label = parse_name_answer(answer, common2id, sci2id)

            total += 1
            per_class_total[true_label] += 1

            if pred_label == true_label:
                correct += 1
                per_class_correct[true_label] += 1
            elif pred_label == -1:
                no_match += 1
                if args.verbose:
                    true_name = id2common.get(true_label, f"class_{true_label}")
                    print(f"  NO PARSE: {img_path} | answer: {answer[:120]}")
            elif args.verbose:
                true_name = id2common.get(true_label, f"class_{true_label}")
                pred_name = id2common.get(pred_label, f"class_{pred_label}")
                print(f"  WRONG: {img_path} | true: {true_name} | pred: {pred_name}")

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total images evaluated:  {total}")
    print(f"Correct predictions:     {correct}")
    print(f"Wrong predictions:       {total - correct - no_match}")
    print(f"Unparseable answers:     {no_match}")
    if no_gt > 0:
        print(f"Missing ground truth:    {no_gt}")
    if no_topk > 0:
        print(f"Missing topk entries:    {no_topk}")
    print(f"\nOverall Accuracy:        {100 * correct / total:.2f}%" if total > 0 else "\nNo images evaluated!")

    # Per-class accuracy (mean per-class accuracy = metric used in the paper)
    if per_class_total:
        class_accs = []
        for cid in sorted(per_class_total.keys()):
            acc = per_class_correct[cid] / per_class_total[cid] if per_class_total[cid] > 0 else 0
            class_accs.append(acc)
        avg_class_acc = sum(class_accs) / len(class_accs)
        print(f"Mean Per-Class Accuracy: {100 * avg_class_acc:.2f}%  (paper metric)")
    print("=" * 50)


if __name__ == "__main__":
    main()
