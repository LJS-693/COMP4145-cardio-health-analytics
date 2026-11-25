"""Score Caroline-format CSV with saved SVM. Outputs margins, not probabilities."""
import argparse
import csv
from pathlib import Path
import joblib
import numpy as np
from train_logistic import decode_records


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=root / 'models/svm/model_pipeline.joblib')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Output must not overwrite input')
    bundle = joblib.load(args.model)
    assert bundle['score_method'] == 'decision_function'
    records = decode_records(args.input, bundle['denormalization_params'])
    X = np.array([[r[c] for c in bundle['features']] for r in records], dtype=float)
    margins = bundle['pipeline'].decision_function(X)
    assert np.isfinite(margins).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'decision_score', 'predicted_cardio'])
        writer.writeheader()
        writer.writerows(dict(id=r['id'], decision_score=float(s), predicted_cardio=int(s >= bundle['threshold']))
                         for r, s in zip(records, margins))
    print(f'Scored {len(records)} records; decision_score is not a probability.')


if __name__ == '__main__':
    main()
