"""RBF SVM baseline: identical grouped development folds; final holdout untouched.

Run: python3 src/train_svm.py
Outputs use uncalibrated decision margins, not probabilities.
"""
import argparse
import base64
import csv
import hashlib
import html
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix, roc_curve)
from sklearn.svm import SVC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_logistic import build_pipeline as logistic_pipeline, write_csv, FEATURE_SET, SEED
from xgb_tuning_data import load_development

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = dict(kernel='rbf', C=1.0, gamma='scale', tol=0.001,
                  shrinking=True, cache_size=512, max_iter=-1,
                  probability=False, class_weight=None, random_state=SEED)
METRICS = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'specificity', 'average_precision']
NUMERIC = ['age_years', 'bmi', 'ap_hi', 'ap_lo']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pipeline(features):
    return logistic_pipeline(features).set_params(classifier=SVC(**PARAMETERS))


def metrics(y, margins):
    predicted = (margins >= 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y, margins)),
                accuracy=float(accuracy_score(y, predicted)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                f1=float(f1_score(y, predicted, zero_division=0)),
                specificity=float(tn / (tn + fp)),
                average_precision=float(average_precision_score(y, margins)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def fit_checked(X, y, features, label):
    model = build_pipeline(features)
    start = time.monotonic()
    print(f'{label}: fitting {len(y):,} records...', flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X, y)
    assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), label
    classifier = model.named_steps['classifier']
    assert classifier.fit_status_ == 0
    scaler = model.named_steps['preprocess'].named_transformers_['numeric']
    assert np.allclose(scaler.mean_, X[:, [features.index(c) for c in NUMERIC]].mean(axis=0))
    details = dict(fit_seconds=time.monotonic() - start,
                   support_vectors_per_class=classifier.n_support_.tolist(),
                   iterations=classifier.n_iter_.tolist(),
                   fitted_gamma=float(classifier._gamma), fit_status=int(classifier.fit_status_))
    print(f'{label}: converged in {details["fit_seconds"]:.1f}s; support vectors={sum(classifier.n_support_):,}.', flush=True)
    return model, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    project, out = args.project_root, args.output_root or args.project_root
    report, models = out / 'reports/svm', out / 'models/svm'
    figures, predictions = out / 'outputs/figures/svm', out / 'outputs/predictions/svm'
    for folder in [report, models, figures, predictions]:
        if folder.exists():
            raise FileExistsError(f'Refusing to overwrite: {folder}')
    X, y, features, records, params, cv, source_sha, baseline = load_development(project)
    config = json.loads((project / 'config/feature_sets.json').read_text())
    assert not set(features) & set(config['exclude_from_X'])
    # Also verify LR fold assignments, since its preprocessing is reused here.
    with (project / 'outputs/predictions/logistic_regression/development_oof_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        lr_folds = {r['id']: int(r['fold']) for r in csv.DictReader(f)}
    assert len(lr_folds) == len(y)
    oof = np.full(len(y), np.nan)
    fold_ids = np.zeros(len(y), dtype=int)
    fold_rows, convergence = [], []
    print(f'{len(y):,} development records; same 10 inputs and grouped five-fold partitions.', flush=True)
    for fold, (train, valid) in enumerate(cv, 1):
        assert all(lr_folds[records[i]['id']] == fold for i in valid)
        assert not np.any(fold_ids[valid])
        model, details = fit_checked(X[train], y[train], features, f'Fold {fold}/5')
        margins = model.decision_function(X[valid])
        assert np.array_equal((margins >= 0).astype(int), model.predict(X[valid]))
        oof[valid], fold_ids[valid] = margins, fold
        scores = metrics(y[valid], margins)
        fold_rows.append(dict(model='svm_rbf', fold=fold, training_records=len(train),
                              validation_records=len(valid), **scores))
        convergence.append(dict(fold=fold, **details))
        print(f'Fold {fold}/5 validation: {json.dumps(scores)}', flush=True)
    assert np.isfinite(oof).all() and np.all(fold_ids > 0)
    pooled = metrics(y, oof)
    means = {m: float(np.mean([r[m] for r in fold_rows])) for m in METRICS}
    for folder in [report, models, figures, predictions]:
        folder.mkdir(parents=True)
    write_csv(report / 'cv_fold_metrics.csv', fold_rows)
    write_csv(report / 'cv_summary.csv', [dict(metric=m, fold_mean=means[m],
        fold_std_sample=float(np.std([r[m] for r in fold_rows], ddof=1)), pooled_oof=pooled[m]) for m in METRICS])
    write_csv(predictions / 'development_oof_predictions.csv', [dict(
        id=r['id'], profile_group=r['profile_group'], fold=int(fold_ids[i]),
        actual_cardio=int(y[i]), decision_score=float(oof[i]), predicted_cardio=int(oof[i] >= 0)
    ) for i, r in enumerate(records)])
    write_csv(report / 'oof_confusion_matrix.csv', [
        dict(actual_cardio=0, predicted_0=pooled['tn'], predicted_1=pooled['fp']),
        dict(actual_cardio=1, predicted_0=pooled['fn'], predicted_1=pooled['tp'])])

    final, refit = fit_checked(X, y, features, 'Full-development refit')
    model_path = models / 'model_pipeline.joblib'
    joblib.dump(dict(pipeline=final, features=features, threshold=0.0,
                    score_method='decision_function', score_type='uncalibrated_margin',
                    target='cardio', feature_set=FEATURE_SET, denormalization_params=params,
                    input_units='original units; original categorical codes', parameters=PARAMETERS),
                model_path, compress=('xz', 3))
    loaded = joblib.load(model_path)
    assert np.allclose(loaded['pipeline'].decision_function(X[:100]), final.decision_function(X[:100]))
    assert np.array_equal(loaded['pipeline'].predict(X[:100]), final.predict(X[:100]))
    write_csv(report / 'scaling_parameters.csv', [dict(field=c, development_mean=float(mean), development_scale_sd=float(sd))
        for c, mean, sd in zip(NUMERIC, final.named_steps['preprocess'].named_transformers_['numeric'].mean_,
                              final.named_steps['preprocess'].named_transformers_['numeric'].scale_)])

    comparison = []
    for name, folder in [('Logistic regression baseline', 'logistic_regression'),
                          ('Random forest baseline', 'random_forest'), ('XGBoost baseline', 'xgboost')]:
        with (project / 'reports' / folder / 'cv_fold_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = [r for r in csv.DictReader(f) if r.get('model', folder) == folder]
        assert len(rows) == 5
        comparison.append(dict(model=name, **{m: float(np.mean([float(r[m]) for r in rows]))
                                               for m in METRICS[:5]}))
    for name, folder in [('Random forest round 1', 'random_forest_gridsearch'),
                         ('Random forest round 2', 'random_forest_gridsearch_round2'),
                         ('XGBoost round 1', 'xgboost_gridsearch'), ('XGBoost round 2', 'xgboost_gridsearch_round2')]:
        meta = json.loads((project / 'reports' / folder / 'search_metadata.json').read_text())
        assert meta['training_sha256'] == source_sha and meta['features'] == features
        comparison.append(dict(model=name, **{m: meta['selected_cv_fold_means'][m] for m in METRICS[:5]}))
    comparison.append(dict(model='SVM RBF baseline', **{m: means[m] for m in METRICS[:5]}))
    comparison.sort(key=lambda r: r['roc_auc'], reverse=True)
    write_csv(report / 'model_comparison_fold_means.csv', comparison)
    metadata = dict(model='SVC', feature_set=FEATURE_SET, features=features, records=len(y),
        target='cardio', threshold=0.0, score_type='uncalibrated decision_function margin; not probability',
        parameters=PARAMETERS, selected_cv_fold_means=means, pooled_oof_metrics=pooled,
        cv='same five StratifiedGroupKFold partitions as LR, RF and XGBoost',
        fold_convergence=convergence, full_development_refit=refit,
        cv_fits=5, full_development_refits=1,
        training_source='data/processed/caroline_cleaned version/cardio_train_80.csv', training_sha256=source_sha,
        training_code_sha256=sha256(Path(__file__)), data_loader_code_sha256=sha256(Path(__file__).parent / 'xgb_tuning_data.py'),
        preprocessing_code_sha256=sha256(Path(__file__).parent / 'train_logistic.py'),
        test_evaluated=False, hyperparameter_tuning=False, threshold_tuning=False,
        scaling='numeric StandardScaler fitted only inside each training fold',
        encoding='predefined one-hot categories for gender/cholesterol/gluc; binary passthrough',
        versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__))
    (report / 'model_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .08, 1, .92))
    fpr, tpr, _ = roc_curve(y, oof)
    axes[0].plot(fpr, tpr, label=f'SVM pooled OOF AUC={pooled["roc_auc"]:.4f}', lw=2)
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='RBF SVM: out-of-fold ROC')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=.2)
    cm = np.array([[pooled['tn'], pooled['fp']], [pooled['fn'], pooled['tp']]])
    axes[1].imshow(cm, cmap='Blues')
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'], title='OOF confusion matrix (margin cutoff=0)')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=18,
                         color='white' if cm[i,j] > cm.max()*.65 else '#26313c')
    fig.text(.01, .015, f'Identical five grouped folds | Development n={len(y):,} | Final test untouched', fontsize=10)
    for suffix in ['png', 'svg']:
        fig.savefig(figures / ('svm_oof_evaluation.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    lines = ['SVM（RBF 核）基线模型报告', '', f'开发集 {len(y):,} 条，10 项输入；相同五折分组验证。',
        '参数：C=1.0、gamma=scale、kernel=rbf、tol=0.001；未调参。',
        '连续变量恢复原单位后，在每折训练部分拟合 StandardScaler；类别编码与逻辑回归一致。',
        '没有使用最终测试集；没有类别重采样或分类门槛调优。',
        '输出为 decision_function 决策分数，分数不是概率，也不限制在 0–1。',
        '默认分数 >= 0 判为 cardio=1；ROC/AUC 使用连续决策分数，无需开启概率拟合。', '', '五折验证均值（模型比较使用此口径）：']
    lines.extend(f'{m}: {means[m]:.6f}' for m in METRICS)
    lines.extend(['', '合并 OOF 结果：'] + [f'{m}: {pooled[m]:.6f}' for m in METRICS])
    lines.extend(['', f'混淆矩阵：TN={pooled["tn"]:,}、FP={pooled["fp"]:,}、FN={pooled["fn"]:,}、TP={pooled["tp"]:,}。',
        '各折分数来自不同 SVM，合并分数的尺度可能不同；模型比较统一使用五折 AUC 均值。',
        '已有树模型经过参数选择；本次 SVM 是固定参数基线，不是各模型同等搜索预算的最终比较。',
        '五折及全开发集重拟合均收敛；完整模型保存预处理、类别编码和 SVM。',
        '不同训练折的 scaler 已核对；每条开发记录恰有一次未参与训练的 OOF 分数。',
        '外部 train/test 原划分可能存在相同健康资料跨集，本次未改动；内部 CV 保证资料组不跨折。', '',
        '复现：python3 src/train_svm.py；依赖沿用 requirements_logistic_locked.txt。',
        '评分：python3 src/predict_svm.py --input INPUT.csv --output predictions.csv',
        '模型输出是现有疾病标签的分类分数，不是未来发病概率。',
        'ChatGPT 协助编写与运行；所有结果由本地计算并核验。'])
    text = '\n'.join(lines) + '\n'
    (report / 'report_zh.txt').write_text(text, encoding='utf-8')
    picture = base64.b64encode((figures / 'svm_oof_evaluation.png').read_bytes()).decode()
    rows = ''.join(f'<tr><td>{r["model"]}</td><td>{r["roc_auc"]:.6f}</td><td>{r["accuracy"]:.2%}</td><td>{r["recall"]:.2%}</td><td>{r["f1"]:.2%}</td></tr>' for r in comparison)
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SVM 基线报告</title>'
        '<style>body{font-family:-apple-system,Arial,sans-serif;max-width:1150px;margin:40px auto;line-height:1.8}'
        'img{width:100%}pre{white-space:pre-wrap;font-family:inherit}table{border-collapse:collapse;width:100%}'
        'td,th{border:1px solid #ddd;padding:9px}</style><h1>SVM（RBF 核）基线报告</h1>'
        '<img alt="SVM OOF ROC and confusion matrix" src="data:image/png;base64,' + picture + '">'
        '<h2>模型比较：五折验证均值</h2><table><tr><th>模型</th><th>AUC</th><th>Accuracy</th><th>Recall</th><th>F1</th></tr>'
        + rows + '</table><pre>' + html.escape(text) + '</pre></html>')
    (report / 'report.html').write_text(page, encoding='utf-8')
    with (predictions / 'development_oof_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == len(y)
    saved_y = np.array([int(r['actual_cardio']) for r in saved])
    saved_scores = np.array([float(r['decision_score']) for r in saved])
    assert metrics(saved_y, saved_scores) == pooled
    assert all(int(r['predicted_cardio']) == int(float(r['decision_score']) >= 0) for r in saved)
    (report / 'validation.txt').write_text(
        'PASS: canonical data, feature whitelist, exact baseline grouped folds; final test CSV not opened.\n'
        'PASS: each fold scaler fitted on training portion only; all six SVM fits converged.\n'
        'PASS: all development records have exactly one OOF score; exports reconcile with metrics.\n'
        'PASS: saved Pipeline reload reproduces margins and labels; outputs are margins, not probabilities.\n'
        'PASS: previous model artifacts preserved.\n', encoding='utf-8')
    print('FOLD_MEANS', json.dumps(means), flush=True)
    print('POOLED_OOF', json.dumps(pooled), flush=True)
    print('SAVED_MODEL', str(model_path), flush=True)


if __name__ == '__main__':
    main()
