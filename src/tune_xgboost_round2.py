"""XGBoost round 2: fix round-1 depth/child weight; tune row/column sampling.

Run from the project: python3 src/tune_xgboost_round2.py
Final holdout is not opened. Previous model artifacts are preserved.
"""
import argparse
import base64
import csv
import hashlib
import html
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_logistic import write_csv
from train_tree_models import build_pipeline
from xgb_tuning_data import load_development

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = [0.6, 0.8, 1.0]
GRID = {
    'classifier__subsample': OPTIONS,
    'classifier__colsample_bytree': OPTIONS,
}
METRICS = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
OUTPUT_NAME = 'xgboost_gridsearch_round2'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    project = args.project_root
    out = args.output_root or project
    report = out / 'reports' / OUTPUT_NAME
    figures = out / 'outputs/figures' / OUTPUT_NAME
    models = out / 'models' / OUTPUT_NAME
    for folder in [report, figures, models]:
        if folder.exists():
            raise FileExistsError(f'Refusing to overwrite existing results: {folder}')

    X, y, features, records, params, cv, source_sha, baseline = load_development(project)
    first_path = project / 'reports/xgboost_gridsearch/search_metadata.json'
    first = json.loads(first_path.read_text())
    assert first['training_sha256'] == source_sha and first['features'] == features
    fixed = {**first['fixed_parameters'], **{
        k.removeprefix('classifier__'): v for k, v in first['best_params'].items()
    }}
    assert fixed['max_depth'] == 3 and fixed['min_child_weight'] == 10
    assert fixed['n_estimators'] == 300 and fixed['learning_rate'] == 0.05
    assert fixed['subsample'] == 1.0 and fixed['colsample_bytree'] == 1.0
    estimator = build_pipeline('xgboost', features).set_params(**{
        'classifier__' + k: v for k, v in fixed.items()
    })
    search = GridSearchCV(
        estimator, GRID, scoring={m: m for m in METRICS}, refit='roc_auc',
        cv=cv, n_jobs=1, return_train_score=True, error_score='raise', verbose=3,
    )
    print(f'Loaded {len(y):,} development rows; exact baseline grouped folds verified.', flush=True)
    print('Fixed depth=3, min_child_weight=10, learning_rate=0.05, 300 rounds.', flush=True)
    print('Searching 9 sampling configurations × 5 folds, plus one full-development refit.', flush=True)
    search.fit(X, y)
    results = search.cv_results_
    best = search.best_index_
    chosen = search.best_params_
    selected = {m: float(results[f'mean_test_{m}'][best]) for m in METRICS}

    table, fold_results = [], []
    for i, candidate in enumerate(results['params']):
        row = {
            'candidate_index': i,
            'rank_auc': int(results['rank_test_roc_auc'][i]),
            'subsample': candidate['classifier__subsample'],
            'colsample_bytree': candidate['classifier__colsample_bytree'],
            'mean_fit_seconds': float(results['mean_fit_time'][i]),
        }
        for m in METRICS:
            for phase, label in [('train', 'training'), ('test', 'validation')]:
                row[f'{label}_{m}_mean'] = float(results[f'mean_{phase}_{m}'][i])
                row[f'{label}_{m}_std_population'] = float(results[f'std_{phase}_{m}'][i])
        table.append(row)
        for fold in range(5):
            values = {k: row[k] for k in ['candidate_index', 'subsample', 'colsample_bytree']}
            values['fold'] = fold + 1
            for m in METRICS:
                values[f'validation_{m}'] = float(results[f'split{fold}_test_{m}'][i])
                values[f'training_{m}'] = float(results[f'split{fold}_train_{m}'][i])
            fold_results.append(values)
    table.sort(key=lambda r: (r['rank_auc'], r['candidate_index']))
    reference = next(r for r in table if r['subsample'] == r['colsample_bytree'] == 1.0)
    for m in METRICS:
        assert np.isclose(reference[f'validation_{m}_mean'],
                          first['selected_cv_fold_means'][m], atol=1e-10, rtol=0)
    best_folds = [r for r in fold_results if r['candidate_index'] == best]
    assert len(table) == 9 and len(fold_results) == 45 and len(best_folds) == 5
    assert np.isclose(search.best_score_, max(r['validation_roc_auc_mean'] for r in table))
    assert np.isclose(search.best_score_, np.mean([r['validation_roc_auc'] for r in best_folds]))

    for folder in [report, figures, models]:
        folder.mkdir(parents=True)
    write_csv(report / 'gridsearch_results.csv', table)
    write_csv(report / 'all_cv_fold_metrics.csv', fold_results)
    write_csv(report / 'best_cv_fold_metrics.csv', best_folds)
    with (project / 'reports/xgboost/cv_fold_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        original = list(csv.DictReader(f))
    comparison = [{
        'metric': m,
        'original_baseline_fold_mean': float(np.mean([float(r[m]) for r in original])),
        'round1_selected_fold_mean': first['selected_cv_fold_means'][m],
        'round2_selected_fold_mean': selected[m],
        'round2_minus_round1': selected[m] - first['selected_cv_fold_means'][m],
    } for m in METRICS]
    write_csv(report / 'round_comparison.csv', comparison)
    full_params = {**fixed, **{k.removeprefix('classifier__'): v for k, v in chosen.items()}}
    model_path = models / 'model_pipeline.joblib'
    joblib.dump({
        'pipeline': search.best_estimator_, 'features': features, 'threshold': 0.5,
        'target': 'cardio', 'feature_set': baseline['feature_set'] if 'feature_set' in baseline else 'complete_logistic_bmi_only',
        'denormalization_params': params,
        'input_units': 'original units; original categorical codes',
        'selected_parameters': full_params,
    }, model_path, compress=('xz', 3))
    loaded = joblib.load(model_path)
    assert np.allclose(loaded['pipeline'].predict_proba(X[:100]),
                       search.best_estimator_.predict_proba(X[:100]))
    for key, value in full_params.items():
        assert loaded['pipeline'].named_steps['classifier'].get_params()[key] == value
    booster_path = models / 'xgboost_booster.json'
    search.best_estimator_.named_steps['classifier'].save_model(booster_path)
    native = xgboost.XGBClassifier()
    native.load_model(booster_path)
    transformed = loaded['pipeline'].named_steps['preprocess'].transform(X[:100])
    assert np.allclose(native.predict_proba(transformed), loaded['pipeline'].predict_proba(X[:100]))

    metadata = {
        'search_round': 2, 'records': len(y), 'training_source': first['training_source'],
        'training_sha256': source_sha, 'features': features,
        'round1_metadata_sha256': sha256(first_path), 'parameter_grid': GRID,
        'fixed_parameters': {k: v for k, v in fixed.items() if k not in ['subsample', 'colsample_bytree']},
        'best_params': chosen, 'selected_full_parameters': full_params,
        'selection_metric': 'mean five-fold validation ROC-AUC',
        'best_mean_cv_auc': float(search.best_score_), 'selected_cv_fold_means': selected,
        'mean_training_accuracy': float(results['mean_train_accuracy'][best]),
        'cv': 'exact same five StratifiedGroupKFold partitions as XGBoost baseline and round 1',
        'candidate_configurations': 9, 'cv_fits': 45, 'full_development_refits': 1,
        'test_evaluated': False, 'threshold': 0.5, 'threshold_tuning': False,
        'early_stopping': False,
        'training_code_sha256': sha256(Path(__file__)),
        'data_loader_code_sha256': sha256(Path(__file__).parent / 'xgb_tuning_data.py'),
        'evaluation_limit': 'Sequential selection on the same development folds; not independent performance estimation.',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__},
    }
    (report / 'search_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    lookup = {(r['subsample'], r['colsample_bytree']): r for r in table}
    matrix = np.array([[lookup[(s, c)]['validation_roc_auc_mean'] for c in OPTIONS] for s in OPTIONS])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.92))
    lo, hi = float(matrix.min()) - 0.001, float(matrix.max()) + 0.001
    im = ax.imshow(matrix, cmap='YlGnBu', vmin=lo, vmax=hi)
    ax.set(xticks=[0, 1, 2], yticks=[0, 1, 2], xticklabels=OPTIONS, yticklabels=OPTIONS,
           xlabel='colsample_bytree', ylabel='subsample', title='XGBoost round 2: mean five-fold validation AUC')
    for i, s in enumerate(OPTIONS):
        for j, c in enumerate(OPTIONS):
            marker = '\nSelected' if s == chosen['classifier__subsample'] and c == chosen['classifier__colsample_bytree'] else ''
            ax.text(j, i, f'{matrix[i,j]:.5f}' + marker, ha='center', va='center',
                    color='white' if (matrix[i,j] - lo) / (hi - lo) > 0.55 else '#26313c')
    fig.colorbar(im, ax=ax, label='Mean validation AUC')
    fig.text(0.01, 0.015, 'Fixed depth=3, child weight=10 | Selection CV; final test untouched', fontsize=9)
    for suffix in ['png', 'svg']:
        fig.savefig(figures / ('gridsearch_auc.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)

    lines = [
        'XGBoost 第二轮 GridSearchCV', '',
        f'开发集 {len(y):,} 条；固定第一轮 max_depth=3、min_child_weight=10。',
        '固定 300 轮、learning_rate=0.05、reg_lambda=1、random_state=4145；未启用提前停止。',
        '搜索 subsample=[0.6,0.8,1.0] × colsample_bytree=[0.6,0.8,1.0]，9 组 × 5 折＝45 次拟合。',
        'subsample 控制每轮使用的训练样本比例；colsample_bytree 控制每棵树使用的编码后特征比例。',
        '复用第一轮完全相同的数据、输入特征和五折资料分组，以平均验证 ROC-AUC 选择。',
        '选中参数：' + json.dumps(chosen), '', '选中组合五折验证均值：',
    ]
    lines.extend(f'{m}: {selected[m]:.6f}' for m in METRICS)
    lines.extend(['', f'选中组合各折训练准确率均值：{metadata["mean_training_accuracy"]:.2%}。',
                  '与第一轮比较，统一使用五折均值：'])
    lines.extend(f'{r["metric"]}: {r["round1_selected_fold_mean"]:.6f} -> {r["round2_selected_fold_mean"]:.6f}，差值 {r["round2_minus_round1"]:+.6f}' for r in comparison)
    lines.extend([
        '', '解释与限制',
        '1.0/1.0 对照已复现第一轮选中配置的所有验证指标。',
        '第二轮最佳仅指固定上述设置的 9 种组合，不是全局最优，也不保证显著改善。',
        '两轮重复使用相同验证折选参，得分可能偏乐观，尚不是独立泛化成绩；未做嵌套交叉验证。',
        'GridSearchCV 的 test 指标指交叉验证的验证折，不是最终测试集；折间 std 不是置信区间。',
        '模型已用全部开发集重拟合；保存完整 Pipeline 和原生 booster（后者需要配合相同预处理）。',
        '模型、原生 booster 重载后的预测均已核对。基线、第一轮和随机森林结果均保留。',
        '本次未读取最终测试 CSV；未做类别重采样或分类门槛调整，门槛仍为 0.5。',
        '连续值恢复原单位并按原始 ID 核对；类别编码位于 Pipeline 内，树模型无需缩放。',
        '外部 train/test 划分仍可能有相同健康资料跨集，本次未改变划分；内部 CV 阻止相同资料跨折。',
        '', '复现：python3 src/tune_xgboost_round2.py；依赖沿用 requirements_tree_models_locked.txt。',
        '评分：python3 src/predict_logistic.py --model models/xgboost_gridsearch_round2/model_pipeline.joblib --input INPUT.csv --output predictions.csv',
        'ChatGPT 协助编写及运行代码，所有统计结果由本地计算并核验。',
    ])
    text = '\n'.join(lines) + '\n'
    (report / 'report_zh.txt').write_text(text, encoding='utf-8')
    picture = base64.b64encode((figures / 'gridsearch_auc.png').read_bytes()).decode()
    rows = ''.join(f'<tr><td>{r["rank_auc"]}</td><td>{r["subsample"]}</td><td>{r["colsample_bytree"]}</td><td>{r["validation_roc_auc_mean"]:.6f}</td><td>{r["validation_accuracy_mean"]:.2%}</td></tr>' for r in table)
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>XGBoost 第二轮调参</title>'
            '<style>body{font-family:-apple-system,Arial,sans-serif;max-width:1100px;margin:40px auto;line-height:1.8}'
            'img{width:100%}pre{white-space:pre-wrap;font-family:inherit}table{border-collapse:collapse;width:100%}'
            'td,th{border:1px solid #ddd;padding:9px}</style><h1>XGBoost 第二轮调参</h1>'
            '<img alt="Nine sampling configurations and validation AUC" src="data:image/png;base64,' + picture + '">'
            '<table><tr><th>排名</th><th>subsample</th><th>colsample_bytree</th><th>验证 AUC 均值</th><th>验证准确率均值</th></tr>'
            + rows + '</table><pre>' + html.escape(text) + '</pre></html>')
    (report / 'report.html').write_text(page, encoding='utf-8')
    with (report / 'gridsearch_results.csv').open(encoding='utf-8-sig', newline='') as f:
        assert len(list(csv.DictReader(f))) == 9
    (report / 'validation.txt').write_text(
        'PASS: source hash, features, exact XGBoost grouped folds, and fixed round-1 settings verified.\n'
        'PASS: 9 candidates, 45 fold results, one full-development refit; highest mean AUC selected.\n'
        'PASS: 1.0/1.0 control reproduces all round-1 validation metrics.\n'
        'PASS: reloaded Pipeline and native booster predictions reconcile; saved parameters verified.\n'
        'PASS: final test CSV not opened; previous results preserved.\n', encoding='utf-8')
    print('BEST_PARAMETERS', json.dumps(chosen), flush=True)
    print('SELECTED_CV_MEANS', json.dumps(selected), flush=True)
    print('TRAIN_ACCURACY', metadata['mean_training_accuracy'], flush=True)


if __name__ == '__main__':
    main()
