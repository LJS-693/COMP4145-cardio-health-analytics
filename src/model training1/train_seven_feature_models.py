"""Seven-feature ablation using the selected round-2 RF/XGBoost parameters.

Run: python3 src/train_seven_feature_models.py
Same development records, original profile groups and five folds; no new tuning.
"""
import argparse
import base64
import csv
import hashlib
import html
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_logistic import metrics, write_csv, SEED
from xgb_tuning_data import load_development

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['age_years', 'gender', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
REMOVED = ['smoke', 'alco', 'active']
METRICS = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'specificity', 'average_precision']
PRIMARY = METRICS[:5]
NAMES = ['random_forest', 'xgboost']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pipeline(name, parameters):
    numeric = ['age_years', 'bmi', 'ap_hi', 'ap_lo']
    categories = ['gender', 'cholesterol', 'gluc']
    preprocess = ColumnTransformer([
        ('numeric', 'passthrough', [FEATURES.index(c) for c in numeric]),
        ('categories', OneHotEncoder(categories=[[1, 2], [1, 2, 3], [1, 2, 3]],
            drop='first', handle_unknown='error', sparse_output=False),
         [FEATURES.index(c) for c in categories]),
    ])
    model = RandomForestClassifier(**parameters) if name == 'random_forest' else xgboost.XGBClassifier(**parameters)
    return Pipeline([('preprocess', preprocess), ('classifier', model)])


def save_html(path, title, text, picture, comparison):
    rows = ''.join(f'<tr><td>{r["model"]}</td><td>{r["feature_count"]}</td><td>{r["roc_auc"]:.6f}</td>'
                   f'<td>{r["accuracy"]:.3%}</td><td>{r["precision"]:.3%}</td><td>{r["recall"]:.3%}</td><td>{r["f1"]:.3%}</td></tr>'
                   for r in comparison)
    encoded = base64.b64encode(picture.read_bytes()).decode()
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>' + title + '</title>'
        '<style>body{font-family:-apple-system,Arial,sans-serif;max-width:1150px;margin:40px auto;line-height:1.8}'
        'img{width:100%}pre{white-space:pre-wrap;font-family:inherit}table{border-collapse:collapse;width:100%}'
        'td,th{border:1px solid #ddd;padding:9px}</style><h1>' + title + '</h1>'
        '<p>模型对比统一使用五折验证均值。</p><table><tr><th>模型</th><th>特征数</th><th>AUC</th><th>Accuracy</th>'
        '<th>Precision</th><th>Recall</th><th>F1</th></tr>' + rows + '</table>'
        '<img alt="Out-of-fold model evaluation" src="data:image/png;base64,' + encoded + '">'
        '<pre>' + html.escape(text) + '</pre></html>')
    path.write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    project, out = args.project_root, args.output_root or args.project_root
    folders = [out / kind / (name + '_seven_features') for name in NAMES
               for kind in ['models', 'reports', 'outputs/figures', 'outputs/predictions']]
    comparison_dir = out / 'reports/seven_feature_comparison'
    comparison_figs = out / 'outputs/figures/seven_feature_comparison'
    for folder in [*folders, comparison_dir, comparison_figs]:
        if folder.exists():
            raise FileExistsError(f'Refusing to overwrite existing results: {folder}')

    X10, y, features10, records, denormalization, cv, source_sha, _ = load_development(project)
    assert set(features10) - set(FEATURES) == set(REMOVED)
    assert len(FEATURES) == 7 and not {'id', 'cardio'} & set(FEATURES)
    X = X10[:, [features10.index(c) for c in FEATURES]]
    assert np.isfinite(X).all()
    folds = np.zeros(len(y), dtype=int)
    for fold, (_, valid) in enumerate(cv, 1):
        assert not np.any(folds[valid])
        folds[valid] = fold
    assert np.all(folds > 0)
    references, parameters, preserved = {}, {}, {}
    for name in NAMES:
        reference_path = project / 'reports' / (name + '_gridsearch_round2') / 'search_metadata.json'
        reference = json.loads(reference_path.read_text())
        assert reference['training_sha256'] == source_sha and reference['features'] == features10
        assert reference['test_evaluated'] is False and reference['threshold'] == 0.5
        references[name] = reference
        parameters[name] = reference['selected_full_parameters']
        original_model = project / 'models' / (name + '_gridsearch_round2') / 'model_pipeline.joblib'
        bundle = joblib.load(original_model)
        assert bundle['features'] == features10
        for key, value in parameters[name].items():
            assert bundle['pipeline'].named_steps['classifier'].get_params()[key] == value
        preserved[str(reference_path)] = sha256(reference_path)
        preserved[str(original_model)] = sha256(original_model)
        with (project / 'outputs/predictions' / name / 'development_oof_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
            old_folds = {r['id']: int(r['fold']) for r in csv.DictReader(f)}
        assert all(old_folds[r['id']] == folds[i] for i, r in enumerate(records))
    print(f'{len(y):,} records, fixed 7 features: {FEATURES}', flush=True)
    print('Same five original profile-group folds; fixed selected round-2 model parameters.', flush=True)
    all_comparisons, changes, seven_probabilities, model_metadata = [], [], {}, {}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    for name in NAMES:
        suffix = name + '_seven_features'
        report, models = out / 'reports' / suffix, out / 'models' / suffix
        figures, predictions = out / 'outputs/figures' / suffix, out / 'outputs/predictions' / suffix
        print('START', name, json.dumps(parameters[name]), flush=True)
        oof = np.full(len(y), np.nan)
        fold_metrics, durations = [], []
        for fold, (train, valid) in enumerate(cv, 1):
            start = time.monotonic()
            pipeline = build_pipeline(name, parameters[name]).fit(X[train], y[train])
            probabilities = pipeline.predict_proba(X[valid])[:, 1]
            assert np.isfinite(probabilities).all()
            assert np.all((probabilities >= 0) & (probabilities <= 1))
            assert np.array_equal(pipeline.classes_, [0, 1])
            oof[valid] = probabilities
            validation = metrics(y[valid], probabilities)
            training = metrics(y[train], pipeline.predict_proba(X[train])[:, 1])
            fold_metrics.append(dict(fold=fold, training_records=len(train), validation_records=len(valid),
                training_accuracy=training['accuracy'], training_auc=training['roc_auc'], **validation))
            durations.append(time.monotonic() - start)
            print(name, f'fold {fold}/5', json.dumps({m: validation[m] for m in PRIMARY}), flush=True)
        assert np.isfinite(oof).all()
        pooled = metrics(y, oof)
        means = {m: float(np.mean([r[m] for r in fold_metrics])) for m in METRICS}
        seven_probabilities[name] = oof
        for folder in [report, models, figures, predictions]:
            folder.mkdir(parents=True)
        write_csv(report / 'cv_fold_metrics.csv', fold_metrics)
        write_csv(report / 'cv_summary.csv', [dict(metric=m, fold_mean=means[m],
            fold_std_sample=float(np.std([r[m] for r in fold_metrics], ddof=1)), pooled_oof=pooled[m]) for m in METRICS])
        write_csv(predictions / 'development_oof_predictions.csv', [dict(id=r['id'], profile_group=r['profile_group'],
            fold=int(folds[i]), actual_cardio=int(y[i]), predicted_score=float(oof[i]), predicted_cardio=int(oof[i] >= 0.5))
            for i, r in enumerate(records)])
        write_csv(report / 'oof_confusion_matrix.csv', [dict(actual_cardio=0, predicted_0=pooled['tn'], predicted_1=pooled['fp']),
                                                      dict(actual_cardio=1, predicted_0=pooled['fn'], predicted_1=pooled['tp'])])
        print('REFIT all development records:', name, flush=True)
        final = build_pipeline(name, parameters[name]).fit(X, y)
        assert final.named_steps['preprocess'].transform(X[:2]).shape[1] == 9
        model_path = models / 'model_pipeline.joblib'
        joblib.dump(dict(pipeline=final, features=FEATURES, threshold=0.5, target='cardio',
            feature_set='seven_features_no_lifestyle', denormalization_params=denormalization,
            input_units='original units; gender 1/2; cholesterol and gluc 1/2/3',
            selected_parameters=parameters[name]), model_path, compress=('xz', 3))
        loaded = joblib.load(model_path)
        assert np.allclose(loaded['pipeline'].predict_proba(X[:100]), final.predict_proba(X[:100]))
        for key, value in parameters[name].items():
            assert loaded['pipeline'].named_steps['classifier'].get_params()[key] == value
        if name == 'xgboost':
            booster_path = models / 'xgboost_booster.json'
            final.named_steps['classifier'].save_model(booster_path)
            native = xgboost.XGBClassifier()
            native.load_model(booster_path)
            assert np.allclose(native.predict_proba(final.named_steps['preprocess'].transform(X[:100])), final.predict_proba(X[:100]))

        original_means = references[name]['selected_cv_fold_means']
        comparison = [dict(model=name, feature_count=10, **{m: original_means[m] for m in PRIMARY}),
                      dict(model=name, feature_count=7, **{m: means[m] for m in PRIMARY})]
        all_comparisons.extend(comparison)
        write_csv(report / 'ten_vs_seven_fold_means.csv', comparison)
        differences = [dict(model=name, metric=m, ten_features=original_means[m], seven_features=means[m],
                            seven_minus_ten=means[m] - original_means[m]) for m in PRIMARY]
        changes.extend(differences)
        write_csv(report / 'metric_changes.csv', differences)
        metadata = dict(model=name, experiment='fixed-parameter seven-feature ablation', records=len(y), features=FEATURES,
            removed_features=REMOVED, encoded_feature_count=9, original_encoded_feature_count=12,
            parameters=parameters[name], source_parameter_metadata='reports/' + name + '_gridsearch_round2/search_metadata.json',
            source_parameter_metadata_sha256=preserved[str(project / 'reports' / (name + '_gridsearch_round2') / 'search_metadata.json')],
            selected_cv_fold_means=means, pooled_oof_metrics=pooled,
            mean_training_accuracy=float(np.mean([r['training_accuracy'] for r in fold_metrics])),
            threshold=0.5, cv='exact same original five StratifiedGroupKFold partitions',
            grouping='unchanged raw 11-predictor profile groups; not regrouped after dropping features',
            cv_fits=5, full_development_refits=1, hyperparameter_tuning=False, threshold_tuning=False, early_stopping=False,
            test_evaluated=False, test_csv_read_this_run=False,
            subset_motivation='Exploratory, pre-specified seven-feature subset after previously viewed full-cohort descriptive charts.',
            evaluation_limit='Repeated development comparisons and prior full-cohort exploration; not independent final performance estimation.',
            training_source=references[name]['training_source'], training_sha256=source_sha,
            code_sha256=sha256(Path(__file__)), loader_code_sha256=sha256(Path(__file__).parent / 'xgb_tuning_data.py'),
            preprocessing='canonical original units, same categorical levels and drop-first encoding, no scaling',
            random_state=SEED, fold_seconds=durations,
            versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__))
        model_metadata[name] = metadata
        (report / 'model_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
        fig.get_layout_engine().set(rect=(0, .08, 1, .92))
        fpr, tpr, _ = roc_curve(y, oof)
        axes[0].plot(fpr, tpr, label=f'Pooled OOF AUC={pooled["roc_auc"]:.4f}', lw=2)
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[0].set(title=name + ': seven-feature OOF ROC', xlabel='False positive rate', ylabel='Recall')
        axes[0].legend(loc='lower right')
        cm = np.array([[pooled['tn'], pooled['fp']], [pooled['fn'], pooled['tp']]])
        axes[1].imshow(cm, cmap='Blues')
        axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'], title='OOF confusion matrix; threshold=0.5')
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=18,
                             color='white' if cm[i,j] > cm.max()*.65 else '#26313c')
        fig.text(.01, .015, f'Fixed round-2 parameters | Same grouped folds | Development n={len(y):,} | Test not scored', fontsize=9)
        for suffix in ['png', 'svg']:
            fig.savefig(figures / ('oof_evaluation.' + suffix), dpi=180, bbox_inches='tight')
        plt.close(fig)
        lines = [name + ' 七项特征实验', '', f'开发集 {len(y):,} 条；固定该模型第二轮选中参数。',
            '输入：' + ', '.join(FEATURES), '删除：smoke、alco、active。',
            '参数：' + json.dumps(parameters[name]),
            '保留相同记录、原始资料分组、五折划分、类别编码和 0.5 分类阈值；没有再次调参。',
            '7 项原始特征经 one-hot 编码后为 9 列；10 项对照经编码后为 12 列。',
            '采样比例等参数保持原值，候选特征数随删除变量而改变。', '', '七项特征五折验证均值：']
        lines.extend(f'{m}: {means[m]:.6f}' for m in METRICS)
        lines.extend(['', '与十项特征的同口径对比：'])
        lines.extend(f'{r["metric"]}: {r["ten_features"]:.6f} -> {r["seven_features"]:.6f}，差值 {r["seven_minus_ten"]:+.6f}' for r in differences)
        lines.extend(['', '说明',
            '十项结果来自已保存的第二轮五折均值，本次核对了源数据哈希、全部参数、输入和逐条折号。',
            'ROC 图标注合并 OOF AUC；表格采用五折 AUC 算术均值，两者略有差别。',
            '新模型在全部开发记录重拟合保存；模型重载及参数已核验。',
            '本次未读取测试 CSV 或全量清洗 CSV；没有对最终测试集计算预测成绩。',
            '特征方案受此前全量描述图启发，且重复使用开发折比较，因此属于探索性开发结果，不能称独立最终测试成绩。',
            '没有进行嵌套交叉验证；小幅指标差异不代表稳定或显著改善。',
            '十项模型、调参结果及其他模型均保留。', '',
            '复现：python3 src/train_seven_feature_models.py；依赖沿用 requirements_tree_models_locked.txt。',
            '评分：python3 src/predict_logistic.py --model models/' + suffix + '/model_pipeline.joblib --input INPUT.csv --output predictions.csv',
            'ChatGPT 协助编写及运行；所有数值由本地计算并核验。'])
        text = '\n'.join(lines) + '\n'
        (report / 'report_zh.txt').write_text(text, encoding='utf-8')
        save_html(report / 'report.html', name + '：七项特征与十项特征比较', text, figures / 'oof_evaluation.png', comparison)
        with (predictions / 'development_oof_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
            saved = list(csv.DictReader(f))
        assert len(saved) == len(y) and len({r['id'] for r in saved}) == len(y)
        assert metrics(np.array([int(r['actual_cardio']) for r in saved]), np.array([float(r['predicted_score']) for r in saved])) == pooled
        assert all(int(r['fold']) == folds[i] and r['id'] == records[i]['id'] for i, r in enumerate(saved))
        assert all(int(r['predicted_cardio']) == int(float(r['predicted_score']) >= .5) for r in saved)
        (report / 'validation.txt').write_text(
            'PASS: exact original records, canonical inputs and fold assignments retained.\n'
            'PASS: only smoke/alco/active removed; seven named inputs become nine encoded columns.\n'
            'PASS: all selected round-2 classifier parameters unchanged; no new tuning or threshold selection.\n'
            'PASS: every development record has one OOF probability; exported labels, folds and metrics reconcile.\n'
            'PASS: model reload probabilities and saved parameters verified.\n'
            'PASS: test and fully-cleaned CSV not opened in this run; original outputs preserved.\n', encoding='utf-8')
        print('DONE', name, json.dumps(means), flush=True)

    comparison_dir.mkdir(parents=True)
    comparison_figs.mkdir(parents=True)
    write_csv(comparison_dir / 'ten_vs_seven_fold_means.csv', all_comparisons)
    write_csv(comparison_dir / 'metric_changes.csv', changes)
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .08, 1, .92))
    for name, probabilities in seven_probabilities.items():
        fpr, tpr, _ = roc_curve(y, probabilities)
        auc = metrics(y, probabilities)['roc_auc']
        ax.plot(fpr, tpr, label=f'{name}: pooled OOF AUC={auc:.4f}', lw=2)
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(title='Seven-feature models: grouped OOF ROC', xlabel='False positive rate', ylabel='Recall')
    ax.legend(loc='lower right')
    fig.text(.01, .015, 'Same development records and folds | Fixed round-2 parameters | Test not scored', fontsize=9)
    for suffix in ['png', 'svg']:
        fig.savefig(comparison_figs / ('seven_feature_roc.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    text = '七项特征消融实验\n\n' + '输入：' + ', '.join(FEATURES) + '\n\n'
    text += '\n'.join(f'{r["model"]}，{r["feature_count"]} 项：AUC={r["roc_auc"]:.6f}；Accuracy={r["accuracy"]:.3%}；Recall={r["recall"]:.3%}；F1={r["f1"]:.3%}' for r in all_comparisons)
    text += '\n\n全部采用五折验证均值；本次总计 10 次 CV 拟合及 2 次全开发集重拟合。\n'
    text += '十项结果来自第二轮保存报告，七项结果来自本次相同折的固定参数实验。\n'
    text += '保留全部旧模型；未读取测试 CSV。本实验属于重复开发比较，且特征选择受已查看的全量描述性分析启发。\n'
    text += '小幅差异不能证明稳定提升；参数在十项特征上选中，七项特征没有重新调参。\n'
    (comparison_dir / 'report_zh.txt').write_text(text, encoding='utf-8')
    save_html(comparison_dir / 'report.html', 'Random Forest / XGBoost：七项特征实验', text,
              comparison_figs / 'seven_feature_roc.png', all_comparisons)
    for path, checksum in preserved.items():
        assert sha256(Path(path)) == checksum
    (comparison_dir / 'experiment_metadata.json').write_text(json.dumps(dict(features=FEATURES, removed=REMOVED,
        training_sha256=source_sha, records=len(y), cv_fits=10, full_development_refits=2,
        fixed_parameters=parameters, old_artifact_hashes=preserved, test_evaluated=False,
        code_sha256=sha256(Path(__file__))), ensure_ascii=False, indent=2), encoding='utf-8')
    (comparison_dir / 'validation.txt').write_text(
        'PASS: two seven-feature models completed; ten total CV fits plus two full-development refits.\n'
        'PASS: identical development records and folds; all selected original hyperparameters retained.\n'
        'PASS: original round-2 model and metadata hashes unchanged.\n'
        'PASS: all individual-model OOF exports and saved-model checks passed.\n', encoding='utf-8')
    print('EXPERIMENT_COMPLETE', json.dumps(all_comparisons), flush=True)


if __name__ == '__main__':
    main()
