"""Joint learning-rate/round-count search with an explicit multi-metric rule.

Run: python3 src/tune_xgboost_multimetric.py
Five primary CV metrics have equal rank weight; final holdout is not read.
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
from sklearn.base import clone
from sklearn.metrics import make_scorer, recall_score, roc_curve
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_logistic import metrics, write_csv, FEATURE_SET
from train_tree_models import build_pipeline
from xgb_tuning_data import load_development

ROOT = Path(__file__).resolve().parents[1]
NAME = 'xgboost_multimetric_gridsearch'
RATES = [0.02, 0.05, 0.1]
ROUNDS = [150, 300, 600, 1000]
GRID = {'classifier__learning_rate': RATES, 'classifier__n_estimators': ROUNDS}
PRIMARY = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
ALL_METRICS = [*PRIMARY, 'specificity', 'average_precision']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_ranks(results):
    return np.mean([results['rank_test_' + m] for m in PRIMARY], axis=0)


def select_candidate(results):
    ranks = mean_ranks(results)
    # Lower mean rank is better; deterministic tie breaks are specified upfront.
    return min(range(len(ranks)), key=lambda i: (
        float(ranks[i]), -float(results['mean_test_roc_auc'][i]),
        int(results['params'][i]['classifier__n_estimators']),
        float(results['params'][i]['classifier__learning_rate']), i,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    project, out = args.project_root, args.output_root or args.project_root
    report, models = out / 'reports' / NAME, out / 'models' / NAME
    figures, predictions = out / 'outputs/figures' / NAME, out / 'outputs/predictions' / NAME
    for folder in [report, models, figures, predictions]:
        if folder.exists():
            raise FileExistsError(f'Refusing to overwrite existing outputs: {folder}')
    X, y, features, records, adapter, cv, source_sha, _ = load_development(project)
    previous_path = project / 'reports/xgboost_gridsearch_round2/search_metadata.json'
    previous = json.loads(previous_path.read_text())
    assert previous['training_sha256'] == source_sha and previous['features'] == features
    fixed = previous['selected_full_parameters']
    assert fixed['learning_rate'] == 0.05 and fixed['n_estimators'] == 300
    old_model = project / 'models/xgboost_gridsearch_round2/model_pipeline.joblib'
    old_hashes = {str(p): sha256(p) for p in [previous_path, old_model]}
    base = build_pipeline('xgboost', features).set_params(**{'classifier__' + k: v for k, v in fixed.items()})
    scoring = {m: m for m in [*PRIMARY, 'average_precision']}
    scoring['specificity'] = make_scorer(recall_score, pos_label=0, zero_division=0)
    search = GridSearchCV(base, GRID, scoring=scoring, refit=select_candidate, cv=cv,
                          n_jobs=1, return_train_score=True, error_score='raise', verbose=2)
    print(f'{len(y):,} development rows; 10 original inputs; exact same five grouped folds.', flush=True)
    print('12 candidates × 5 folds = 60 fits; 5 selected-OOF replay fits; 1 full-development refit.', flush=True)
    print('Selection: equal mean ranks of AUC/accuracy/precision/recall/F1; ties: AUC, fewer trees, lower rate.', flush=True)
    search.fit(X, y)
    results = search.cv_results_
    best = int(search.best_index_)
    chosen = search.best_params_
    ranks = mean_ranks(results)
    assert best == select_candidate(results)
    count = len(results['params'])
    assert count == 12
    table, all_folds = [], []
    for i, params in enumerate(results['params']):
        row = dict(candidate_index=i, learning_rate=params['classifier__learning_rate'],
                   n_estimators=params['classifier__n_estimators'], mean_primary_rank=float(ranks[i]),
                   selected=i == best, mean_fit_seconds=float(results['mean_fit_time'][i]))
        for m in ALL_METRICS:
            row['rank_' + m] = int(results['rank_test_' + m][i])
            for phase, label in [('train', 'training'), ('test', 'validation')]:
                row[f'{label}_{m}_mean'] = float(results[f'mean_{phase}_{m}'][i])
                row[f'{label}_{m}_std_population'] = float(results[f'std_{phase}_{m}'][i])
        table.append(row)
        for fold in range(5):
            values = dict(candidate_index=i, learning_rate=row['learning_rate'], n_estimators=row['n_estimators'], fold=fold+1)
            for m in ALL_METRICS:
                values['validation_' + m] = float(results[f'split{fold}_test_{m}'][i])
                values['training_' + m] = float(results[f'split{fold}_train_{m}'][i])
            all_folds.append(values)
    values = np.array([[row[f'validation_{m}_mean'] for m in PRIMARY] for row in table])
    for i, row in enumerate(table):
        row['pareto_efficient_primary_metrics'] = not any(
            np.all(values[j] >= values[i]) and np.any(values[j] > values[i]) for j in range(count) if j != i)
    reference = next(row for row in table if row['learning_rate'] == 0.05 and row['n_estimators'] == 300)
    for m in PRIMARY:
        assert np.isclose(reference['validation_' + m + '_mean'], previous['selected_cv_fold_means'][m], atol=1e-10, rtol=0)
    table.sort(key=lambda row: (row['mean_primary_rank'], -row['validation_roc_auc_mean'], row['n_estimators'], row['learning_rate']))
    assert table[0]['candidate_index'] == best
    for folder in [report, models, figures, predictions]:
        folder.mkdir(parents=True)
    write_csv(report / 'gridsearch_results.csv', table)
    write_csv(report / 'all_cv_fold_metrics.csv', all_folds)
    metric_winners = []
    for m in ALL_METRICS:
        highest = max(row['validation_' + m + '_mean'] for row in table)
        for row in table:
            if row['validation_' + m + '_mean'] == highest:
                metric_winners.append(dict(metric=m, candidate_index=row['candidate_index'],
                    learning_rate=row['learning_rate'], n_estimators=row['n_estimators'],
                    validation_mean=highest, also_selected_by_composite=row['selected']))
    write_csv(report / 'metric_winners.csv', metric_winners)
    write_csv(report / 'pareto_candidates.csv', [row for row in table if row['pareto_efficient_primary_metrics']])

    # Recreate held-out predictions for the selected configuration, never use refit predictions for CV metrics.
    oof = np.full(len(y), np.nan)
    fold_ids = np.zeros(len(y), dtype=int)
    selected_folds = []
    for fold, (train, valid) in enumerate(cv, 1):
        estimator = clone(search.best_estimator_).fit(X[train], y[train])
        probability = estimator.predict_proba(X[valid])[:, 1]
        assert not np.any(fold_ids[valid])
        oof[valid], fold_ids[valid] = probability, fold
        scores = metrics(y[valid], probability)
        for m in ALL_METRICS:
            assert np.isclose(scores[m], results[f'split{fold-1}_test_{m}'][best], atol=1e-10, rtol=0)
        selected_folds.append(dict(fold=fold, training_records=len(train), validation_records=len(valid), **scores))
        print('SELECTED OOF fold', fold, json.dumps({m: scores[m] for m in PRIMARY}), flush=True)
    assert np.isfinite(oof).all() and np.all((oof >= 0) & (oof <= 1)) and np.all(fold_ids > 0)
    pooled = metrics(y, oof)
    selected = {m: float(results['mean_test_' + m][best]) for m in ALL_METRICS}
    write_csv(report / 'best_cv_fold_metrics.csv', selected_folds)
    write_csv(report / 'cv_summary.csv', [dict(metric=m, fold_mean=selected[m],
        fold_std_sample=float(np.std([r[m] for r in selected_folds], ddof=1)), pooled_oof=pooled[m]) for m in ALL_METRICS])
    write_csv(predictions / 'development_oof_predictions.csv', [dict(id=r['id'], profile_group=r['profile_group'],
        fold=int(fold_ids[i]), actual_cardio=int(y[i]), predicted_score=float(oof[i]), predicted_cardio=int(oof[i]>=0.5))
        for i, r in enumerate(records)])
    write_csv(report / 'oof_confusion_matrix.csv', [dict(actual_cardio=0, predicted_0=pooled['tn'], predicted_1=pooled['fp']),
                                                  dict(actual_cardio=1, predicted_0=pooled['fn'], predicted_1=pooled['tp'])])
    comparison = [dict(metric=m, previous_round2_fold_mean=previous['selected_cv_fold_means'][m],
        selected_fold_mean=selected[m], change=selected[m]-previous['selected_cv_fold_means'][m]) for m in PRIMARY]
    write_csv(report / 'previous_vs_selected.csv', comparison)
    full_params = {**fixed, **{k.removeprefix('classifier__'): v for k, v in chosen.items()}}
    model_path = models / 'model_pipeline.joblib'
    joblib.dump(dict(pipeline=search.best_estimator_, features=features, threshold=0.5, target='cardio',
        feature_set=FEATURE_SET, denormalization_params=adapter, selected_parameters=full_params,
        input_units='original units; original categorical codes'), model_path, compress=('xz', 3))
    loaded = joblib.load(model_path)
    assert np.allclose(loaded['pipeline'].predict_proba(X[:100]), search.best_estimator_.predict_proba(X[:100]))
    for key, value in full_params.items():
        assert loaded['pipeline'].named_steps['classifier'].get_params()[key] == value
    booster_path = models / 'xgboost_booster.json'
    search.best_estimator_.named_steps['classifier'].save_model(booster_path)
    native = xgboost.XGBClassifier()
    native.load_model(booster_path)
    assert np.allclose(native.predict_proba(loaded['pipeline'].named_steps['preprocess'].transform(X[:100])), loaded['pipeline'].predict_proba(X[:100]))
    metadata = dict(experiment='joint learning_rate/n_estimators multi-metric search', records=len(y), features=features,
        training_source=previous['training_source'], training_sha256=source_sha,
        previous_round2_metadata_sha256=sha256(previous_path), parameter_grid=GRID,
        fixed_parameters={k:v for k,v in fixed.items() if k not in ['learning_rate','n_estimators']},
        best_params=chosen, selected_full_parameters=full_params,
        selection_rule='minimize equally weighted mean of five validation-metric ranks; tie: higher AUC, fewer trees, lower learning_rate',
        selection_metrics=PRIMARY, metric_weights={m:0.2 for m in PRIMARY},
        tie_rank_method='minimum rank for exact ties (GridSearchCV rank_test_*); differences not significance-tested',
        selected_mean_rank=float(ranks[best]), selected_cv_fold_means=selected, pooled_oof_metrics=pooled,
        mean_training_accuracy=float(results['mean_train_accuracy'][best]),
        cv='exact same five original StratifiedGroupKFold partitions', candidate_configurations=count,
        grid_cv_fits=count*5, selected_oof_replay_fits=5, full_development_refits=1, total_fits=count*5+6,
        threshold=0.5, threshold_tuning=False, early_stopping=False, test_evaluated=False,
        test_csv_read_this_run=False, fully_cleaned_csv_read_this_run=False,
        evaluation_limit='Exploratory repeated development selection with a newly declared composite criterion; no nested CV or independent final evaluation. Full-cohort descriptive results have previously been viewed.',
        composite_limit='Metrics overlap; equal rank weights are a transparent project convention, not a clinical utility function. Rank gaps do not measure practical or statistical effect size.',
        training_code_sha256=sha256(Path(__file__)), loader_code_sha256=sha256(Path(__file__).parent/'xgb_tuning_data.py'),
        versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__))
    (report/'search_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    lookup = {(row['learning_rate'],row['n_estimators']):row for row in table}
    fig, axes = plt.subplots(1, 2, figsize=(12,5.7), layout='constrained')
    fig.get_layout_engine().set(rect=(0,.08,1,.92))
    for ax, key, title, cmap, fmt in [
        (axes[0],'mean_primary_rank','Mean rank of five metrics (lower is better)','YlGnBu_r','.2f'),
        (axes[1],'validation_roc_auc_mean','Mean five-fold validation AUC','YlGnBu','.5f')]:
        matrix = np.array([[lookup[(lr,n)][key] for n in ROUNDS] for lr in RATES])
        low, high = float(matrix.min()), float(matrix.max())
        high = high if high>low else low+1
        im = ax.imshow(matrix,cmap=cmap,vmin=low,vmax=high,aspect='auto')
        ax.set(xticks=range(len(ROUNDS)),yticks=range(len(RATES)),xticklabels=ROUNDS,yticklabels=RATES,
               xlabel='n_estimators',ylabel='learning_rate',title=title)
        for i, lr in enumerate(RATES):
            for j, n in enumerate(ROUNDS):
                rgba = im.cmap(im.norm(matrix[i,j]))
                luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                mark='\nSelected' if lookup[(lr,n)]['selected'] else ''
                ax.text(j,i,format(matrix[i,j],fmt)+mark,ha='center',va='center',fontsize=10,
                        color='white' if luminance<.5 else '#202b36')
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.text(.01,.015,'12 candidates | Fixed 10 inputs and grouped folds | Composite selection; test not scored',fontsize=9)
    for suffix in ['png','svg']:
        fig.savefig(figures/('search_summary.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.08,1,.92))
    fpr,tpr,_=roc_curve(y,oof)
    axes[0].plot(fpr,tpr,label=f'Pooled OOF AUC={pooled["roc_auc"]:.4f}',lw=2)
    axes[0].plot([0,1],[0,1],'--',color='gray')
    axes[0].set(title='Selected XGBoost: grouped OOF ROC',xlabel='False positive rate',ylabel='Recall')
    axes[0].legend(loc='lower right')
    cm=np.array([[pooled['tn'],pooled['fp']],[pooled['fn'],pooled['tp']]])
    axes[1].imshow(cm,cmap='Blues')
    axes[1].set(xticks=[0,1],yticks=[0,1],xticklabels=['Predicted 0','Predicted 1'],yticklabels=['Actual 0','Actual 1'],
                title='OOF confusion matrix; threshold=0.5')
    for i in range(2):
        for j in range(2):
            axes[1].text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=18,color='white' if cm[i,j]>cm.max()*.65 else '#26313c')
    fig.text(.01,.015,f'Selected candidate replayed on the same five folds | Development n={len(y):,} | Test not scored',fontsize=9)
    for suffix in ['png','svg']:
        fig.savefig(figures/('selected_oof_evaluation.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)

    lines=['XGBoost 多指标联合调参','',f'开发集 {len(y):,} 条；保留 10 项输入，沿用相同五折分组。',
        'learning_rate=[0.02,0.05,0.1] × n_estimators=[150,300,600,1000]：12 种组合、60 次 CV 拟合。',
        '另外重放选中组合的五折预测 5 次，生成 OOF 混淆矩阵；GridSearchCV 全开发集重拟合 1 次，总计 66 次。',
        '固定：max_depth=3、min_child_weight=10、subsample=1、colsample_bytree=0.8、reg_lambda=1。',
        '选优规则：五折验证 AUC、Accuracy、Precision、Recall、F1 各自排名，等权平均；平均名次越小越好。',
        '各指标相同分数采用相同的最小名次；综合同分时依次优先较高 AUC、较少树、较低学习率。',
        'Specificity 和 Average Precision 为辅助报告指标；混淆矩阵以计数形式报告，不与比率混合打分。',
        '这些指标存在重叠，等权排名是公开的项目折中规则，不是临床效用函数；名次差不代表差异显著或实际收益大小。',
        '训练指标仅诊断过拟合，不参与选优；分类阈值固定 0.5，未启用提前停止。',
        '选中组合：'+json.dumps(chosen),f'综合平均名次：{ranks[best]:.3f}。','', '选中组合五折验证均值：']
    lines.extend(f'{m}: {selected[m]:.6f}' for m in ALL_METRICS)
    lines.extend(['',f'各折训练准确率均值：{metadata["mean_training_accuracy"]:.3%}。','与第二轮结果比较：'])
    lines.extend(f'{row["metric"]}: {row["previous_round2_fold_mean"]:.6f} -> {row["selected_fold_mean"]:.6f}，差值 {row["change"]:+.6f}' for row in comparison)
    lines.extend(['','各单项指标的最佳候选：'])
    lines.extend(f'{row["metric"]}: learning_rate={row["learning_rate"]}, n_estimators={row["n_estimators"]}, score={row["validation_mean"]:.6f}' for row in metric_winners)
    lines.extend(['','解释与验证',
        '旧组合 learning_rate=0.05/n_estimators=300 包含在本轮搜索，已精确复现第二轮全部五项验证指标。',
        '选中组合的 OOF 重放已逐折复现网格搜索的七项指标；重放不是新的独立评估。',
        '曲线标注合并 OOF AUC；选参表格采用五折 AUC 算术均值，数值可能略有差别。',
        '旧模型在前两轮按 AUC 选择，本轮主动改为多指标折中；不要把规则改变直接解释为模型整体更强。',
        '反复使用相同开发折选参，且此前查看过全量描述性图，结果是探索性开发成绩；未做嵌套 CV。',
        '本轮没有读取最终测试 CSV 或全量清洗 CSV，没有最终测试预测成绩。',
        '旧模型与旧报告保留；新模型已全开发集重拟合并通过 Pipeline/原生 booster 重载核验。','',
        '复现：python3 src/tune_xgboost_multimetric.py；依赖沿用 requirements_tree_models_locked.txt。',
        '评分：python3 src/predict_logistic.py --model models/'+NAME+'/model_pipeline.joblib --input INPUT.csv --output predictions.csv',
        'ChatGPT 协助编写及运行代码；所有统计由本地计算并核验。'])
    text='\n'.join(lines)+'\n'
    (report/'report_zh.txt').write_text(text,encoding='utf-8')
    images=''.join('<img alt="Multi-metric search evaluation" src="data:image/png;base64,'+
        base64.b64encode((figures/name).read_bytes()).decode()+'">' for name in ['search_summary.png','selected_oof_evaluation.png'])
    header=''.join('<th>'+m+'</th>' for m in PRIMARY)
    rows=''.join('<tr><td>'+str(row['learning_rate'])+'</td><td>'+str(row['n_estimators'])+'</td><td>'+
        f'{row["mean_primary_rank"]:.2f}'+(' Selected' if row['selected'] else '')+'</td>'+
        ''.join('<td>'+f'{row["validation_"+m+"_mean"]:.6f}'+'</td>' for m in PRIMARY)+'</tr>' for row in table)
    page=('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>XGBoost 多指标联合调参</title>'
        '<style>body{font-family:-apple-system,Arial,sans-serif;max-width:1200px;margin:40px auto;line-height:1.8}'
        'img{width:100%}pre{white-space:pre-wrap;font-family:inherit}table{border-collapse:collapse;width:100%;font-size:13px}'
        'td,th{border:1px solid #ddd;padding:8px}</style><h1>XGBoost 多指标联合调参</h1>'+images+
        '<table><tr><th>learning_rate</th><th>n_estimators</th><th>平均名次 ↓</th>'+header+'</tr>'+rows+'</table><pre>'+html.escape(text)+'</pre></html>')
    (report/'report.html').write_text(page,encoding='utf-8')
    with (predictions/'development_oof_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        saved=list(csv.DictReader(f))
    assert len(saved)==len(y) and len({r['id'] for r in saved})==len(y)
    assert metrics(np.array([int(r['actual_cardio']) for r in saved]),np.array([float(r['predicted_score']) for r in saved]))==pooled
    assert all(int(r['predicted_cardio'])==int(float(r['predicted_score'])>=.5) for r in saved)
    assert len(all_folds)==60 and np.isclose(ranks[best],min(row['mean_primary_rank'] for row in table))
    for path, checksum in old_hashes.items():
        assert sha256(Path(path))==checksum
    (report/'validation.txt').write_text(
        'PASS: source hash, features and exact baseline grouped folds retained.\n'
        'PASS: 12 candidates / 60 grid CV fits; rank rule and tie breaks reconcile with selection.\n'
        'PASS: original 0.05/300 configuration reproduces all five previous validation metrics.\n'
        'PASS: selected five-fold OOF replay reproduces all seven search metrics; 66 total fits.\n'
        'PASS: exported OOF probabilities, labels and confusion matrix reconcile.\n'
        'PASS: saved Pipeline and native booster reload predictions and parameters verified.\n'
        'PASS: previous model and metadata hashes unchanged; test CSV not opened.\n',encoding='utf-8')
    print('SELECTED_PARAMETERS',json.dumps(chosen),flush=True)
    print('SELECTED_CV_MEANS',json.dumps(selected),flush=True)
    print('METRIC_WINNERS',json.dumps(metric_winners),flush=True)


if __name__=='__main__':
    main()
