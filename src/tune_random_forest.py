"""GridSearchCV for RF depth/leaf size; preserve baseline and final holdout.
Nine configurations, five identical grouped folds, plus one full-development refit.
Run: python3 src/tune_random_forest.py
"""
import argparse, csv, json, hashlib, base64, html, platform
from pathlib import Path
import numpy as np
import joblib, sklearn
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_tree_models import build_pipeline, MODEL_CONFIGS
from train_logistic import decode_records, write_csv, DATA_DIR, FEATURE_SET, SEED

ROOT=Path(__file__).resolve().parents[1]
GRID={'classifier__max_depth':[6,10,14], 'classifier__min_samples_leaf':[10,30,60]}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args(); project=args.project_root; out=args.output_root or project
    data=project/DATA_DIR/'cardio_train_80.csv'
    params=json.loads((project/DATA_DIR/'denormalization_params.json').read_text())
    features=json.loads((project/'config/feature_sets.json').read_text())[FEATURE_SET]
    records=decode_records(data,params)
    with (project/'data/raw/cardio_train.csv').open(encoding='utf-8-sig',newline='') as f:
        raw={r['id']:r for r in csv.DictReader(f,delimiter=';')}
    profile_fields=['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
    for r in records:
        original=raw[r['id']]
        expected={c:float(original[c]) for c in ['height','weight','ap_hi','ap_lo']}
        expected['age_years']=float(original['age'])/365.25
        expected['bmi']=float(original['weight'])/(float(original['height'])/100)**2
        for c,v in expected.items(): assert abs(r[c]-v)<1e-8; r[c]=v
        for c in ['gender','cholesterol','gluc','smoke','alco','active','cardio']:
            assert r[c]==int(original[c])
        sig=json.dumps([float(original[c]) for c in profile_fields],separators=(',',':'))
        r['profile_group']=hashlib.sha256(sig.encode()).hexdigest()
    X=np.array([[r[c] for c in features] for r in records],dtype=float)
    y=np.array([r['cardio'] for r in records]); groups=np.array([r['profile_group'] for r in records])
    assert np.isfinite(X).all() and len({r['id'] for r in records})==len(records)
    baseline=json.loads((project/'reports/random_forest/model_metadata.json').read_text())
    sha=hashlib.sha256(data.read_bytes()).hexdigest()
    assert baseline['features']==features and baseline['training_sha256']==sha
    with (project/'outputs/predictions/random_forest/development_oof_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        baseline_folds={r['id']:int(r['fold']) for r in csv.DictReader(f)}
    cv=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups))
    fold_assign=np.zeros(len(y),int)
    for fold,(tr,va) in enumerate(cv,1):
        assert not set(groups[tr])&set(groups[va])
        assert not np.any(fold_assign[va]); fold_assign[va]=fold
        assert all(baseline_folds[records[i]['id']]==fold for i in va)
    assert np.all(fold_assign>0)
    scoring={'roc_auc':'roc_auc','accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1'}
    search=GridSearchCV(build_pipeline('random_forest',features),param_grid=GRID,
        scoring=scoring,refit='roc_auc',cv=cv,n_jobs=1,return_train_score=True,
        error_score='raise',verbose=3)
    print('Searching 9 combinations × 5 grouped folds; 300 trees unchanged.',flush=True)
    search.fit(X,y)
    results=search.cv_results_; best=search.best_index_
    report=out/'reports/random_forest_gridsearch'; models=out/'models/random_forest_gridsearch'
    figs=out/'outputs/figures/random_forest_gridsearch'
    for d in [report,models,figs]: d.mkdir(parents=True,exist_ok=True)
    table=[]
    for i,param in enumerate(results['params']):
        row={'rank_auc':int(results['rank_test_roc_auc'][i]),
             'max_depth':param['classifier__max_depth'],'min_samples_leaf':param['classifier__min_samples_leaf'],
             'mean_fit_seconds':float(results['mean_fit_time'][i])}
        for metric in scoring:
            for phase in ['train','test']:
                # sklearn calls held-out CV folds 'test'; these are NOT final test.csv.
                display='validation' if phase=='test' else 'training'
                row[f'{display}_{metric}_mean']=float(results[f'mean_{phase}_{metric}'][i])
                row[f'{display}_{metric}_std_population']=float(results[f'std_{phase}_{metric}'][i])
        table.append(row)
    table.sort(key=lambda r:(r['rank_auc'],r['max_depth'],r['min_samples_leaf']))
    write_csv(report/'gridsearch_results.csv',table)
    best_folds=[]
    for fold in range(5):
        row={'fold':fold+1}
        for metric in scoring:
            row[f'validation_{metric}']=float(results[f'split{fold}_test_{metric}'][best])
            row[f'training_{metric}']=float(results[f'split{fold}_train_{metric}'][best])
        best_folds.append(row)
    write_csv(report/'best_cv_fold_metrics.csv',best_folds)
    with (project/'reports/random_forest/cv_fold_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        base_folds=list(csv.DictReader(f))
    comparison=[]
    for metric in scoring:
        previous=float(np.mean([float(r[metric]) for r in base_folds]))
        current=float(results[f'mean_test_{metric}'][best])
        comparison.append({'metric':metric,'baseline_fold_mean':previous,
            'selected_config_fold_mean':current,'difference':current-previous})
    write_csv(report/'baseline_vs_selected.csv',comparison)
    model_path=models/'model_pipeline.joblib'
    joblib.dump({'pipeline':search.best_estimator_,'features':features,'threshold':.5,'target':'cardio',
        'feature_set':FEATURE_SET,'denormalization_params':params,
        'input_units':'original units; original 1/2 gender and 1/2/3 cholesterol/gluc codes',
        'selected_parameters':search.best_params_},model_path,compress=('xz',3))
    loaded=joblib.load(model_path)
    assert np.allclose(loaded['pipeline'].predict_proba(X[:100]),search.best_estimator_.predict_proba(X[:100]))
    selection_metrics={metric:float(results[f'mean_test_{metric}'][best]) for metric in scoring}
    metadata={'records':len(y),'training_source':str(DATA_DIR/'cardio_train_80.csv'),
        'training_sha256':sha,'features':features,'parameter_grid':GRID,
        'fixed_parameters':{k:v for k,v in MODEL_CONFIGS['random_forest'].items() if k not in ['max_depth','min_samples_leaf']},
        'best_params':search.best_params_,'selection_metric':'mean validation ROC-AUC',
        'best_mean_cv_auc':float(search.best_score_),'selected_cv_fold_means':selection_metrics,
        'mean_training_accuracy':float(results['mean_train_accuracy'][best]),
        'cv':'5-fold StratifiedGroupKFold, exact same fold assignments as RF baseline',
        'random_state':SEED,'candidate_configurations':len(table),'cv_fits':len(table)*5,'full_development_refits':1,
        'test_evaluated':False,'threshold':.5,'threshold_tuning':False,
        'evaluation_limit':'Selection CV scores are not unbiased independent evaluation of the tuned procedure; no nested CV or final holdout evaluation.',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}}
    (report/'search_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    matrix=np.empty((3,3)); lookup={(r['max_depth'],r['min_samples_leaf']):r for r in table}
    for i,depth in enumerate([6,10,14]):
        for j,leaf in enumerate([10,30,60]): matrix[i,j]=lookup[(depth,leaf)]['validation_roc_auc_mean']
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained'); fig.get_layout_engine().set(rect=(0,.08,1,.92))
    color_min=min(float(matrix.min()),.76); color_max=float(matrix.max())+.005
    im=ax.imshow(matrix,cmap='YlGnBu',vmin=color_min,vmax=color_max)
    ax.set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=[10,30,60],yticklabels=[6,10,14],
        xlabel='min_samples_leaf',ylabel='max_depth',title='Random forest: mean 5-fold validation ROC-AUC')
    for i,depth in enumerate([6,10,14]):
        for j,leaf in enumerate([10,30,60]):
            star='\nSelected' if depth==search.best_params_['classifier__max_depth'] and leaf==search.best_params_['classifier__min_samples_leaf'] else ''
            ax.text(j,i,f'{matrix[i,j]:.4f}'+star,ha='center',va='center',color='white' if (matrix[i,j]-color_min)/(color_max-color_min)>.55 else '#26313c')
    fig.colorbar(im,ax=ax,label='Mean validation AUC')
    fig.text(.01,.015,'9 candidates | Same development folds | Selection scores; final test untouched',fontsize=9)
    fig.savefig(figs/'gridsearch_auc.png',dpi=180,bbox_inches='tight');fig.savefig(figs/'gridsearch_auc.svg',bbox_inches='tight');plt.close(fig)
    lines=['随机森林 GridSearchCV 调参结果','',f'开发数据：{len(y):,}条；9种参数组合 × 五折分组验证＝45次拟合，选优后全部开发集重拟合1次。',
        '搜索范围：max_depth=[6,10,14]；min_samples_leaf=[10,30,60]。',
        '固定：300棵树，max_features=sqrt，class_weight=None，random_state=4145。',
        '选优指标：五折验证ROC-AUC的算术平均值；其他指标仅作辅助。',
        '最优仅指本次9种组合之中，不是所有参数空间的全局最优。',
        '选中参数：'+json.dumps(search.best_params_), '', '选中组合的五折验证均值：']
    lines += [f'{key}: {value:.6f}' for key,value in selection_metrics.items()]
    lines += ['',f'选中组合的各折训练准确率均值：{metadata["mean_training_accuracy"]:.2%}。',
        '与原基线的比较统一使用五折均值，而不是将折均值与合并OOF指标混比：']
    lines += [f'{r["metric"]}: {r["baseline_fold_mean"]:.6f} -> {r["selected_config_fold_mean"]:.6f}，差值{r["difference"]:+.6f}' for r in comparison]
    lines += ['', '重要解释',
        '此验证分数参与了参数选择，可能偏乐观，不是调参流程的独立泛化成绩；未进行嵌套交叉验证。',
        'GridSearchCV的cv_results_里test字样指交叉验证的验证折，不是最终cardio_test_20.csv。',
        '所有候选使用同一数据、特征和分组；相同资料组合不跨训练和验证折。',
        '不需要缩放树模型；沿用原单位输入与既定类别编码，处理在Pipeline中。',
        '保存模型由GridSearchCV自动在全部开发集重拟合；保存模型的训练成绩不能作为独立成绩。',
        '最终测试CSV没有读取或评估；分类门槛仍为0.5，无过采样。',
        '保留原随机森林基线，调参结果单独存于random_forest_gridsearch，方便写调参前后对比。',
        '全量清洗后按普通分层方法分出的train/test仍存在相同健康资料跨集的可能，本次未改变划分。',
        '原数据为疾病状态标签，模型评分不是临床诊断或未来发病风险。','',
        '复现：python3 src/tune_random_forest.py；依赖沿用requirements_tree_models_locked.txt。',
        '评分：python3 src/predict_logistic.py --model models/random_forest_gridsearch/model_pipeline.joblib --input INPUT.csv --output predictions.csv',
        'ChatGPT协助编写并运行搜索代码，统计结果由本地程序计算和核验。']
    text='\n'.join(lines)+'\n';(report/'report_zh.txt').write_text(text,encoding='utf-8')
    b64=base64.b64encode((figs/'gridsearch_auc.png').read_bytes()).decode()
    rows=''.join(f'<tr><td>{r["rank_auc"]}</td><td>{r["max_depth"]}</td><td>{r["min_samples_leaf"]}</td><td>{r["validation_roc_auc_mean"]:.6f}</td><td>{r["training_accuracy_mean"]:.2%}</td><td>{r["validation_accuracy_mean"]:.2%}</td></tr>' for r in table)
    (report/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>随机森林网格搜索结果</title><style>body{font-family:-apple-system,Arial,sans-serif;max-width:1100px;margin:40px auto;line-height:1.8}img{width:100%}pre{white-space:pre-wrap;font-family:inherit}table{border-collapse:collapse;width:100%}td,th{border:1px solid #d9d9d9;padding:9px}</style><h1>随机森林网格搜索结果</h1><img alt="Mean validation AUC by hyperparameters" src="data:image/png;base64,'+b64+'"><table><tr><th>排名</th><th>深度</th><th>叶节点最少样本</th><th>验证AUC均值</th><th>训练准确率均值</th><th>验证准确率均值</th></tr>'+rows+'</table><pre>'+html.escape(text)+'</pre></html>',encoding='utf-8')
    assert len(table)==9 and np.isclose(search.best_score_,max(r['validation_roc_auc_mean'] for r in table))
    assert np.isclose(selection_metrics['roc_auc'],np.mean([r['validation_roc_auc'] for r in best_folds]))
    with (report/'gridsearch_results.csv').open(encoding='utf-8-sig',newline='') as f: saved=list(csv.DictReader(f))
    assert len(saved)==9
    (report/'validation.txt').write_text('PASS: nine configurations, five grouped folds, and one full-development refit.\nPASS: all folds exactly match baseline RF; no profile group crosses folds.\nPASS: selected parameter score matches maximum candidate mean AUC.\nPASS: best-fold AUC mean and exported results reconcile.\nPASS: saved model reload reproduces predictions.\nPASS: final test CSV not opened; baseline artifacts preserved.\n',encoding='utf-8')
    print('BEST_PARAMETERS',json.dumps(search.best_params_),flush=True)
    print('SELECTED_CV_MEANS',json.dumps(selection_metrics),flush=True)
    print('TRAIN_ACCURACY_MEAN',metadata['mean_training_accuracy'],flush=True)

if __name__=='__main__': main()
