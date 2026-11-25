"""RF round 2: hold round-1 depth/leaf fixed and search feature/row sampling.
Run: python3 src/tune_random_forest_round2.py
"""
import argparse,csv,json,hashlib,base64,html,platform
from pathlib import Path
import numpy as np
import joblib,sklearn
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_tree_models import build_pipeline,MODEL_CONFIGS
from train_logistic import write_csv,SEED
from rf_tuning_data import load_development

ROOT=Path(__file__).resolve().parents[1]
FEATURE_OPTIONS=['sqrt',.7,1.0]
SAMPLE_OPTIONS=[.6,.8,None]
GRID={'classifier__max_features':FEATURE_OPTIONS,'classifier__max_samples':SAMPLE_OPTIONS}
METRICS=['roc_auc','accuracy','precision','recall','f1']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--output-root',type=Path)
    args=p.parse_args();project=args.project_root;out=args.output_root or project
    X,y,features,records,params,cv,sha,baseline=load_development(project)
    first_path=project/'reports/random_forest_gridsearch/search_metadata.json'
    first=json.loads(first_path.read_text())
    assert first['training_sha256']==sha and first['features']==features
    depth=first['best_params']['classifier__max_depth']
    leaf=first['best_params']['classifier__min_samples_leaf']
    assert (depth,leaf)==(14,10)
    model=build_pipeline('random_forest',features).set_params(
        classifier__max_depth=depth,classifier__min_samples_leaf=leaf,classifier__bootstrap=True)
    scoring={m:m for m in METRICS}
    search=GridSearchCV(model,GRID,scoring=scoring,refit='roc_auc',cv=cv,n_jobs=1,
        return_train_score=True,error_score='raise',verbose=3)
    print(f'Fixed depth={depth}, leaf={leaf}, 300 trees; 9 candidates × 5 grouped folds.',flush=True)
    search.fit(X,y);r=search.cv_results_;best=search.best_index_
    report=out/'reports/random_forest_gridsearch_round2'
    figs=out/'outputs/figures/random_forest_gridsearch_round2'
    models=out/'models/random_forest_gridsearch_round2'
    for d in [report,figs,models]:d.mkdir(parents=True,exist_ok=True)
    table=[];all_folds=[]
    for i,param in enumerate(r['params']):
        row={'rank_auc':int(r['rank_test_roc_auc'][i]),
             'max_features':str(param['classifier__max_features']),
             'max_samples':str(param['classifier__max_samples']),
             'mean_fit_seconds':float(r['mean_fit_time'][i])}
        for m in METRICS:
            for phase in ['train','test']:
                label='training' if phase=='train' else 'validation'
                row[f'{label}_{m}_mean']=float(r[f'mean_{phase}_{m}'][i])
                row[f'{label}_{m}_std_population']=float(r[f'std_{phase}_{m}'][i])
        table.append(row)
        for fold in range(5):
            f={'candidate_index':i,'max_features':row['max_features'],
               'max_samples':row['max_samples'],'fold':fold+1}
            for m in METRICS:
                f[f'validation_{m}']=float(r[f'split{fold}_test_{m}'][i])
                f[f'training_{m}']=float(r[f'split{fold}_train_{m}'][i])
            all_folds.append(f)
    table.sort(key=lambda x:x['rank_auc'])
    write_csv(report/'gridsearch_results.csv',table)
    write_csv(report/'all_cv_fold_metrics.csv',all_folds)
    chosen=r['params'][best]
    selected_folds=[f for f in all_folds if f['candidate_index']==best]
    write_csv(report/'best_cv_fold_metrics.csv',selected_folds)
    reference=next(row for row in table if row['max_features']=='sqrt' and row['max_samples']=='None')
    for m in METRICS:
        assert np.isclose(reference[f'validation_{m}_mean'],first['selected_cv_fold_means'][m],atol=1e-10,rtol=0)
    with (project/'reports/random_forest/cv_fold_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        original=list(csv.DictReader(f))
    comparisons=[]
    for m in METRICS:
        comparisons.append({'metric':m,'original_baseline_fold_mean':float(np.mean([float(f[m]) for f in original])),
            'round1_selected_fold_mean':first['selected_cv_fold_means'][m],
            'round2_selected_fold_mean':float(r[f'mean_test_{m}'][best]),
            'round2_minus_round1':float(r[f'mean_test_{m}'][best])-first['selected_cv_fold_means'][m]})
    write_csv(report/'round_comparison.csv',comparisons)
    selected_metrics={m:float(r[f'mean_test_{m}'][best]) for m in METRICS}
    final_params={**MODEL_CONFIGS['random_forest'],'max_depth':depth,'min_samples_leaf':leaf,'bootstrap':True,
        'max_features':chosen['classifier__max_features'],'max_samples':chosen['classifier__max_samples']}
    path=models/'model_pipeline.joblib'
    joblib.dump({'pipeline':search.best_estimator_,'features':features,'threshold':.5,'target':'cardio',
        'denormalization_params':params,'input_units':'original units; original categorical codes',
        'selected_parameters':final_params},path,compress=('xz',3))
    assert np.allclose(joblib.load(path)['pipeline'].predict_proba(X[:100]),search.best_estimator_.predict_proba(X[:100]))
    metadata={'search_round':2,'records':len(y),'training_sha256':sha,
        'training_source':first['training_source'],'features':features,
        'round1_metadata_sha256':hashlib.sha256(first_path.read_bytes()).hexdigest(),
        'parameter_grid':GRID,'fixed_depth':depth,'fixed_min_samples_leaf':leaf,
        'best_params':chosen,'selected_full_parameters':final_params,
        'selection_metric':'mean five-fold validation ROC-AUC','best_mean_cv_auc':float(search.best_score_),
        'selected_cv_fold_means':selected_metrics,'mean_training_accuracy':float(r['mean_train_accuracy'][best]),
        'cv':'same five StratifiedGroupKFold partitions as original baseline and round 1',
        'candidate_configurations':9,'cv_fits':45,'full_development_refits':1,
        'test_evaluated':False,'threshold':.5,'threshold_tuning':False,
        'training_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data_loader_code_sha256':hashlib.sha256((Path(__file__).parent/'rf_tuning_data.py').read_bytes()).hexdigest(),
        'evaluation_limit':'Sequential selection on the same development folds; not independent performance estimation.',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}}
    (report/'search_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    lookup={(row['max_features'],row['max_samples']):row for row in table}
    matrix=np.array([[lookup[(str(feature),str(sample))]['validation_roc_auc_mean']
                      for sample in SAMPLE_OPTIONS] for feature in FEATURE_OPTIONS])
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained');fig.get_layout_engine().set(rect=(0,.08,1,.92))
    vmin=min(float(matrix.min()),.78);vmax=float(matrix.max())+.004
    image=ax.imshow(matrix,cmap='YlGnBu',vmin=vmin,vmax=vmax)
    ax.set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=['0.6','0.8','None (full draw)'],
        yticklabels=['sqrt','0.7','1.0'],xlabel='max_samples',ylabel='max_features',
        title='RF round 2: mean five-fold validation AUC')
    for i,feature in enumerate(FEATURE_OPTIONS):
        for j,sample in enumerate(SAMPLE_OPTIONS):
            selected=feature==chosen['classifier__max_features'] and sample==chosen['classifier__max_samples']
            ax.text(j,i,f'{matrix[i,j]:.5f}'+('\nSelected' if selected else ''),ha='center',va='center',
                color='white' if (matrix[i,j]-vmin)/(vmax-vmin)>.55 else '#26313c')
    fig.colorbar(image,ax=ax,label='Mean validation AUC')
    fig.text(.01,.015,f'Fixed depth={depth}, leaf={leaf}, 300 trees | Selection CV; final test untouched',fontsize=9)
    fig.savefig(figs/'gridsearch_auc.png',dpi=180,bbox_inches='tight');fig.savefig(figs/'gridsearch_auc.svg',bbox_inches='tight');plt.close(fig)
    lines=['随机森林第二轮 GridSearchCV','',f'开发集{len(y):,}条，固定第一轮max_depth={depth}、min_samples_leaf={leaf}，300棵树、bootstrap=True。',
        '搜索max_features=[sqrt,0.7,1.0]和max_samples=[0.6,0.8,None]，9种组合×五折＝45次拟合，选中后全开发集重拟合1次。',
        'max_features作用于每次分支；max_samples是每棵树有放回抽样次数的比例；None表示抽样次数等于训练折记录数，不代表每条记录都使用一次。',
        '同样的数据、输入特征和五折资料分组，以平均验证ROC-AUC选择。',
        '本轮选中参数：'+json.dumps(chosen,ensure_ascii=False),'','选中组合五折验证均值：']
    lines += [f'{m}: {selected_metrics[m]:.6f}' for m in METRICS]
    lines += ['',f'选中组合各折训练准确率均值：{metadata["mean_training_accuracy"]:.2%}。','与第一轮比较，统一使用五折均值：']
    lines += [f'{row["metric"]}: {row["round1_selected_fold_mean"]:.6f} -> {row["round2_selected_fold_mean"]:.6f}，差值{row["round2_minus_round1"]:+.6f}' for row in comparisons]
    lines += ['', '解释与限制',
        'sqrt/None组合包含在搜索中，已复现第一轮所有验证指标，保证对照一致。',
        '第一、二轮都使用同一开发验证折做选参，结果可能偏乐观，不是独立泛化评估；未做嵌套交叉验证。',
        '本轮最佳仅指固定深度和叶节点设置下的9种组合，不是所有参数的全局最优。',
        '未读取最终测试CSV；未做类别重采样、无门槛调整，门槛仍为0.5。',
        '树模型使用恢复后的原单位数据，连续值核对原始ID，类别编码在Pipeline内处理，无需缩放。',
        '保存的模型在全部开发集重拟合，不以其训练分数作为独立成绩。基线、第一轮和其他模型均保留。',
        'GridSearchCV中test指标指交叉验证的验证折，不是最终测试集；std为折间总体标准差，不是置信区间。',
        '外部train/test划分仍可能有相同健康资料跨集，本次未改动该划分。',
        '数据是现有疾病状态标签；评分不是临床诊断或未来风险。','',
        '复现：python3 src/tune_random_forest_round2.py，依赖沿用requirements_tree_models_locked.txt。',
        '评分：python3 src/predict_logistic.py --model models/random_forest_gridsearch_round2/model_pipeline.joblib --input INPUT.csv --output predictions.csv',
        'ChatGPT协助编写和运行代码，全部统计结果由本地计算并核验。']
    text='\n'.join(lines)+'\n';(report/'report_zh.txt').write_text(text,encoding='utf-8')
    b64=base64.b64encode((figs/'gridsearch_auc.png').read_bytes()).decode()
    rows=''.join(f'<tr><td>{row["rank_auc"]}</td><td>{row["max_features"]}</td><td>{row["max_samples"]}</td><td>{row["validation_roc_auc_mean"]:.6f}</td><td>{row["validation_accuracy_mean"]:.2%}</td></tr>' for row in table)
    (report/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>随机森林第二轮网格搜索</title><style>body{font-family:-apple-system,Arial,sans-serif;max-width:1100px;margin:40px auto;line-height:1.8}img{width:100%}pre{white-space:pre-wrap;font-family:inherit}table{border-collapse:collapse;width:100%}td,th{border:1px solid #d9d9d9;padding:9px}</style><h1>随机森林第二轮网格搜索</h1><img alt="Sampling hyperparameter search" src="data:image/png;base64,'+b64+'"><table><tr><th>排名</th><th>max_features</th><th>max_samples</th><th>验证AUC均值</th><th>验证准确率均值</th></tr>'+rows+'</table><pre>'+html.escape(text)+'</pre></html>',encoding='utf-8')
    assert len(table)==9 and len(all_folds)==45 and np.isclose(search.best_score_,max(row['validation_roc_auc_mean'] for row in table))
    assert np.isclose(np.mean([f['validation_roc_auc'] for f in selected_folds]),search.best_score_)
    with (report/'gridsearch_results.csv').open(encoding='utf-8-sig',newline='') as f: assert len(list(csv.DictReader(f)))==9
    (report/'validation.txt').write_text('PASS: original baseline source, features, and exact grouped fold assignments retained.\nPASS: fixed depth=14, leaf=10, bootstrap=True, and 300 trees.\nPASS: sqrt/None candidate reproduces all round-1 CV metrics.\nPASS: nine candidates and 45 exported fold results; best score matches highest mean AUC.\nPASS: saved model reload reproduces predictions.\nPASS: test CSV not read; no previous outputs overwritten.\n',encoding='utf-8')
    print('BEST_PARAMETERS',json.dumps(chosen),flush=True)
    print('SELECTED_CV_MEANS',json.dumps(selected_metrics),flush=True)
    print('TRAIN_ACCURACY',metadata['mean_training_accuracy'],flush=True)

if __name__=='__main__':main()
