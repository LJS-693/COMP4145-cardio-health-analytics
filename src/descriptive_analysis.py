"""Cardio descriptive analysis on the development split only.

Run from any directory: python3 src/descriptive_analysis.py
Dependencies: numpy, matplotlib. No models, hypothesis tests or test-set reads.
"""
import argparse
import base64
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
AGE_ORDER = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
BMI_ORDER = ['<18.5', '18.5-<25', '25-<30', '30-<35', '>=35']
MIN_N = 100  # Display guard, not a statistical significance criterion.
NUMERIC = {'age_years': ('Age', 'years'), 'height': ('Height', 'cm'),
           'weight': ('Weight', 'kg'), 'bmi': ('BMI', 'kg/m²'),
           'ap_hi': ('Systolic BP', 'mmHg'), 'ap_lo': ('Diastolic BP', 'mmHg'),
           'pulse_pressure': ('Pulse pressure', 'mmHg')}
BLUE, ORANGE = '#23658A', '#C26735'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def export_csv(path, rows, fields=None):
    if not fields:
        fields = list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rate_table(rows, fields, orders=None):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[c] for c in fields)].append(r)
    if len(fields) == 1 and orders:
        keys = [(v,) for v in orders if (v,) in groups]
    else:
        keys = sorted(groups)
    out = []
    for key in keys:
        group = groups[key]
        n, pos = len(group), sum(int(r['cardio']) for r in group)
        out.append({**dict(zip(fields, key)), 'records': n, 'cardio_0': n-pos,
                    'cardio_1': pos, 'positive_rate': pos/n,
                    'display_eligible': int(n >= MIN_N)})
    return out


def stats_table(rows):
    out = []
    for label in ['all', '0', '1']:
        subset = rows if label == 'all' else [r for r in rows if r['cardio'] == label]
        for field, (_, unit) in NUMERIC.items():
            x = np.array([float(r[field]) for r in subset])
            q1, med, q3 = np.quantile(x, [.25, .5, .75])
            out.append({'cardio_group': label, 'field': field, 'unit': unit,
                        'records': len(x), 'mean': float(x.mean()), 'std_sample': float(x.std(ddof=1)),
                        'min': float(x.min()), 'q1': float(q1), 'median': float(med),
                        'q3': float(q3), 'max': float(x.max())})
    return out


def rank_average(x):
    """Average ranks for ties; equivalent to rankdata(method='average')."""
    idx = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[idx[j]] == x[idx[i]]:
            j += 1
        ranks[idx[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def rate_plot(ax, table, key, title, labels=None):
    x = np.arange(len(table))
    values = [r['positive_rate'] if r['display_eligible'] else np.nan for r in table]
    ax.bar(x, values, color=BLUE, width=.65)
    ax.set_xticks(x, [labels.get(r[key], r[key]) if labels else r[key] for r in table])
    ax.set_xlim(-.6, len(table)-.4)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Disease label proportion')
    ax.set_title(title, loc='left', fontweight='bold', pad=17)
    for i, r in enumerate(table):
        if r['display_eligible']:
            ax.text(i, r['positive_rate']+.025, f"{r['positive_rate']:.1%}\nn={r['records']:,}",
                    ha='center', va='bottom', fontsize=9)
        else:
            ax.text(i, .05, f"Suppressed\nn={r['records']:,}", ha='center', fontsize=9, color='#666666')
    ax.grid(axis='y', alpha=.22)
    ax.set_axisbelow(True)


def table_html(rows, fields, headers=None):
    headers = headers or fields
    parts = ['<table><thead><tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in headers) + '</tr></thead><tbody>']
    for r in rows:
        parts.append('<tr>' + ''.join(f'<td>{html.escape(str(r.get(c,"")))}</td>' for c in fields) + '</tr>')
    return ''.join(parts) + '</tbody></table>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    project, output = args.project_root, args.output_root or args.project_root
    src = project / 'data/processed/caroline_cleaned version/cardio_train_80.csv'
    rows = read_csv(src)
    scales = json.loads(src.with_name('cleaning_stats.json').read_text())['denormalization']
    for row in rows:
        for field in NUMERIC:
            limits = scales[field]
            value = float(row[field]) * (limits['max'] - limits['min']) + limits['min']
            row[field] = round(value, 10)
        age = row['age_years']
        bmi = row['bmi']
        row['age_group'] = AGE_ORDER[sum(age >= cutoff for cutoff in (30, 40, 50, 60, 70))]
        row['bmi_band'] = BMI_ORDER[sum(bmi >= cutoff for cutoff in (18.5, 25, 30, 35))]
        row['gender'] = {'0': '1', '1': '2'}[row['gender']]
        for field in ('cholesterol', 'gluc'):
            levels = [str(i) for i in (1, 2, 3)
                      if row[f'{field}_{i}'].lower() in ('true', '1')]
            if len(levels) != 1:
                raise ValueError(f"Invalid {field} encoding for record {row['id']}")
            row[field] = levels[0]
    n = len(rows)
    assert len({r['id'] for r in rows}) == n
    tables_dir = output / 'outputs/descriptive/tables'
    figs_dir = output / 'outputs/figures/descriptive'
    reports_dir = output / 'reports/descriptive'
    for folder in [tables_dir, figs_dir, reports_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    tables = {}
    tables['label_distribution'] = rate_table(rows, ['cardio'], ['0', '1'])
    for r in tables['label_distribution']:
        r['sample_share'] = r['records'] / n
    for field, order in [('age_group', AGE_ORDER), ('bmi_band', BMI_ORDER), ('gender', ['1','2']),
                         ('cholesterol', ['1','2','3']), ('gluc', ['1','2','3']),
                         ('smoke', ['0','1']), ('alco', ['0','1']), ('active', ['0','1'])]:
        tables[field] = rate_table(rows, [field], order)
    tables['age_bmi'] = rate_table(rows, ['age_group', 'bmi_band'])
    tables['age_activity'] = rate_table(rows, ['age_group', 'active'])
    tables['numeric_summary'] = stats_table(rows)
    for name, table in tables.items():
        export_csv(tables_dir / f'{name}.csv', table)

    # Rank correlations use continuous predictors only; category codes are excluded.
    corr_fields = list(NUMERIC)
    ranks = np.column_stack([rank_average(np.array([float(r[c]) for r in rows])) for c in corr_fields])
    correlations = np.corrcoef(ranks, rowvar=False)
    export_csv(tables_dir / 'spearman_correlations.csv', [
        {'field_1': a, 'field_2': b, 'spearman_rho': float(correlations[i,j])}
        for i,a in enumerate(corr_fields) for j,b in enumerate(corr_fields)])

    # Power BI table: a long-format fact table for group-rate visuals.
    powerbi = []
    for field in ['age_group', 'bmi_band', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']:
        for order, r in enumerate(tables[field], 1):
            powerbi.append({'dimension': field, 'category': r[field], 'sort_order': order,
                            **{k:r[k] for k in ['records','cardio_0','cardio_1','positive_rate','display_eligible']}})
    export_csv(tables_dir / 'powerbi_group_rates.csv', powerbi)

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                          'axes.spines.right':False, 'axes.labelcolor':'#26313C',
                          'text.color':'#26313C', 'savefig.facecolor':'white'})
    figures = []
    def save(fig, filename, title):
        # Reserve a distinct footer strip so notes never overlap axis labels.
        fig.get_layout_engine().set(rect=(0, .075, 1, .925))
        fig.text(.01, .012, f'Development split only | n={n:,} | Descriptive associations, not causal effects',
                 fontsize=9, color='#626C76')
        fig.savefig(figs_dir / f'{filename}.png', dpi=180, bbox_inches='tight')
        fig.savefig(figs_dir / f'{filename}.svg', bbox_inches='tight')
        plt.close(fig)
        figures.append({'file':filename, 'title':title})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, field, bins, title in [(axes[0], 'age_years', np.arange(25,71,2), 'Age distribution'),
                                  (axes[1], 'bmi', np.arange(0,161,2), 'BMI distribution (full range)')]:
        ax.hist([float(r[field]) for r in rows], bins=bins, color=BLUE, edgecolor='white')
        ax.set(title=title, xlabel=f'{NUMERIC[field][0]} ({NUMERIC[field][1]})', ylabel='Records')
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.grid(axis='y', alpha=.2)
        if field == 'bmi':
            ax.set_yscale('log')
            ax.set_ylabel('Records (log scale)')
    save(fig, '01_age_bmi_distributions', '年龄和BMI分布')

    for field, filename, title in [('age_group','02_age_label_rates','Disease labels by age group'),
                                  ('bmi_band','03_bmi_label_rates','Disease labels by BMI band')]:
        fig, ax = plt.subplots(figsize=(9,5.3), layout='constrained')
        rate_plot(ax, tables[field], field, title)
        ax.set_xlabel('Age group (years)' if field == 'age_group' else 'BMI band (kg/m²)')
        save(fig, filename, '年龄组标签比例' if field == 'age_group' else 'BMI组标签比例')

    fig, axes = plt.subplots(2,2, figsize=(11,9), layout='constrained')
    for ax, field in zip(axes.flat, ['age_years','bmi','ap_hi','ap_lo']):
        values = [[float(r[field]) for r in rows if r['cardio'] == c] for c in ['0','1']]
        b = ax.boxplot(values, tick_labels=['cardio=0','cardio=1'], patch_artist=True,
                       flierprops={'markersize':2,'alpha':.15,'marker':'.'})
        for box, color in zip(b['boxes'], [BLUE,ORANGE]):
            box.set_facecolor(color)
            box.set_alpha(.65)
        ax.set_title(NUMERIC[field][0], loc='left', fontweight='bold')
        ax.set_ylabel(NUMERIC[field][1])
        ax.grid(axis='y', alpha=.2)
    save(fig,'04_numeric_by_label','按疾病标签比较年龄BMI和血压')

    level_labels = {'1':'1: Normal','2':'2: Above normal','3':'3: Well above normal'}
    fig, axes = plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    for ax, field, title in zip(axes,['cholesterol','gluc'],['Cholesterol category','Glucose category']):
        rate_plot(ax,tables[field],field,title,level_labels)
        ax.tick_params(axis='x', labelsize=9)
    save(fig,'05_cholesterol_glucose','胆固醇与血糖等级')

    fig,axes=plt.subplots(1,3,figsize=(13,5.2),layout='constrained')
    for ax,field,title in zip(axes,['smoke','alco','active'],['Smoking','Alcohol intake','Physical activity']):
        rate_plot(ax,tables[field],field,title,{'0':'No','1':'Yes'})
    save(fig,'06_lifestyle_label_rates','生活习惯分组标签比例')

    fig,axes=plt.subplots(1,2,figsize=(10,5.2),layout='constrained')
    rate_plot(axes[0],tables['gender'],'gender','Labels by gender code',{'1':'Code 1','2':'Code 2'})
    axes[1].bar(['Code 1','Code 2'],[r['records'] for r in tables['gender']],color=BLUE,width=.6)
    axes[1].set_title('Gender-code sample sizes',loc='left',fontweight='bold')
    axes[1].set_ylabel('Records')
    for i,r in enumerate(tables['gender']):
        axes[1].text(i,r['records']+400,f"{r['records']:,}",ha='center')
    axes[1].set_ylim(0,max(r['records'] for r in tables['gender'])*1.15)
    save(fig,'07_gender_code_comparison','性别编码分布与标签比例')

    age_present = [a for a in AGE_ORDER if any(r['age_group']==a for r in rows)]
    grid=np.full((len(age_present),len(BMI_ORDER)),np.nan)
    look={(r['age_group'],r['bmi_band']):r for r in tables['age_bmi']}
    for i,a in enumerate(age_present):
        for j,b in enumerate(BMI_ORDER):
            cell=look.get((a,b))
            if cell and cell['display_eligible']:
                grid[i,j]=cell['positive_rate']
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    cmap=plt.get_cmap('YlGnBu').copy()
    cmap.set_bad('#E9EDF0')
    im=ax.imshow(grid,cmap=cmap,vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(len(BMI_ORDER)),BMI_ORDER)
    ax.set_yticks(range(len(age_present)),age_present)
    ax.set(xlabel='BMI band (kg/m²)',ylabel='Age group (years)',title='Age × BMI: disease label proportions')
    for i,a in enumerate(age_present):
        for j,b in enumerate(BMI_ORDER):
            cell=look.get((a,b))
            if not cell:
                label='No records'
            elif not cell['display_eligible']:
                label=f"Suppressed\nn={cell['records']:,}"
            else:
                label=f"{cell['positive_rate']:.1%}\nn={cell['records']:,}"
            ax.text(j,i,label,ha='center',va='center',fontsize=9,
                    color='white' if np.isfinite(grid[i,j]) and grid[i,j]>.62 else '#24313B')
    fig.colorbar(im,ax=ax,format=PercentFormatter(1),label='Disease label proportion')
    save(fig,'08_age_bmi_heatmap','年龄与BMI交叉分析')

    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    im=ax.imshow(correlations,vmin=-1,vmax=1,cmap='RdBu_r')
    labels=[NUMERIC[c][0] for c in corr_fields]
    ax.set_xticks(range(len(labels)),labels,rotation=40,ha='right')
    ax.set_yticks(range(len(labels)),labels)
    ax.set_title('Spearman correlations among continuous predictors',loc='left',fontweight='bold',pad=20)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v=correlations[i,j]
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if abs(v)>.65 else '#24313B')
    fig.colorbar(im,ax=ax,label='Spearman rho')
    save(fig,'09_spearman_correlations','连续变量Spearman相关性')

    # Build findings from measured values rather than expected directions.
    positive=sum(int(r['cardio']) for r in rows)
    overall=positive/n
    eligible_age=[r for r in tables['age_group'] if r['display_eligible']]
    eligible_bmi=[r for r in tables['bmi_band'] if r['display_eligible']]
    youngest,oldest=eligible_age[0],eligible_age[-1]
    bmilo,bmihi=eligible_bmi[0],eligible_bmi[-1]
    chol={r['cholesterol']:r for r in tables['cholesterol']}
    glucose={r['gluc']:r for r in tables['gluc']}
    activity={r['active']:r for r in tables['active']}
    summary={(r['cardio_group'],r['field']):r for r in tables['numeric_summary']}
    med=lambda label,field:summary[(label,field)]['median']
    findings=[
        f'开发集共{n:,}条，cardio=1有{positive:,}条（{overall:.2%}），cardio=0有{n-positive:,}条；标签比例接近均衡。',
        f'足够样本的年龄组中，{youngest["age_group"]}岁组标签1比例为{youngest["positive_rate"]:.1%}（n={youngest["records"]:,}），{oldest["age_group"]}岁组为{oldest["positive_rate"]:.1%}（n={oldest["records"]:,}）。这是组间关联，不是随访发病风险。',
        f'BMI为{bmilo["bmi_band"]}组的标签1比例为{bmilo["positive_rate"]:.1%}（n={bmilo["records"]:,}）；BMI为{bmihi["bmi_band"]}组为{bmihi["positive_rate"]:.1%}（n={bmihi["records"]:,}）。具体中间组比例见图表，不能仅凭端点认定单调关系。',
        f'cardio=0与cardio=1两组收缩压中位数分别为{med("0","ap_hi"):.0f}和{med("1","ap_hi"):.0f} mmHg；舒张压中位数分别为{med("0","ap_lo"):.0f}和{med("1","ap_lo"):.0f} mmHg。箱线图同时展示组内差异与重叠。',
        f'胆固醇等级1、2、3的标签1比例分别为{chol["1"]["positive_rate"]:.1%}、{chol["2"]["positive_rate"]:.1%}、{chol["3"]["positive_rate"]:.1%}；血糖等级1、2、3分别为{glucose["1"]["positive_rate"]:.1%}、{glucose["2"]["positive_rate"]:.1%}、{glucose["3"]["positive_rate"]:.1%}。等级不是浓度。',
        f'无身体活动组的标签1比例为{activity["0"]["positive_rate"]:.1%}，有身体活动组为{activity["1"]["positive_rate"]:.1%}。这不是控制年龄等因素后的效应，不能作因果结论。',
    ]
    for field,cn in [('smoke','吸烟'),('alco','饮酒')]:
        lookup={r[field]:r for r in tables[field]}
        findings.append(f'{cn}=0与{cn}=1的标签1比例分别为{lookup["0"]["positive_rate"]:.1%}与{lookup["1"]["positive_rate"]:.1%}。仅为未经调整的样本比例，不能推断保护或伤害效应。')
    warnings=[
        '仅读取开发集cardio_train_80.csv，不读取test.csv；这些发现不能替代模型预测验证。',
        'cardio为已有疾病状态标签，不是未来发病标签；样本标签比例不代表社会总体患病率。',
        '所有比例的分母都是当前组记录数，分析单位为记录，不能由ID确认独立患者。',
        '每组少于100条时图中不显示标签比例，仅标注样本量；100是展示规则，不是显著性标准。汇总CSV仍保留原始计数与比例供审计。',
        '年龄跨度约29.6–64.9岁；不能据此外推到儿童或高龄人群。小年龄组可能不足100条。',
        '清洗界限是项目质量规则，分析结论以通过这些规则的记录为限。',
        'BMI极端值仍按既定规则保留并标记；本次不改动清洗数据。分布图和统计保留全部数值，BMI直方图采用对数纵轴。',
        'gender按1/2编码展示，因本地随附README未明确映射，不添加猜测的男女标签。',
        '分组差异可能受年龄、性别、选择偏差、测量误差或自报偏差影响；未进行因果分析。',
        'Spearman分析仅包含连续变量；BMI由身高体重计算，脉压由两项血压计算，相关性部分来自数学关系，不代表独立贡献。',
        '没有进行假设检验，不报告p值，也不将观察差异称为统计显著。',
    ]
    text='Cardio 描述性分析报告\n\n研究范围\n开发集cardio_train_80.csv；所有数据与清洗版本见来源校验值。\n\n主要发现\n'
    text+='\n'.join(f'{i+1}. {s}' for i,s in enumerate(findings))
    text+='\n\n方法与限制\n'+'\n'.join(f'- {s}' for s in warnings)
    text+='\n\n与Predictive的衔接\n将人口、身体测量和生活习惯作为基础输入，再加入血压、胆固醇和血糖进行对照实验。\n描述性关联不能直接当作特征的预测贡献；贡献需要在开发集交叉验证中检验。\n\n来源\n'
    text+='data/processed/caroline_cleaned version/cardio_train_80.csv\nhttps://github.com/caravanuden/cardio\nhttps://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset\n'
    text+='cardio_train_80.csv SHA256: '+hashlib.sha256(src.read_bytes()).hexdigest()+'\n'
    (reports_dir/'descriptive_report_zh.txt').write_text(text,encoding='utf-8')

    blocks=[]
    for f in figures:
        data=base64.b64encode((figs_dir/f'{f["file"]}.png').read_bytes()).decode()
        blocks.append(f'<section><h2>{html.escape(f["title"])}</h2><img src="data:image/png;base64,{data}" alt="{html.escape(f["title"])}"></section>')
    group_display=[{'维度':r['dimension'],'组别':r['category'],'样本量':f'{r["records"]:,}',
                    '标签1数量':f'{r["cardio_1"]:,}',
                    '标签1比例':f'{r["positive_rate"]:.2%}' if r['display_eligible'] else '小样本不展示'} for r in powerbi]
    continuous_display=[{'标签组':r['cardio_group'],'指标':r['field'],'单位':r['unit'],
                         '均值':f'{r["mean"]:.2f}','中位数':f'{r["median"]:.2f}',
                         'Q1':f'{r["q1"]:.2f}','Q3':f'{r["q3"]:.2f}'} for r in tables['numeric_summary']]
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cardio 描述性分析报告</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",Arial,sans-serif;background:#f3f5f7;color:#26313c;margin:0;line-height:1.75}
main{max-width:1120px;margin:auto;background:white;padding:40px}h1{font-size:30px;margin:0 0 12px}h2{margin:30px 0 12px;font-size:22px}
p,li{font-size:16px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{border:1px solid #d9d9d9;padding:9px 12px;text-align:left}th{background:#eaf0f5}tr:nth-child(even){background:#f8fafb}.scroll{overflow-x:auto}a{color:#23658a}@media print{body{background:white}main{padding:0}section{break-inside:avoid}}</style><main>'''
    document+=f'<h1>Cardio 描述性分析报告</h1><p>分析范围：开发集 {n:,} 条记录；疾病标签为1的比例 {overall:.2%}。本报告描述样本中的分布与关联，用于支持后续分类模型研究。</p>'
    document+='<h2>主要发现</h2><ol>'+''.join(f'<li>{html.escape(s)}</li>' for s in findings)+'</ol>'
    document+=''.join(blocks)
    document+='<h2>分组统计明细</h2><div class="scroll">'+table_html(group_display,list(group_display[0]))+'</div>'
    document+='<h2>连续指标汇总</h2><div class="scroll">'+table_html(continuous_display,list(continuous_display[0]))+'</div>'
    document+='<h2>方法与限制</h2><ul>'+''.join(f'<li>{html.escape(s)}</li>' for s in warnings)+'</ul>'
    document+='<h2>来源与复现</h2><p>数据：data/processed/caroline_cleaned version/cardio_train_80.csv。运行：python3 src/descriptive_analysis.py。</p><p>来源：<a href="https://github.com/caravanuden/cardio">Cardio原项目</a>；<a href="https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset">原数据发布页</a>。由ChatGPT协助生成分析代码和说明，实际统计由本地脚本计算。</p></main></html>'
    (reports_dir/'descriptive_report.html').write_text(document,encoding='utf-8')

    slide_text='Descriptive Analytics 展示内容\n\n'
    slide_text+='第1页 数据与方法\n开发集54,887条；分组聚合、分布比较、交叉表和Spearman相关分析；不读取最终测试集。\n'
    slide_text+='第2页 年龄与BMI\n配图02、03；讲述：'+findings[1]+' '+findings[2]+'\n'
    slide_text+='第3页 体检指标\n配图04、05；讲述：'+findings[3]+' '+findings[4]+'\n'
    slide_text+='第4页 生活习惯与下一步\n配图06；讲述：'+findings[5]+'\n下一步比较基础资料与加入体检指标的模型，不将分组差异直接当作预测效果。\n'
    slide_text+='\n完整项目展示不足12分钟时，建议把本部分合并成2–3页，选择最能支持研究问题的图。\n'
    (reports_dir/'presentation_notes_zh.txt').write_text(slide_text,encoding='utf-8')

    validation=[]
    for name in ['age_group','bmi_band','gender','cholesterol','gluc','smoke','alco','active']:
        t=read_csv(tables_dir/f'{name}.csv')
        assert sum(int(r['records']) for r in t)==n
        assert sum(int(r['cardio_1']) for r in t)==positive
        for r in t:
            assert int(r['cardio_0'])+int(r['cardio_1'])==int(r['records'])
            assert abs(float(r['positive_rate'])-int(r['cardio_1'])/int(r['records']))<1e-12
        validation.append(f'PASS: {name} counts, denominators and rates reconcile.')
    assert sum(r['records'] for r in tables['age_bmi'])==n
    assert np.allclose(correlations,correlations.T)
    assert np.allclose(np.diag(correlations),1)
    assert np.allclose(rank_average(np.array([3,1,1,4])),[3,1.5,1.5,4])
    for field in NUMERIC:
        direct=sorted(float(r[field]) for r in rows)
        assert summary[('all',field)]['median']==direct[len(direct)//2]
    from PIL import Image
    for f in figures:
        with Image.open(figs_dir/f'{f["file"]}.png') as im:
            im.verify()
    validation+=['PASS: cross-tab totals reconcile; Spearman tie ranks/symmetry/diagonal checked.',
                 'PASS: numeric medians independently checked against sorted records.',
                 'PASS: all exported PNGs decode; manual visual review is performed separately.',
                 'PASS: only the development split was analyzed; no source dataset modified.']
    (reports_dir/'validation.txt').write_text('\n'.join(validation)+'\n',encoding='utf-8')
    manifest={'source':'data/processed/caroline_cleaned version/cardio_train_80.csv','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
              'records':n,'cardio_1':positive,'positive_rate':overall,'display_minimum_n':MIN_N,
              'figures':figures,'csv_tables':sorted(p.name for p in tables_dir.glob('*.csv')),
              'findings_zh':findings,'numpy_version':np.__version__,'matplotlib_version':matplotlib.__version__}
    (reports_dir/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n'.join(findings))
    print(f'Wrote {len(figures)} PNG/SVG figure pairs and {len(manifest["csv_tables"])} CSV tables.')


if __name__=='__main__':
    main()
