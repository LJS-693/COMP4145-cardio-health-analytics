"""Reproducible Cardio QC and fixed, target-independent group split. Standard library only."""
import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
          'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
PREDICTORS = [c for c in FIELDS if c not in ('id', 'cardio')]
INTEGER_FIELDS = [c for c in FIELDS if c != 'weight']
DERIVED = ['age_years', 'bmi', 'pulse_pressure', 'age_group', 'bmi_band']
META = ['source_row', 'profile_group', 'split', 'quality_flags']


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def profile(rows):
    result = {'rows': len(rows), 'label_counts': dict(Counter(str(r['cardio']) for r in rows))}
    labels = [str(r['cardio']) for r in rows if str(r['cardio']) in ('0', '1')]
    result['valid_label_records'] = len(labels)
    result['positive_fraction'] = labels.count('1') / len(labels) if labels else None
    result['numeric_summary'] = {}
    for field in ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'age_years', 'bmi']:
        values = []
        for r in rows:
            try:
                value = float(r[field])
                if math.isfinite(value):
                    values.append(value)
            except (KeyError, ValueError, TypeError):
                continue
        values.sort()
        if values:
            q = statistics.quantiles(values, n=4, method='inclusive') if len(values) > 1 else values * 3
            result['numeric_summary'][field] = {
                'min': min(values), 'q1': q[0], 'median': statistics.median(values),
                'q3': q[2], 'max': max(values), 'mean': statistics.mean(values)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/cardio_train.csv')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/cleaning_rules.json')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.output_root
    processed, reports = root / 'data/processed', root / 'reports'
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    policy = json.loads(args.config.read_text(encoding='utf-8'))
    with args.input.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        if reader.fieldnames != FIELDS:
            raise ValueError(f'Unexpected source schema: {reader.fieldnames}')
        source = list(reader)
    if not source:
        raise ValueError('Source is empty')
    ids = Counter(r['id'] for r in source)
    missing = {c: sum(not r[c].strip() for r in source) for c in FIELDS}
    clean, excluded, annotated, all_reasons, first_reasons = [], [], [], Counter(), Counter()
    signature_ids, signature_labels = defaultdict(list), defaultdict(set)
    full_rows = Counter(tuple(r[c] for c in FIELDS) for r in source)
    raw_profile_count = Counter(tuple(r[c] for c in FIELDS if c != 'id') for r in source)

    for line, original in enumerate(source, start=2):
        values, reasons, flags = {}, [], []
        for c in FIELDS:
            try:
                value = float(original[c])
                if not math.isfinite(value):
                    raise ValueError('nonfinite')
                if c in INTEGER_FIELDS:
                    if not value.is_integer():
                        raise ValueError('noninteger')
                    value = int(value)
                values[c] = value
            except (ValueError, TypeError):
                reasons.append(f'missing_or_invalid_numeric:{c}')
        if 'id' in values and (values['id'] < 0 or ids[original['id']] > 1):
            reasons.append('invalid_or_duplicate_id')
        for c, allowed in policy['allowed_categories'].items():
            if c in values and values[c] not in allowed:
                reasons.append(f'invalid_category:{c}')
        for c, (lo, hi) in policy['bounds'].items():
            if c in values and not lo <= values[c] <= hi:
                reasons.append(f'outside_qc_bounds:{c}')
        if (policy['require_systolic_above_diastolic'] and 'ap_hi' in values
                and 'ap_lo' in values and values['ap_hi'] <= values['ap_lo']):
            reasons.append('systolic_not_above_diastolic')

        derived = dict.fromkeys(DERIVED, '')
        if 'age' in values:
            age = values['age'] / 365.25
            derived['age_years'] = round(age, 6)
            derived['age_group'] = ('<30' if age < 30 else '30-39' if age < 40 else
                                    '40-49' if age < 50 else '50-59' if age < 60 else
                                    '60-69' if age < 70 else '70+')
        if values.get('height', 0) > 0 and values.get('weight', 0) > 0:
            bmi = values['weight'] / (values['height'] / 100) ** 2
            derived['bmi'] = round(bmi, 6)
            derived['bmi_band'] = ('<18.5' if bmi < 18.5 else '18.5-<25' if bmi < 25 else
                                   '25-<30' if bmi < 30 else '30-<35' if bmi < 35 else '>=35')
            lo, hi = policy['bmi_review_bounds']
            if not lo <= bmi <= hi:
                flags.append('bmi_outside_review_bounds')
        if 'ap_hi' in values and 'ap_lo' in values:
            derived['pulse_pressure'] = values['ap_hi'] - values['ap_lo']

        group, split = '', ''
        if not reasons:
            # Group by original numeric predictors, excluding ID and disease label.
            signature = json.dumps([float(values[c]) for c in PREDICTORS], separators=(',', ':'))
            group = hashlib.sha256(signature.encode()).hexdigest()
            split_hash = hashlib.sha256((policy['split_seed'] + group).encode()).hexdigest()
            split = 'test' if int(split_hash[:16], 16) / 2**64 < policy['test_fraction'] else 'train'
            signature_ids[group].append(values['id'])
            signature_labels[group].add(values['cardio'])
            clean.append({**values, **derived, 'source_row': line, 'profile_group': group,
                          'split': split, 'quality_flags': '|'.join(flags)})
        else:
            all_reasons.update(reasons)
            first_reasons[reasons[0]] += 1
            excluded.append({**original, **derived, 'source_row': line,
                             'exclusion_reasons': '|'.join(reasons), 'quality_flags': '|'.join(flags)})
        annotated.append({**original, **derived, 'source_row': line, 'profile_group': group,
                          'split': split, 'quality_flags': '|'.join(flags),
                          'qc_status': 'excluded' if reasons else 'retained',
                          'exclusion_reasons': '|'.join(reasons)})

    columns = FIELDS + DERIVED + META
    write_csv(processed / 'cardio_clean.csv', clean, columns)
    write_csv(processed / 'cardio_excluded.csv', excluded,
              FIELDS + DERIVED + ['source_row', 'exclusion_reasons', 'quality_flags'])
    write_csv(processed / 'cardio_all_annotated.csv', annotated,
              columns + ['qc_status', 'exclusion_reasons'])
    train = [r for r in clean if r['split'] == 'train']
    test = [r for r in clean if r['split'] == 'test']
    write_csv(processed / 'train.csv', train, columns)
    write_csv(processed / 'test.csv', test, columns)
    write_csv(reports / 'cleaning_rule_counts.csv', [
        {'rule': key, 'rows_matching_rule': all_reasons[key], 'rows_first_excluded_by_rule': first_reasons[key]}
        for key in all_reasons], ['rule', 'rows_matching_rule', 'rows_first_excluded_by_rule'])
    duplicates = [{'profile_group': g, 'record_count': len(record_ids),
                   'record_ids': '|'.join(map(str, record_ids)),
                   'label_values': '|'.join(map(str, sorted(signature_labels[g])))}
                  for g, record_ids in signature_ids.items() if len(record_ids) > 1]
    write_csv(reports / 'matching_profiles.csv', duplicates,
              ['profile_group', 'record_count', 'record_ids', 'label_values'])
    for name, rows in [('before', source), ('after', clean)]:
        groups = Counter((r['gender'], r['cardio']) for r in rows)
        write_csv(reports / f'gender_label_{name}.csv',
                  [{'gender': g, 'cardio': y, 'records': n} for (g, y), n in sorted(groups.items())],
                  ['gender', 'cardio', 'records'])

    summary = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(),
        'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'policy': policy, 'missing_by_field': missing,
        'duplicate_id_excess_rows': sum(n - 1 for n in ids.values()),
        'exact_duplicate_excess_rows': sum(n - 1 for n in full_rows.values()),
        'same_profile_and_label_excess_rows_ignoring_id': sum(n - 1 for n in raw_profile_count.values()),
        'before': profile(source), 'after': profile(clean),
        'excluded_rows': len(excluded), 'excluded_fraction': len(excluded) / len(source),
        'rule_counts_overlapping': dict(all_reasons), 'first_exclusion_counts': dict(first_reasons),
        'retained_review_flag_rows': sum(bool(r['quality_flags']) for r in clean),
        'matching_predictor_groups_retained': len(duplicates),
        'train': profile(train), 'test': profile(test),
        'split_method': 'Deterministic SHA256 by original predictor profile; approximately 80/20; target-independent; not exact stratification.',
    }
    (reports / 'cleaning_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    # Validate on the exported files, not just in-memory objects.
    def load(name):
        with (processed / name).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    saved, saved_train, saved_test = load('cardio_clean.csv'), load('train.csv'), load('test.csv')
    assert len(saved) + len(load('cardio_excluded.csv')) == len(source)
    assert len(load('cardio_all_annotated.csv')) == len(source)
    assert len(saved_train) + len(saved_test) == len(saved)
    assert len({r['id'] for r in saved}) == len(saved)
    assert not ({r['id'] for r in saved_train} & {r['id'] for r in saved_test})
    assert not ({r['profile_group'] for r in saved_train} & {r['profile_group'] for r in saved_test})
    for r in saved:
        assert all(v != '' for k, v in r.items() if k != 'quality_flags')
        assert float(r['ap_hi']) > float(r['ap_lo'])
        assert abs(float(r['bmi']) - float(r['weight']) / (float(r['height']) / 100)**2) < 1e-6
        assert abs(float(r['age_years']) - float(r['age']) / 365.25) < 1e-6
        assert float(r['pulse_pressure']) == float(r['ap_hi']) - float(r['ap_lo'])
        for c, (lo, hi) in policy['bounds'].items():
            assert lo <= float(r[c]) <= hi
        for c, allowed in policy['allowed_categories'].items():
            assert int(r[c]) in allowed
    (reports / 'validation.txt').write_text(
        'PASS: exported CSVs reloaded successfully.\n'
        'PASS: retained + excluded = source; all annotated rows preserved.\n'
        'PASS: unique retained IDs; no train/test ID or predictor-profile overlap.\n'
        'PASS: ranges, categories, required fields, BP order and all derived numeric values checked.\n'
        'PASS: no target-based feature engineering or learned preprocessing.\n', encoding='utf-8')
    lines = [
        'Cardio 数据清洗报告',
        f'原始记录：{len(source):,}；保留：{len(clean):,}；隔离：{len(excluded):,}（{len(excluded)/len(source):.2%}）。',
        f'疾病标签为1的比例：清洗前 {summary["before"]["positive_fraction"]:.4%}；清洗后 {summary["after"]["positive_fraction"]:.4%}。',
        f'原始缺失值总数：{sum(missing.values())}；重复ID多余记录：{summary["duplicate_id_excess_rows"]}。',
        f'忽略ID后相同资料及标签的多余记录：{summary["same_profile_and_label_excess_rows_ignoring_id"]}，未直接删除。',
        '', '清洗规则（项目质量控制假设，不是临床正常值或诊断界限）',
        '年龄：1–43830天；身高：100–250cm；体重：20–300kg；收缩压：50–300mmHg；舒张压：30–200mmHg。界限包含端点。',
        '收缩压必须高于舒张压；类别须属于数据编码集合；缺失、非数值、非有限值或无效ID隔离。',
        '超界值无法确认录入错误还是实际极端值，因此只隔离，不推测修正、不交换血压、不截断。',
        '上述规则改变分析人群范围，可能影响结果；评估结论仅适用于通过这些规则的记录。',
        'BMI <10或>70只标记复核，不自动排除。未使用IQR、分位数或疾病标签挑选记录。',
        '', '各规则命中数可重叠；首次排除数互斥，求和等于隔离总数：',
    ]
    for k in all_reasons:
        lines.append(f'{k}: 命中 {all_reasons[k]}；首次排除 {first_reasons[k]}')
    lines += ['', '衍生字段', 'age_years = age / 365.25；bmi = weight / (height/100)^2；pulse_pressure = ap_hi - ap_lo。',
              '数值衍生字段保留6位小数；分组基于未四舍五入值；原始测量值保持不变。',
              'age_group为展示区间；bmi_band为数值展示区间，不是新增临床诊断。',
              f'保留记录中BMI复核标记数量：{summary["retained_review_flag_rows"]}。',
              '', '预测训练准备',
              f'train.csv：{len(train):,}条，标签1比例{summary["train"]["positive_fraction"]:.4%}。',
              f'test.csv：{len(test):,}条，标签1比例{summary["test"]["positive_fraction"]:.4%}。',
              '按原始11项预测特征的相同组合分组，再用固定种子的SHA256划分，约80/20，并非严格分层抽样。',
              '相同健康特征组合不跨训练/测试集；不同ID的相同资料可能是真实不同人，因此保留。',
              '开发、探索、调参和阈值选择用train.csv；test.csv留作最终评估。',
              'cardio为目标；id、source_row、profile_group、split、quality_flags不能作为模型输入。',
              '没有标准化、插补、过采样或模型训练；后续所有拟合型预处理须在训练折内部完成。',
              'gender保留1/2编码；随附README没有明确性别映射，未猜测添加男女标签。',
              '', '来源', '本地用户提供的cardio-master/cardio_train.csv，原始文件完整保留。',
              'https://github.com/caravanuden/cardio',
              'https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset',
              f'SHA256: {summary["source_sha256"]}', '', '验证结果：见 validation.txt。']
    (reports / 'cleaning_report_zh.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['excluded_rows', 'rule_counts_overlapping', 'retained_review_flag_rows']}, ensure_ascii=False))
    print(f'Raw={len(source)} Clean={len(clean)} Train={len(train)} Test={len(test)}')


if __name__ == '__main__':
    main()
