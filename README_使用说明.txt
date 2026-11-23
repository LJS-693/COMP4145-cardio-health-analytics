COMP4145 Cardio Final Project

当前完成范围
已完成原始文件归档、数据质量检查、可复现清洗、衍生字段、训练/测试文件准备及导出验证。
已完成描述性分析及逻辑回归基线的五折分组交叉验证；最终测试集尚未用于模型评估。逻辑回归结果见reports/logistic_regression。

目录
data/raw/cardio_train.csv                用户提供的原始数据，分号分隔，保持原样
data/processed/cardio_clean.csv          全部通过清洗规则的记录，可导入Power BI
data/processed/cardio_all_annotated.csv  所有原始记录加状态和排除原因，便于审计
data/processed/cardio_excluded.csv       隔离记录及具体原因，不等于确定的录入错误
data/processed/train.csv                 约80%开发数据，用于EDA、交叉验证、调参
data/processed/test.csv                  约20%最终测试数据，不用于选模型和调门槛
config/cleaning_rules.json               固定清洗界限及复核规则
config/feature_sets.json                 预测特征清单和排除字段
src/clean_data.py                        标准库清洗程序，不依赖pandas或sklearn
reports/cleaning_report_zh.txt           中文清洗报告，包含界限依据与限制
reports/cleaning_summary.json            机器可读统计和原始数据校验值
reports/cleaning_rule_counts.csv         规则命中和首次排除数量
reports/data_dictionary.csv              原始、衍生与辅助字段说明
reports/matching_profiles.csv            相同预测资料记录，用于重复资料复核
reports/gender_label_before.csv          清洗前性别编码与标签分布
reports/gender_label_after.csv           清洗后性别编码与标签分布
reports/validation.txt                   已执行的数据验证结果
references/original_cardio-master/       用户提供的整个文件夹副本，仅作参考
models/                                 后续保存拟合好的模型
notebooks/                              后续保存EDA与训练Notebook
outputs/figures/                         后续保存图表
outputs/predictions/                     后续保存测试集预测与评估结果
powerbi/                                后续保存Power BI报告

重新运行
在本文件夹打开终端，运行：
python3 src/clean_data.py

程序以脚本所在位置定位项目根目录，所以项目路径包含空格也能运行。
修改config中的界限后重新运行会覆盖processed和对应reports文件，请先备份需要保留的版本。

Power BI使用
导入data/processed/cardio_clean.csv即可浏览清洗后总体；探索开发优先导入train.csv。
输出CSV为逗号分隔、UTF-8 BOM编码。cardio、gender及类别编码应为整数；连续字段应为数值。
若CSV路径变化，需在Power BI中重新指定来源路径。
cardio均值可表示当前筛选组的疾病标签比例；同时显示组内记录数。

预测训练使用
特征与目标字段分开读取，使用config/feature_sets.json中的白名单，不能将整表直接当作X。
train/test采用固定、与目标无关的近似80/20分组哈希划分，不是严格分层抽样。
profile_group来自11项原始预测字段，排除了id和cardio；相同资料不会跨train/test。
交叉验证也建议使用profile_group作groups，避免同资料跨折；group本身不是模型输入。
id是记录编号，不能证明不同记录对应不同患者。
标准化、缺失填补、特征选择和采样仅在训练折拟合，测试集只作最终评估。
评价结果适用于通过质量规则的人群；对原始人群应用时必须报告不可评分/需复核比例。
不以cardio标签或模型性能决定清洗阈值。BMI复核标记不是疾病标签。

原有工作与引用
references中的notebooks和预处理文件是第三方历史工作，未执行、未当作本组结果。
本次从原始cardio_train.csv独立运行src/clean_data.py；原有清洗、归一化数据没有用于本次输出。
原README不能充当经核实的最新临床证据；项目陈述中的医学和经济数字需另行核实。
数据源：https://github.com/caravanuden/cardio
原数据发布页：https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
提交时按课程要求披露ChatGPT使用：本次用于生成清洗程序、目录结构和说明，并执行数据校验。
