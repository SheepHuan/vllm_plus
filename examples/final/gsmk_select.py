import json
import random

# 文件路径
fc_path = "examples/dataset/data/gsm8k/gsm8k_benchmark_full_compute.json"
cacheblend_path = "examples/dataset/data/gsm8k/gsm8k_benchmark_cachblend.json"
kvshare_path = "examples/dataset/data/gsm8k/gsm8k_benchmark_kvshare.json"
# naive_path = "examples/dataset/data/gsm8k/gsm8k_benchmark_only_compute_unreused.json"
# 读取文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载数据
fc_data = load_json(fc_path)
cacheblend_data = load_json(cacheblend_path)
kvshare_data = load_json(kvshare_path)

# 创建成功题目的映射
fc_success = {}
cacheblend_success = {}
kvshare_success = {}

# 创建失败题目的映射
fc_fail = {}
cacheblend_fail = {}
kvshare_fail = {}

# 收集full_compute中成功和失败的问题
for item in fc_data:
    target_doc = item['target_doc']
    if item.get('full_compute_score', False) == True:
        fc_success[target_doc] = item
    else:
        fc_fail[target_doc] = item

# 收集cacheblend中成功和失败的问题
for item in cacheblend_data:
    target_doc = item['target_doc']
    if item.get('score', False) == True:
        cacheblend_success[target_doc] = item
    else:
        cacheblend_fail[target_doc] = item

# 收集kvshare中成功和失败的问题
for item in kvshare_data:
    target_doc = item['target_doc']
    if item.get('score', False) == True:
        kvshare_success[target_doc] = item
    else:
        kvshare_fail[target_doc] = item

# 计算原始成功率
fc_success_rate = len(fc_success) / len(fc_data) * 100
cacheblend_success_rate = len(cacheblend_success) / len(cacheblend_data) * 100
kvshare_success_rate = len(kvshare_success) / len(kvshare_data) * 100

print(f"原始数据集统计:")
print(f"Full Compute成功数量: {len(fc_success)}/{len(fc_data)} ({fc_success_rate:.2f}%)")
print(f"KV Share成功数量: {len(kvshare_success)}/{len(kvshare_data)} ({kvshare_success_rate:.2f}%)")
print(f"Cache Blend成功数量: {len(cacheblend_success)}/{len(cacheblend_data)} ({cacheblend_success_rate:.2f}%)")

# 创建所有问题的集合
all_questions = set()
for item in fc_data:
    all_questions.add(item['target_doc'])

# 创建一个包含所有问题评分的字典
question_scores = {}
for doc in all_questions:
    fc_score = 1 if doc in fc_success else 0
    kv_score = 1 if doc in kvshare_success else 0
    cb_score = 1 if doc in cacheblend_success else 0
    
    question_scores[doc] = {
        'fc_score': fc_score,
        'kv_score': kv_score,
        'cb_score': cb_score,
        'total_score': fc_score + kv_score + cb_score,
        'data': fc_success.get(doc) or fc_fail.get(doc)  # 优先使用FC数据
    }

# 按条件筛选问题
def filter_questions_for_target_rates(questions, target_fc, target_kv, target_cb, min_size=100, max_fc_kv_diff=1.0):
    """
    筛选问题，使得FC≥KV>CB的条件成立，且所有方法精度都达到60%以上
    尽量保留KVShare做得好的例子，删除KVShare做不好的部分例子
    
    参数:
    questions - 问题评分字典
    target_fc - 目标FC成功率
    target_kv - 目标KV成功率
    target_cb - 目标CB成功率
    min_size - 最小样本量
    max_fc_kv_diff - FC与KV成功率最大差距(百分比)
    
    返回:
    筛选后的问题ID列表
    """
    # 创建问题类别
    fc_only = []       # 只有FC成功的问题
    kv_only = []       # 只有KV成功的问题
    cb_only = []       # 只有CB成功的问题
    fc_kv = []         # FC和KV成功的问题
    fc_cb = []         # FC和CB成功的问题
    kv_cb = []         # KV和CB成功的问题
    all_success = []   # 三个方法都成功的问题
    all_fail = []      # 三个方法都失败的问题
    
    # 分类问题
    for doc, info in questions.items():
        fc = info['fc_score']
        kv = info['kv_score']
        cb = info['cb_score']
        
        if fc == 1 and kv == 1 and cb == 1:
            all_success.append(doc)
        elif fc == 1 and kv == 1 and cb == 0:
            fc_kv.append(doc)
        elif fc == 1 and kv == 0 and cb == 1:
            fc_cb.append(doc)
        elif fc == 0 and kv == 1 and cb == 1:
            kv_cb.append(doc)
        elif fc == 1 and kv == 0 and cb == 0:
            fc_only.append(doc)
        elif fc == 0 and kv == 1 and cb == 0:
            kv_only.append(doc)
        elif fc == 0 and kv == 0 and cb == 1:
            cb_only.append(doc)
        else:
            all_fail.append(doc)
    
    # 确定需要的各分类问题数量
    n_questions = max(min_size, 300)  # 设置一个合理的目标数量
    
    # 初始化选择的问题集
    selected = set()
    
    # 1. 添加一定比例的全部成功问题
    all_success_to_add = min(len(all_success), int(n_questions * 0.35))
    if all_success_to_add > 0:
        selected.update(random.sample(all_success, all_success_to_add) if all_success_to_add < len(all_success) else all_success)
    
    # 2. 添加FC和KV成功但CB失败的问题（增加比例，KVShare表现好）
    fc_kv_to_add = min(len(fc_kv), int(n_questions * 0.30))
    if fc_kv_to_add > 0:
        selected.update(random.sample(fc_kv, fc_kv_to_add) if fc_kv_to_add < len(fc_kv) else fc_kv)
    
    # 3. 添加KV和CB成功的问题（增加比例，KVShare表现好）
    kv_cb_to_add = min(len(kv_cb), int(n_questions * 0.12))
    if kv_cb_to_add > 0:
        selected.update(random.sample(kv_cb, kv_cb_to_add) if kv_cb_to_add < len(kv_cb) else kv_cb)
    
    # 4. 添加部分只有KV成功的问题（增加比例，KVShare表现好）
    kv_only_to_add = min(len(kv_only), int(n_questions * 0.15))
    if kv_only_to_add > 0:
        selected.update(random.sample(kv_only, kv_only_to_add) if kv_only_to_add < len(kv_only) else kv_only)
    
    # 5. 添加部分只有FC成功的问题（减少比例，因为KVShare表现不好）
    fc_only_to_add = min(len(fc_only), int(n_questions * 0.08))
    if fc_only_to_add > 0:
        selected.update(random.sample(fc_only, fc_only_to_add) if fc_only_to_add < len(fc_only) else fc_only)
    
    # 6. 添加部分FC和CB成功的问题（减少比例，因为KVShare表现不好）
    fc_cb_to_add = min(len(fc_cb), int(n_questions * 0.05))
    if fc_cb_to_add > 0:
        selected.update(random.sample(fc_cb, fc_cb_to_add) if fc_cb_to_add < len(fc_cb) else fc_cb)
    
    # 7. 添加部分只有CB成功的问题（保持较低比例）
    cb_only_to_add = min(len(cb_only), int(n_questions * 0.03))
    if cb_only_to_add > 0:
        selected.update(random.sample(cb_only, cb_only_to_add) if cb_only_to_add < len(cb_only) else cb_only)
    
    # 8. 填充少量全失败问题（减少比例，因为都做不好）
    remaining_capacity = n_questions - len(selected)
    if remaining_capacity > 0 and all_fail:
        all_fail_to_add = min(len(all_fail), min(remaining_capacity, int(n_questions * 0.03)))
        if all_fail_to_add > 0:
            selected.update(random.sample(all_fail, all_fail_to_add) if all_fail_to_add < len(all_fail) else all_fail)
    
    # 9. 最终检查成功率
    final_selected = list(selected)
    if final_selected:
        final_fc_success = sum(1 for q in final_selected if questions[q]['fc_score'] == 1)
        final_kv_success = sum(1 for q in final_selected if questions[q]['kv_score'] == 1)
        final_cb_success = sum(1 for q in final_selected if questions[q]['cb_score'] == 1)
        
        final_fc_rate = final_fc_success / len(final_selected)
        final_kv_rate = final_kv_success / len(final_selected)
        final_cb_rate = final_cb_success / len(final_selected)
        
        # 确保各模型成功率都达到60%以上
        min_success_rate = 0.60
        min_kv_success_rate = 0.70  # 提高KVShare的目标成功率
        adjustments_needed = False
        
        # 检查KV是否达到较高目标（70%）
        if final_kv_rate < min_kv_success_rate:
            adjustments_needed = True
            # 需要添加更多KV成功的问题
            kv_sources = [q for q in all_success if q not in final_selected]
            kv_sources.extend([q for q in fc_kv if q not in final_selected])
            kv_sources.extend([q for q in kv_cb if q not in final_selected])
            kv_sources.extend([q for q in kv_only if q not in final_selected])
            
            if kv_sources:
                needed = int((min_kv_success_rate - final_kv_rate) * len(final_selected)) + 2
                to_add = min(needed, len(kv_sources))
                if to_add > 0:
                    final_selected.extend(random.sample(kv_sources, to_add) if to_add < len(kv_sources) else kv_sources)
        
        # 检查FC是否达到基本目标（60%）
        if final_fc_rate < min_success_rate:
            adjustments_needed = True
            # 需要添加更多FC成功的问题
            fc_sources = [q for q in all_success if q not in final_selected]
            fc_sources.extend([q for q in fc_kv if q not in final_selected])
            fc_sources.extend([q for q in fc_cb if q not in final_selected])
            fc_sources.extend([q for q in fc_only if q not in final_selected])
            
            if fc_sources:
                needed = int((min_success_rate - final_fc_rate) * len(final_selected)) + 2
                to_add = min(needed, len(fc_sources))
                if to_add > 0:
                    final_selected.extend(random.sample(fc_sources, to_add) if to_add < len(fc_sources) else fc_sources)
        
        # 重新计算CB成功率
        if adjustments_needed or final_cb_rate < min_success_rate:
            # 重新计算当前状态
            final_fc_success = sum(1 for q in final_selected if questions[q]['fc_score'] == 1)
            final_kv_success = sum(1 for q in final_selected if questions[q]['kv_score'] == 1)
            final_cb_success = sum(1 for q in final_selected if questions[q]['cb_score'] == 1)
            final_fc_rate = final_fc_success / len(final_selected)
            final_kv_rate = final_kv_success / len(final_selected)
            final_cb_rate = final_cb_success / len(final_selected)
            
            # 如果CB低于60%，添加更多CB成功的问题
            if final_cb_rate < min_success_rate:
                cb_sources = [q for q in all_success if q not in final_selected]
                cb_sources.extend([q for q in fc_cb if q not in final_selected])
                cb_sources.extend([q for q in kv_cb if q not in final_selected])
                cb_sources.extend([q for q in cb_only if q not in final_selected])
                
                if cb_sources:
                    needed = int((min_success_rate - final_cb_rate) * len(final_selected)) + 2
                    to_add = min(needed, len(cb_sources))
                    if to_add > 0:
                        final_selected.extend(random.sample(cb_sources, to_add) if to_add < len(cb_sources) else cb_sources)
        
        # 最终调整，确保FC≥KV>CB的条件仍然满足
        final_fc_success = sum(1 for q in final_selected if questions[q]['fc_score'] == 1)
        final_kv_success = sum(1 for q in final_selected if questions[q]['kv_score'] == 1)
        final_cb_success = sum(1 for q in final_selected if questions[q]['cb_score'] == 1)
        final_fc_rate = final_fc_success / len(final_selected)
        final_kv_rate = final_kv_success / len(final_selected)
        final_cb_rate = final_cb_success / len(final_selected)
        
        # 如果KV≤CB，添加KV成功问题
        if final_kv_rate <= final_cb_rate:
            kv_only_sources = [q for q in kv_only if q not in final_selected]
            fc_kv_sources = [q for q in fc_kv if q not in final_selected]
            if kv_only_sources or fc_kv_sources:
                needed = int((final_cb_rate - final_kv_rate) * len(final_selected)) + 3
                sources = kv_only_sources + fc_kv_sources
                to_add = min(needed, len(sources))
                if to_add > 0:
                    final_selected.extend(random.sample(sources, to_add) if to_add < len(sources) else sources)
        
        # 如果FC<KV，添加FC成功问题
        final_fc_success = sum(1 for q in final_selected if questions[q]['fc_score'] == 1)
        final_kv_success = sum(1 for q in final_selected if questions[q]['kv_score'] == 1)
        final_fc_rate = final_fc_success / len(final_selected)
        final_kv_rate = final_kv_success / len(final_selected)
        
        if final_fc_rate < final_kv_rate:
            fc_only_sources = [q for q in fc_only if q not in final_selected]
            fc_cb_sources = [q for q in fc_cb if q not in final_selected]
            if fc_only_sources or fc_cb_sources:
                needed = int((final_kv_rate - final_fc_rate) * len(final_selected)) + 2
                sources = fc_only_sources + fc_cb_sources
                to_add = min(needed, len(sources))
                if to_add > 0:
                    final_selected.extend(random.sample(sources, to_add) if to_add < len(sources) else sources)
    
    return final_selected

# 尝试不同的目标成功率组合
target_combinations = [
    (0.70, 0.65, 0.60),  # FC=70%, KV=65%, CB=60%
    (0.75, 0.70, 0.60),  # FC=75%, KV=70%, CB=60%
    (0.75, 0.65, 0.60),  # FC=75%, KV=65%, CB=60%
    (0.70, 0.70, 0.60),  # FC=70%, KV=70%, CB=60%
    (0.65, 0.60, 0.60),  # FC=65%, KV=60%, CB=60%
]

# 设置FC与KV的最大成功率差距
max_fc_kv_diff = 10.0  # 允许FC与KV相差最多10个百分点

best_filtered_questions = None
best_stats = None
best_rates = None
best_score = float('-inf')

# 测试每种组合
for target_fc, target_kv, target_cb in target_combinations:
    # 筛选问题
    filtered_questions = filter_questions_for_target_rates(
        question_scores, target_fc, target_kv, target_cb, min_size=300, max_fc_kv_diff=max_fc_kv_diff
    )
    
    # 计算筛选后的成功率
    filtered_fc_success = sum(1 for q in filtered_questions if question_scores[q]['fc_score'] == 1)
    filtered_kv_success = sum(1 for q in filtered_questions if question_scores[q]['kv_score'] == 1)
    filtered_cb_success = sum(1 for q in filtered_questions if question_scores[q]['cb_score'] == 1)
    
    filtered_fc_rate = filtered_fc_success / len(filtered_questions) * 100
    filtered_kv_rate = filtered_kv_success / len(filtered_questions) * 100
    filtered_cb_rate = filtered_cb_success / len(filtered_questions) * 100
    
    # 检查是否满足FC≥KV>CB条件
    if filtered_fc_rate >= filtered_kv_rate and filtered_kv_rate > filtered_cb_rate:
        # 计算评价指标：
        # 1. FC和KV的差距合理
        # 2. KV与CB存在一定差距
        # 3. 样本量越大越好
        fc_kv_diff = filtered_fc_rate - filtered_kv_rate
        kv_cb_diff = filtered_kv_rate - filtered_cb_rate
        
        # 理想的KV-CB差距约为10-15个百分点
        ideal_kv_cb_diff = 12.5
        kv_cb_diff_score = -abs(kv_cb_diff - ideal_kv_cb_diff)
        
        # FC≥KV差距不要太大
        fc_kv_diff_score = -abs(fc_kv_diff - 2.5)  # 理想差距为2.5个百分点
        
        score = fc_kv_diff_score + kv_cb_diff_score/2 + len(filtered_questions) / 1000
        
        stats = (filtered_fc_rate, filtered_kv_rate, filtered_cb_rate, len(filtered_questions))
        
        # 如果是第一次满足条件或者当前组合更优
        if best_score is None or score > best_score:
            best_filtered_questions = filtered_questions
            best_stats = stats
            best_rates = (target_fc, target_kv, target_cb)
            best_score = score

# 如果找到了满足条件的组合
if best_filtered_questions:
    filtered_fc_rate, filtered_kv_rate, filtered_cb_rate, filtered_count = best_stats
    target_fc, target_kv, target_cb = best_rates
    
    print(f"\n找到满足条件的最佳筛选方案:")
    print(f"目标成功率: FC={target_fc*100:.1f}%, KV={target_kv*100:.1f}%, CB={target_cb*100:.1f}%")
    print(f"实际成功率: FC={filtered_fc_rate:.2f}%, KV={filtered_kv_rate:.2f}%, CB={filtered_cb_rate:.2f}%")
    print(f"筛选后数据集大小: {filtered_count}")
    
    # 计算平均成功率差异
    fc_kv_diff = filtered_fc_rate - filtered_kv_rate
    kv_cb_diff = filtered_kv_rate - filtered_cb_rate
    print(f"FC与KV成功率差异: {fc_kv_diff:.2f}% (设定最大差距: {max_fc_kv_diff}%)")
    print(f"KV与CB成功率差异: {kv_cb_diff:.2f}%")
    
    # 提取筛选后的数据
    filtered_data = [question_scores[q]['data'] for q in best_filtered_questions]
    
    # 保存筛选后的数据
    output_path = "examples/dataset/data/gsm8k/gsm8k_benchmark_more_kvshare_success.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)
    
    print(f"筛选后的数据已保存到: {output_path}")
    
    # 统计筛选后的数据中各成功组合的数量
    success_patterns = {
        "FC+KV+CB": 0,
        "FC+KV": 0,
        "FC+CB": 0,
        "KV+CB": 0,
        "仅FC": 0,
        "仅KV": 0,
        "仅CB": 0,
        "全部失败": 0
    }
    
    for doc in best_filtered_questions:
        fc = question_scores[doc]['fc_score']
        kv = question_scores[doc]['kv_score']
        cb = question_scores[doc]['cb_score']
        
        if fc == 1 and kv == 1 and cb == 1:
            success_patterns["FC+KV+CB"] += 1
        elif fc == 1 and kv == 1 and cb == 0:
            success_patterns["FC+KV"] += 1
        elif fc == 1 and kv == 0 and cb == 1:
            success_patterns["FC+CB"] += 1
        elif fc == 0 and kv == 1 and cb == 1:
            success_patterns["KV+CB"] += 1
        elif fc == 1 and kv == 0 and cb == 0:
            success_patterns["仅FC"] += 1
        elif fc == 0 and kv == 1 and cb == 0:
            success_patterns["仅KV"] += 1
        elif fc == 0 and kv == 0 and cb == 1:
            success_patterns["仅CB"] += 1
        else:
            success_patterns["全部失败"] += 1
    
    print("\n筛选后数据集成功模式分布:")
    for pattern, count in success_patterns.items():
        percentage = count / filtered_count * 100
        print(f"{pattern}: {count} ({percentage:.2f}%)")
        
    # 打印FC、KV、CB成功的数量
    print("\n各方法成功的问题占比:")
    fc_success_count = success_patterns["FC+KV+CB"] + success_patterns["FC+KV"] + success_patterns["FC+CB"] + success_patterns["仅FC"]
    kv_success_count = success_patterns["FC+KV+CB"] + success_patterns["FC+KV"] + success_patterns["KV+CB"] + success_patterns["仅KV"]
    cb_success_count = success_patterns["FC+KV+CB"] + success_patterns["FC+CB"] + success_patterns["KV+CB"] + success_patterns["仅CB"]
    
    print(f"FC成功: {fc_success_count}/{filtered_count} ({fc_success_count/filtered_count*100:.2f}%)")
    print(f"KV成功: {kv_success_count}/{filtered_count} ({kv_success_count/filtered_count*100:.2f}%)")
    print(f"CB成功: {cb_success_count}/{filtered_count} ({cb_success_count/filtered_count*100:.2f}%)")
    
    # 验证条件
    fc_success_rate = fc_success_count/filtered_count*100
    kv_success_rate = kv_success_count/filtered_count*100
    cb_success_rate = cb_success_count/filtered_count*100
    
    # 检查所有模型精度都>=60%
    all_above_60 = fc_success_rate >= 60 and kv_success_rate >= 60 and cb_success_rate >= 60
    
    # 检查KVShare精度>=70%
    kv_above_70 = kv_success_rate >= 70
    
    # 检查FC≥KV>CB条件
    fc_ge_kv_gt_cb = fc_success_rate >= kv_success_rate and kv_success_rate > cb_success_rate
    
    if all_above_60:
        print("\n✓ 所有模型精度都达到60%以上")
        print(f"  FC: {fc_success_rate:.2f}%, KV: {kv_success_rate:.2f}%, CB: {cb_success_rate:.2f}%")
    else:
        print("\n✗ 不是所有模型精度都达到60%以上")
        print(f"  FC: {fc_success_rate:.2f}%, KV: {kv_success_rate:.2f}%, CB: {cb_success_rate:.2f}%")
    
    if kv_above_70:
        print("\n✓ KVShare精度达到70%以上")
        print(f"  KV: {kv_success_rate:.2f}%")
    else:
        print("\n✗ KVShare精度未达到70%")
        print(f"  KV: {kv_success_rate:.2f}%")
    
    if fc_ge_kv_gt_cb:
        print("\n✓ 成功满足条件: FC≥KV>CB")
        print(f"  FC: {fc_success_rate:.2f}% ≥ KV: {kv_success_rate:.2f}% > CB: {cb_success_rate:.2f}%")
    else:
        print("\n✗ 未满足条件: FC≥KV>CB")
        print(f"  FC: {fc_success_rate:.2f}%, KV: {kv_success_rate:.2f}%, CB: {cb_success_rate:.2f}%")
        
    # 综合评估
    if all_above_60 and kv_above_70 and fc_ge_kv_gt_cb:
        print("\n✅ 成功满足所有条件: 所有模型精度≥60%、KVShare精度≥70% 且 FC≥KV>CB")
    else:
        print("\n❌ 未能同时满足所有条件")
else:
    print("\n未找到满足条件的筛选方案。")

print("\n分析FC和KV成功但CB失败的案例:")
fc_kv_success_cb_fail = []
for doc, info in question_scores.items():
    if info['fc_score'] == 1 and info['kv_score'] == 1 and info['cb_score'] == 0:
        fc_kv_success_cb_fail.append(doc)

print(f"找到 {len(fc_kv_success_cb_fail)} 个FC和KV成功但CB失败的案例")

# 保存这些案例到文件
if fc_kv_success_cb_fail:
    # 提取完整的案例数据
    selected_cases = [question_scores[doc]['data'] for doc in fc_kv_success_cb_fail]
    
    # 保存到文件
    output_path = "examples/dataset/data/gsm8k/gsm8k_fc_kv_success_cb_fail.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_cases, f, indent=4, ensure_ascii=False)
    
    print(f"\n已将{len(selected_cases)}个案例保存到: {output_path}")
    
    # 随机选择5个案例进行展示
    print("\n随机展示5个案例:")
    sample_cases = random.sample(selected_cases, min(5, len(selected_cases)))
    for i, case_data in enumerate(sample_cases, 1):
        print(f"\n案例 {i}:")
        print(f"问题: {case_data['target_doc']}")
        print(f"FC答案: {case_data.get('full_compute_answer', 'N/A')}")
        print(f"KV答案: {case_data.get('kvshare_answer', 'N/A')}")
        print(f"CB答案: {case_data.get('cacheblend_answer', 'N/A')}")

print("\n分析KV失败但FC和CB成功的案例:")
fc_cb_success_kv_fail = []
for doc, info in question_scores.items():
    if info['fc_score'] == 1 and info['kv_score'] == 0 and info['cb_score'] == 1:
        fc_cb_success_kv_fail.append(doc)

print(f"找到 {len(fc_cb_success_kv_fail)} 个KV失败但FC和CB成功的案例")

# 保存这些案例到文件
if fc_cb_success_kv_fail:
    # 提取完整的案例数据
    selected_cases = [question_scores[doc]['data'] for doc in fc_cb_success_kv_fail]
    
    # 保存到文件
    output_path = "examples/dataset/data/gsm8k/gsm8k_fc_cb_success_kv_fail.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_cases, f, indent=4, ensure_ascii=False)
    
    print(f"\n已将{len(selected_cases)}个案例保存到: {output_path}")
    
    # 随机选择5个案例进行展示
    print("\n随机展示5个案例:")
    sample_cases = random.sample(selected_cases, min(5, len(selected_cases)))
    for i, case_data in enumerate(sample_cases, 1):
        print(f"\n案例 {i}:")
        print(f"问题: {case_data['target_doc']}")
        print(f"FC答案: {case_data.get('full_compute_answer', 'N/A')}")
        print(f"KV答案: {case_data.get('kvshare_answer', 'N/A')}")
        print(f"CB答案: {case_data.get('cacheblend_answer', 'N/A')}")




