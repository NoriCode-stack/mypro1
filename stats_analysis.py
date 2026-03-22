import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Union, Optional

def calc_stats(data: List[Union[int, float, str, None]]) -> Dict[str, Optional[Union[float, int]]]:
    """
    计算列表数据的统计值
    修复了两个bug：
    1. 处理字符串数字和None值的情况
    2. 处理空列表的情况
    """
    # 过滤并转换有效数据
    valid_data = []
    for item in data:
        if item is None:
            continue
        try:
            num = float(item)
            valid_data.append(num)
        except (ValueError, TypeError):
            continue
    
    if not valid_data:
        return {
            'mean': None,
            'max': None,
            'min': None,
            'count': 0
        }
    
    return {
        'mean': sum(valid_data) / len(valid_data),
        'max': max(valid_data),
        'min': min(valid_data),
        'count': len(valid_data)
    }

def test_calc_stats():
    """回归测试用例"""
    print("=" * 50)
    print("运行回归测试用例...")
    print("=" * 50)
    
    # 测试用例1：包含字符串数字和None的混合输入
    test_data1 = [10, '20', None, 30, '40.5', None, 50]
    result1 = calc_stats(test_data1)
    expected_mean = (10 + 20 + 30 + 40.5 + 50) / 5  # 150.5 / 5 = 30.1
    print(f"测试用例1 - 混合输入（包含字符串数字和None）:")
    print(f"  输入: {test_data1}")
    print(f"  期望平均值: {expected_mean:.2f}, 实际平均值: {result1['mean']:.2f}")
    print(f"  期望数量: 5, 实际数量: {result1['count']}")
    print(f"  测试结果: {'通过' if abs(result1['mean'] - expected_mean) < 0.001 and result1['count'] == 5 else '失败'}")
    print()
    
    # 测试用例2：空列表输入
    test_data2 = []
    result2 = calc_stats(test_data2)
    print(f"测试用例2 - 空列表输入:")
    print(f"  输入: {test_data2}")
    print(f"  期望平均值: None, 实际平均值: {result2['mean']}")
    print(f"  期望数量: 0, 实际数量: {result2['count']}")
    print(f"  测试结果: {'通过' if result2['mean'] is None and result2['count'] == 0 else '失败'}")
    print()
    
    # 测试用例3：纯数字输入（基准测试）
    test_data3 = [1, 2, 3, 4, 5]
    result3 = calc_stats(test_data3)
    print(f"测试用例3 - 纯数字输入:")
    print(f"  输入: {test_data3}")
    print(f"  期望平均值: 3.0, 实际平均值: {result3['mean']}")
    print(f"  测试结果: {'通过' if result3['mean'] == 3.0 else '失败'}")
    print()
    
    # 测试用例4：包含无效字符串的输入
    test_data4 = [10, 'abc', None, '20', 'xyz', 30]
    result4 = calc_stats(test_data4)
    print(f"测试用例4 - 包含无效字符串的输入:")
    print(f"  输入: {test_data4}")
    print(f"  有效数据: [10, 20, 30]")
    print(f"  期望平均值: 20.0, 实际平均值: {result4['mean']}")
    print(f"  测试结果: {'通过' if result4['mean'] == 20.0 else '失败'}")
    print()
    
    print("=" * 50)
    print("所有测试用例执行完成!")
    print("=" * 50)

def process_and_analyze_data():
    """数据处理、统计分析和可视化完整流程"""
    # 1. 模拟原始数据（包含各种类型的混合数据）
    raw_data = {
        'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D'],
        'value': [10, '25', None, 30, '45', 50, '20', None, '35', 60, '75', None],
        'extra_info': ['x', 'y', 'z', 'w', 'v', 'u', 't', 's', 'r', 'q', 'p', 'o']
    }
    
    print("\n" + "=" * 50)
    print("原始数据:")
    print("=" * 50)
    df_raw = pd.DataFrame(raw_data)
    print(df_raw)
    print()
    
    # 2. 使用calc_stats函数处理每个类别的数据
    categories = df_raw['category'].unique()
    results = []
    
    for cat in categories:
        cat_data = df_raw[df_raw['category'] == cat]['value'].tolist()
        stats = calc_stats(cat_data)
        results.append({
            'category': cat,
            'mean': stats['mean'],
            'max': stats['max'],
            'min': stats['min'],
            'count': stats['count']
        })
    
    # 3. 转换为DataFrame并整理数据
    df_stats = pd.DataFrame(results)
    df_stats = df_stats.dropna(subset=['mean'])  # 移除没有有效数据的类别
    
    print("=" * 50)
    print("统计分析结果:")
    print("=" * 50)
    print(df_stats.round(2))
    print()
    
    # 4. 计算整体统计
    all_values = df_raw['value'].tolist()
    overall_stats = calc_stats(all_values)
    print("=" * 50)
    print("整体数据统计:")
    print("=" * 50)
    print(f"有效数据数量: {overall_stats['count']}")
    print(f"整体平均值: {overall_stats['mean']:.2f}" if overall_stats['mean'] else "整体平均值: 无有效数据")
    print(f"整体最大值: {overall_stats['max']}" if overall_stats['max'] else "整体最大值: 无有效数据")
    print(f"整体最小值: {overall_stats['min']}" if overall_stats['min'] else "整体最小值: 无有效数据")
    print()
    
    # 5. 生成数据分布柱状图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制柱状图
    x = np.arange(len(df_stats))
    width = 0.6
    bars = ax.bar(x, df_stats['mean'], width, label='平均值', color='#3498db', alpha=0.7)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    # 添加最大值和最小值参考线
    for i, (_, row) in enumerate(df_stats.iterrows()):
        # 最大值线
        ax.plot([i - width/2, i + width/2], [row['max'], row['max']], 
                color='#e74c3c', linestyle='--', linewidth=2, label='最大值' if i == 0 else "")
        # 最小值线
        ax.plot([i - width/2, i + width/2], [row['min'], row['min']], 
                color='#2ecc71', linestyle='--', linewidth=2, label='最小值' if i == 0 else "")
        
        # 添加最大值和最小值标注
        ax.text(i, row['max'] + 2, f'Max:{row["max"]:.0f}', 
                ha='center', va='bottom', fontsize=8, color='#e74c3c')
        ax.text(i, row['min'] - 3, f'Min:{row["min"]:.0f}', 
                ha='center', va='top', fontsize=8, color='#2ecc71')
    
    # 添加整体平均值参考线
    if overall_stats['mean']:
        ax.axhline(y=overall_stats['mean'], color='#f39c12', linestyle='-.', linewidth=2,
                   label=f'整体平均值 ({overall_stats["mean"]:.1f})')
    
    # 设置图表属性
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('数值', fontsize=12)
    ax.set_title('各类别数据统计分布', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_stats['category'])
    ax.legend()
    ax.set_ylim(bottom=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('d:/mytest/mypro1/data_distribution.png', dpi=300, bbox_inches='tight')
    print("=" * 50)
    print("图表已保存为: data_distribution.png")
    print("=" * 50)
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    # 运行回归测试
    test_calc_stats()
    
    # 运行完整的数据分析流程
    process_and_analyze_data()
