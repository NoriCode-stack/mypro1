import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Dict, Any, Optional

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calc_stats(data: List[Union[int, float, str, None]]) -> Dict[str, Any]:
    """
    计算列表数据的统计信息

    Args:
        data: 包含数字、字符串数字或 None 的列表

    Returns:
        包含统计信息的字典，包括平均值、最大值、最小值、数量
        如果输入为空或没有有效数据，返回 None 值
    """
    if not data:
        return {
            'mean': None,
            'max': None,
            'min': None,
            'count': 0,
            'valid_count': 0
        }

    valid_numbers = []
    for item in data:
        if item is None:
            continue
        try:
            num = float(item)
            valid_numbers.append(num)
        except (ValueError, TypeError):
            continue

    if not valid_numbers:
        return {
            'mean': None,
            'max': None,
            'min': None,
            'count': len(data),
            'valid_count': 0
        }

    return {
        'mean': sum(valid_numbers) / len(valid_numbers),
        'max': max(valid_numbers),
        'min': min(valid_numbers),
        'count': len(data),
        'valid_count': len(valid_numbers)
    }


def test_calc_stats():
    """回归测试用例"""
    print("=" * 50)
    print("运行回归测试...")
    print("=" * 50)

    test_cases = [
        {
            'name': '正常数字列表',
            'input': [10, 20, 30, 40, 50],
            'expected_mean': 30.0,
            'expected_max': 50.0,
            'expected_min': 10.0,
            'expected_count': 5,
            'expected_valid_count': 5
        },
        {
            'name': '包含字符串数字',
            'input': [10, '20', 30, '40', 50],
            'expected_mean': 30.0,
            'expected_max': 50.0,
            'expected_min': 10.0,
            'expected_count': 5,
            'expected_valid_count': 5
        },
        {
            'name': '包含 None 值',
            'input': [10, None, 30, None, 50],
            'expected_mean': 30.0,
            'expected_max': 50.0,
            'expected_min': 10.0,
            'expected_count': 5,
            'expected_valid_count': 3
        },
        {
            'name': '空列表',
            'input': [],
            'expected_mean': None,
            'expected_max': None,
            'expected_min': None,
            'expected_count': 0,
            'expected_valid_count': 0
        },
        {
            'name': '混合类型（字符串数字、None、正常数字）',
            'input': ['10', None, 30, '40', None, 50, 'abc'],
            'expected_mean': 32.5,
            'expected_max': 50.0,
            'expected_min': 10.0,
            'expected_count': 7,
            'expected_valid_count': 4
        },
        {
            'name': '全部为 None',
            'input': [None, None, None],
            'expected_mean': None,
            'expected_max': None,
            'expected_min': None,
            'expected_count': 3,
            'expected_valid_count': 0
        },
        {
            'name': '全部为无效字符串',
            'input': ['abc', 'xyz', 'hello'],
            'expected_mean': None,
            'expected_max': None,
            'expected_min': None,
            'expected_count': 3,
            'expected_valid_count': 0
        }
    ]

    all_passed = True
    for i, test in enumerate(test_cases, 1):
        result = calc_stats(test['input'])
        passed = (
            result['mean'] == test['expected_mean'] and
            result['max'] == test['expected_max'] and
            result['min'] == test['expected_min'] and
            result['count'] == test['expected_count'] and
            result['valid_count'] == test['expected_valid_count']
        )

        status = "[PASS]" if passed else "[FAIL]"
        print(f"\n测试 {i}: {test['name']} - {status}")
        print(f"  输入: {test['input']}")
        print(f"  结果: {result}")

        if not passed:
            all_passed = False
            print(f"  期望: mean={test['expected_mean']}, max={test['expected_max']}, "
                  f"min={test['expected_min']}, count={test['expected_count']}, "
                  f"valid_count={test['expected_valid_count']}")

    print("\n" + "=" * 50)
    if all_passed:
        print("所有测试通过！")
    else:
        print("存在失败的测试！")
    print("=" * 50)

    return all_passed


def process_data_with_pandas(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    使用 Pandas 处理原始数据

    Args:
        raw_data: 包含原始数据的字典列表

    Returns:
        处理后的 DataFrame
    """
    df = pd.DataFrame(raw_data)

    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

    print("\n" + "=" * 50)
    print("Pandas 数据处理结果:")
    print("=" * 50)
    print(f"\n原始数据:\n{df}")
    print(f"\n数据信息:")
    print(df.info())
    print(f"\n基本统计信息:")
    print(df.describe())

    return df


def visualize_data_distribution(df: pd.DataFrame, column: str = 'value'):
    """
    使用 Matplotlib 生成数据分布柱状图

    Args:
        df: 包含数据的 DataFrame
        column: 要可视化的列名
    """
    if column not in df.columns:
        print(f"错误: 列 '{column}' 不存在于 DataFrame 中")
        return

    valid_data = df[column].dropna()

    if len(valid_data) == 0:
        print(f"错误: 列 '{column}' 没有有效数据")
        return

    stats = calc_stats(valid_data.tolist())

    fig, ax = plt.subplots(figsize=(12, 6))

    bins = min(20, len(valid_data.unique()))
    if bins < 5:
        bins = 5

    counts, bin_edges, patches = ax.hist(valid_data, bins=bins, alpha=0.7,
                                          color='skyblue', edgecolor='black',
                                          label='数据分布')

    if stats['mean'] is not None:
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                   label=f'平均值: {stats["mean"]:.2f}')

    if stats['max'] is not None:
        ax.axvline(stats['max'], color='green', linestyle='-.', linewidth=2,
                   label=f'最大值: {stats["max"]:.2f}')

    if stats['min'] is not None:
        ax.axvline(stats['min'], color='orange', linestyle='-.', linewidth=2,
                   label=f'最小值: {stats["min"]:.2f}')

    ax.set_xlabel('数值', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title(f'{column} 数据分布柱状图', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    mean_str = f"{stats['mean']:.2f}" if stats['mean'] is not None else 'N/A'
    max_str = f"{stats['max']:.2f}" if stats['max'] is not None else 'N/A'
    min_str = f"{stats['min']:.2f}" if stats['min'] is not None else 'N/A'

    stats_text = f"""
统计摘要:
样本总数: {stats['count']}
有效样本: {stats['valid_count']}
平均值: {mean_str}
最大值: {max_str}
最小值: {min_str}
    """

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = 'data_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")

    plt.show()


def main():
    """主函数：演示完整的数据处理、统计分析和可视化流程"""

    print("\n" + "=" * 60)
    print("数据统计分析全流程演示")
    print("=" * 60)

    test_calc_stats()

    raw_data = [
        {'id': 1, 'value': 25, 'category': 'A'},
        {'id': 2, 'value': '30', 'category': 'B'},
        {'id': 3, 'value': None, 'category': 'A'},
        {'id': 4, 'value': 45, 'category': 'C'},
        {'id': 5, 'value': '55', 'category': 'B'},
        {'id': 6, 'value': 35, 'category': 'A'},
        {'id': 7, 'value': None, 'category': 'C'},
        {'id': 8, 'value': '40', 'category': 'B'},
        {'id': 9, 'value': 50, 'category': 'A'},
        {'id': 10, 'value': 28, 'category': 'C'},
        {'id': 11, 'value': 'invalid', 'category': 'A'},
        {'id': 12, 'value': 60, 'category': 'B'},
        {'id': 13, 'value': None, 'category': 'C'},
        {'id': 14, 'value': 33, 'category': 'A'},
        {'id': 15, 'value': '48', 'category': 'B'},
    ]

    print("\n" + "=" * 50)
    print("原始数据:")
    print("=" * 50)
    for item in raw_data:
        print(item)

    df = process_data_with_pandas(raw_data)

    value_list = df['value'].tolist()
    stats = calc_stats(value_list)

    print("\n" + "=" * 50)
    print("calc_stats 统计结果:")
    print("=" * 50)
    print(f"平均值: {stats['mean']}")
    print(f"最大值: {stats['max']}")
    print(f"最小值: {stats['min']}")
    print(f"总数量: {stats['count']}")
    print(f"有效数量: {stats['valid_count']}")

    print("\n" + "=" * 50)
    print("生成数据分布可视化...")
    print("=" * 50)
    visualize_data_distribution(df, 'value')

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
