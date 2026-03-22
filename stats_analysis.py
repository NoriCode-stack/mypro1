import unittest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Optional, Dict


def calc_stats(data: List[Union[int, float, str, None]]) -> Dict[str, float]:
    valid_numbers = []
    for item in data:
        if item is None:
            continue
        if isinstance(item, str):
            try:
                num = float(item)
                valid_numbers.append(num)
            except ValueError:
                continue
        elif isinstance(item, (int, float)):
            valid_numbers.append(float(item))
    
    if len(valid_numbers) == 0:
        return {
            'average': 0.0,
            'maximum': 0.0,
            'minimum': 0.0,
            'count': 0
        }
    
    return {
        'average': sum(valid_numbers) / len(valid_numbers),
        'maximum': max(valid_numbers),
        'minimum': min(valid_numbers),
        'count': len(valid_numbers)
    }


class TestCalcStats(unittest.TestCase):
    
    def test_normal_numbers(self):
        data = [10, 20, 30, 40, 50]
        result = calc_stats(data)
        self.assertEqual(result['average'], 30.0)
        self.assertEqual(result['maximum'], 50.0)
        self.assertEqual(result['minimum'], 10.0)
        self.assertEqual(result['count'], 5)
    
    def test_string_numbers(self):
        data = [10, '20', '30', 40]
        result = calc_stats(data)
        self.assertEqual(result['average'], 25.0)
        self.assertEqual(result['maximum'], 40.0)
        self.assertEqual(result['minimum'], 10.0)
        self.assertEqual(result['count'], 4)
    
    def test_none_values(self):
        data = [10, None, 30, None, 50]
        result = calc_stats(data)
        self.assertEqual(result['average'], 30.0)
        self.assertEqual(result['maximum'], 50.0)
        self.assertEqual(result['minimum'], 10.0)
        self.assertEqual(result['count'], 3)
    
    def test_mixed_types(self):
        data = [10, '20', None, 30, 'invalid', 40]
        result = calc_stats(data)
        self.assertEqual(result['average'], 25.0)
        self.assertEqual(result['maximum'], 40.0)
        self.assertEqual(result['minimum'], 10.0)
        self.assertEqual(result['count'], 4)
    
    def test_empty_list(self):
        data = []
        result = calc_stats(data)
        self.assertEqual(result['average'], 0.0)
        self.assertEqual(result['maximum'], 0.0)
        self.assertEqual(result['minimum'], 0.0)
        self.assertEqual(result['count'], 0)
    
    def test_all_none(self):
        data = [None, None, None]
        result = calc_stats(data)
        self.assertEqual(result['average'], 0.0)
        self.assertEqual(result['maximum'], 0.0)
        self.assertEqual(result['minimum'], 0.0)
        self.assertEqual(result['count'], 0)
    
    def test_all_invalid_strings(self):
        data = ['abc', 'def', 'ghi']
        result = calc_stats(data)
        self.assertEqual(result['average'], 0.0)
        self.assertEqual(result['maximum'], 0.0)
        self.assertEqual(result['minimum'], 0.0)
        self.assertEqual(result['count'], 0)
    
    def test_float_strings(self):
        data = ['10.5', '20.5', 30]
        result = calc_stats(data)
        self.assertEqual(result['average'], 20.333333333333332)
        self.assertEqual(result['maximum'], 30.0)
        self.assertEqual(result['minimum'], 10.5)
        self.assertEqual(result['count'], 3)


def process_and_visualize(raw_data: List[Union[int, float, str, None]], title: str = "数据分布分析"):
    stats = calc_stats(raw_data)
    
    valid_numbers = []
    for item in raw_data:
        if item is None:
            continue
        if isinstance(item, str):
            try:
                num = float(item)
                valid_numbers.append(num)
            except ValueError:
                continue
        elif isinstance(item, (int, float)):
            valid_numbers.append(float(item))
    
    df = pd.DataFrame({
        '原始值': raw_data,
        '有效数值': [valid_numbers[i] if i < len(valid_numbers) else None for i in range(len(raw_data))]
    })
    
    if valid_numbers:
        df_valid = pd.DataFrame({'数值': valid_numbers})
    else:
        df_valid = pd.DataFrame({'数值': pd.Series(dtype=float)})
    
    print("=" * 50)
    print("数据处理结果")
    print("=" * 50)
    print(f"\n原始数据: {raw_data}")
    print(f"有效数据数量: {stats['count']}")
    print(f"平均值: {stats['average']:.2f}")
    print(f"最大值: {stats['maximum']:.2f}")
    print(f"最小值: {stats['minimum']:.2f}")
    
    if stats['count'] > 0:
        print("\n" + "=" * 50)
        print("Pandas DataFrame 统计分析")
        print("=" * 50)
        print(f"\n{df_valid.describe()}")
    
    if stats['count'] > 0:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_positions = range(len(valid_numbers))
        bars = ax.bar(x_positions, valid_numbers, color='steelblue', alpha=0.7, edgecolor='navy', label='数据值')
        
        ax.axhline(y=stats['average'], color='red', linestyle='--', linewidth=2, label=f"平均值: {stats['average']:.2f}")
        ax.axhline(y=stats['maximum'], color='green', linestyle='-.', linewidth=2, label=f"最大值: {stats['maximum']:.2f}")
        ax.axhline(y=stats['minimum'], color='orange', linestyle='-.', linewidth=2, label=f"最小值: {stats['minimum']:.2f}")
        
        ax.set_xlabel('数据索引', fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, value) in enumerate(zip(bars, valid_numbers)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n图表已保存为: data_distribution.png")
    else:
        print("\n警告: 没有有效数据，无法生成图表")
    
    return stats, df_valid


if __name__ == '__main__':
    print("运行回归测试...")
    print("=" * 50)
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n\n" + "=" * 50)
    print("数据处理与可视化演示")
    print("=" * 50)
    
    sample_data = [25, '30', None, 45, 'invalid', 55, None, '70', 80, '90.5', 'abc', 100]
    
    stats, df = process_and_visualize(sample_data, title="示例数据分布分析")
