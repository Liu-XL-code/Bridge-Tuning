"""
实验结果表格生成工具

生成标准化的方法×中心结果表格，支持多种格式输出
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import json


class ResultTableGenerator:
    """生成实验结果表格"""

    def collect_cv_summaries(self, base_dir, timestamp):
        """
        收集所有cv_summary.json文件

        Args:
            base_dir: 基础目录
            timestamp: 时间戳

        Returns:
            List[dict]: 所有cv_summary的列表
        """
        pattern = f"{base_dir}/{timestamp}/**/cv_summary.json"
        files = glob.glob(pattern, recursive=True)

        summaries = []
        for file_path in files:
            try:
                with open(file_path) as f:
                    summaries.append(json.load(f))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Collected {len(summaries)} CV summaries")
        return summaries

    def generate_method_center_table(self, summaries):
        """
        生成方法×中心表格

        表格格式：
        - 行：(Shot, PEFT Method, Weight)
        - 列：Center_Metric (Center-1_Dice, Center-1_HD95, Center-1_IOU, ...)

        Args:
            summaries: cv_summary列表

        Returns:
            pd.DataFrame: 方法×中心表格
        """
        # 定义行索引（Shot, Method, Weight）
        rows = []
        for shot in ['5-shot', '10-shot']:
            for method in ['full_finetuning', 'linear_prob', 'bitfit', 'lora']:
                for weight in ['foundation', 'only_tcga', 'bridge']:
                    rows.append((shot, method, weight))

        # 定义列（Center_Metric）
        centers = ['Center-1', 'Center-2', 'Center-3', 'Center-4', 'Center-5']
        columns = []
        for center in centers:
            columns.extend([f'{center}_Dice', f'{center}_HD95', f'{center}_IOU'])

        # 初始化DataFrame（全部填充NaN）
        data = np.full((len(rows), len(columns)), np.nan)
        df = pd.DataFrame(
            data,
            index=pd.MultiIndex.from_tuples(rows, names=['Shot', 'Method', 'Weight']),
            columns=columns
        )

        # 从summaries填充数据
        for summary in summaries:
            method = summary.get('peft_method', '')
            center = summary.get('center', '')
            weight = summary.get('weight', '')
            results = summary.get('results', {})

            for shot_key in results.keys():
                # shot_key格式：'5shot' 或 '10shot'
                k_shot = shot_key.replace('shot', '')  # '5shot' -> '5'
                row_key = (f'{k_shot}-shot', method, weight)

                # 检查行是否存在
                if row_key in df.index:
                    shot_data = results[shot_key]

                    # 填充Dice, HD95, IOU
                    df.loc[row_key, f'{center}_Dice'] = shot_data['dice']['mean']
                    df.loc[row_key, f'{center}_HD95'] = shot_data['hd95']['mean']
                    df.loc[row_key, f'{center}_IOU'] = shot_data['iou']['mean']

        return df

    def save_table(self, df, output_dir, base_name='by_method_center'):
        """
        保存表格为多种格式

        Args:
            df: pandas DataFrame
            output_dir: 输出目录
            base_name: 基础文件名
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. CSV格式（便于Excel打开）
        csv_path = output_dir / f'{base_name}.csv'
        df.to_csv(csv_path)
        print(f"Saved CSV: {csv_path}")

        # 2. Excel格式（带格式）
        try:
            excel_path = output_dir / f'{base_name}.xlsx'
            df.to_excel(excel_path)
            print(f"Saved Excel: {excel_path}")
        except Exception as e:
            print(f"Warning: Could not save Excel format: {e}")

        # 3. LaTeX格式（用于论文）
        try:
            latex_path = output_dir / f'{base_name}.tex'
            with open(latex_path, 'w') as f:
                f.write(df.to_latex(float_format="%.4f"))
            print(f"Saved LaTeX: {latex_path}")
        except Exception as e:
            print(f"Warning: Could not save LaTeX format: {e}")

        # 4. Markdown格式（便于查看）
        try:
            md_path = output_dir / f'{base_name}.md'
            with open(md_path, 'w') as f:
                f.write(df.to_markdown())
            print(f"Saved Markdown: {md_path}")
        except Exception as e:
            print(f"Warning: Could not save Markdown format: {e}")

    def print_table_preview(self, df, max_rows=20, max_cols=15):
        """
        打印表格预览

        Args:
            df: pandas DataFrame
            max_rows: 最大显示行数
            max_cols: 最大显示列数
        """
        print("\n" + "="*120)
        print("Results Summary Table Preview")
        print("="*120)

        # 设置显示选项
        with pd.option_context('display.max_rows', max_rows,
                              'display.max_columns', max_cols,
                              'display.width', 120):
            print(df.to_string())

        print("="*120)

    def generate_summary_statistics(self, df):
        """
        生成汇总统计

        Args:
            df: 结果DataFrame

        Returns:
            dict: 统计信息
        """
        stats = {
            'total_experiments': len(df),
            'completed_experiments': df.notna().sum().sum(),
            'missing_experiments': df.isna().sum().sum(),
            'centers': [],
            'methods': [],
            'weights': []
        }

        # 提取维度信息
        if isinstance(df.index, pd.MultiIndex):
            stats['methods'] = df.index.get_level_values('Method').unique().tolist()
            stats['weights'] = df.index.get_level_values('Weight').unique().tolist()

        # 提取中心信息
        center_cols = [col.split('_')[0] for col in df.columns if '_Dice' in col]
        stats['centers'] = list(set(center_cols))

        return stats
