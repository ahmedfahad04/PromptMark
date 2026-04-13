import pandas as pd
import json

# Read the summary CSV
df = pd.read_csv('results/experiment_codebleu/all_experiments_summary.csv')

print("=" * 100)
print("📊 COMPREHENSIVE CODEBLEU EVALUATION SUMMARY - ALL EXPERIMENTS")
print("=" * 100)

# Overall statistics
print(f"\n✅ Total Evaluations: {len(df)}")
print(f"📁 Total Experiments: {df['experiment'].nunique()}")
print(f"🤖 Total Models: {df['model'].nunique()}")
print(f"📝 Total Samples Evaluated: {df['total_samples'].sum()}")

# Overall CodeBLEU statistics
print(f"\n🎯 OVERALL CODEBLEU STATISTICS:")
print(f"   Average:  {df['avg_codebleu'].mean():.4f}")
print(f"   Median:   {df['avg_codebleu'].median():.4f}")
print(f"   Std Dev:  {df['avg_codebleu'].std():.4f}")
print(f"   Min:      {df['avg_codebleu'].min():.4f}")
print(f"   Max:      {df['avg_codebleu'].max():.4f}")

# Top 15 models
print(f"\n🏆 TOP 15 MODELS BY CODEBLEU SCORE:")
print("-" * 100)
top_models = df.nlargest(15, 'avg_codebleu')
for i, row in enumerate(top_models.itertuples(), 1):
    print(f"{i:2d}. {row.model[:55]:55s} | {row.avg_codebleu:.4f} | {row.total_samples:3d} samples | {row.experiment}")

# Bottom 10 models
print(f"\n📉 BOTTOM 10 MODELS BY CODEBLEU SCORE:")
print("-" * 100)
bottom_models = df.nsmallest(10, 'avg_codebleu')
for i, row in enumerate(bottom_models.itertuples(), 1):
    print(f"{i:2d}. {row.model[:55]:55s} | {row.avg_codebleu:.4f} | {row.total_samples:3d} samples | {row.experiment}")

# Experiment-level analysis
print(f"\n📊 EXPERIMENT-LEVEL ANALYSIS:")
print("-" * 100)
exp_stats = df.groupby('experiment').agg({
    'avg_codebleu': ['mean', 'std', 'min', 'max'],
    'model': 'count',
    'total_samples': 'sum'
}).round(4)

exp_stats.columns = ['Avg_CodeBLEU', 'Std_Dev', 'Min', 'Max', 'Num_Models', 'Total_Samples']
exp_stats = exp_stats.sort_values('Avg_CodeBLEU', ascending=False)

print(exp_stats.to_string())

# Model family analysis (extract model family from model name)
print(f"\n🤖 MODEL FAMILY ANALYSIS:")
print("-" * 100)

def extract_model_family(model_name):
    """Extract model family from model name"""
    if 'codegemma' in model_name.lower():
        return 'CodeGemma'
    elif 'gemini' in model_name.lower():
        return 'Gemini'
    elif 'qwen' in model_name.lower():
        return 'Qwen'
    elif 'claude' in model_name.lower():
        return 'Claude'
    elif 'openai' in model_name.lower():
        return 'OpenAI'
    else:
        return 'Other'

df['model_family'] = df['model'].apply(extract_model_family)

family_stats = df.groupby('model_family').agg({
    'avg_codebleu': ['mean', 'std', 'min', 'max'],
    'model': 'count',
    'total_samples': 'sum'
}).round(4)

family_stats.columns = ['Avg_CodeBLEU', 'Std_Dev', 'Min', 'Max', 'Num_Evaluations', 'Total_Samples']
family_stats = family_stats.sort_values('Avg_CodeBLEU', ascending=False)

print(family_stats.to_string())

# Save enhanced summary
enhanced_summary = {
    'overall_statistics': {
        'total_evaluations': int(len(df)),
        'total_experiments': int(df['experiment'].nunique()),
        'total_models': int(df['model'].nunique()),
        'total_samples': int(df['total_samples'].sum()),
        'avg_codebleu': float(df['avg_codebleu'].mean()),
        'median_codebleu': float(df['avg_codebleu'].median()),
        'std_codebleu': float(df['avg_codebleu'].std()),
        'min_codebleu': float(df['avg_codebleu'].min()),
        'max_codebleu': float(df['avg_codebleu'].max())
    },
    'top_10_models': [
        {
            'rank': i,
            'model': row.model,
            'experiment': row.experiment,
            'avg_codebleu': float(row.avg_codebleu),
            'total_samples': int(row.total_samples)
        }
        for i, row in enumerate(df.nlargest(10, 'avg_codebleu').itertuples(), 1)
    ],
    'experiment_statistics': {
        exp: {
            'avg_codebleu': float(stats['Avg_CodeBLEU']),
            'std_dev': float(stats['Std_Dev']),
            'min': float(stats['Min']),
            'max': float(stats['Max']),
            'num_models': int(stats['Num_Models']),
            'total_samples': int(stats['Total_Samples'])
        }
        for exp, stats in exp_stats.iterrows()
    },
    'model_family_statistics': {
        family: {
            'avg_codebleu': float(stats['Avg_CodeBLEU']),
            'std_dev': float(stats['Std_Dev']),
            'min': float(stats['Min']),
            'max': float(stats['Max']),
            'num_evaluations': int(stats['Num_Evaluations']),
            'total_samples': int(stats['Total_Samples'])
        }
        for family, stats in family_stats.iterrows()
    }
}

with open('results/experiment_codebleu/comprehensive_summary.json', 'w') as f:
    json.dump(enhanced_summary, f, indent=2)

print(f"\n💾 Enhanced summary saved to: results/experiment_codebleu/comprehensive_summary.json")
print("=" * 100)
