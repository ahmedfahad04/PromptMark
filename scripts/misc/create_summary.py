import pandas as pd
import json

# Read the CSV file
df = pd.read_csv('results/codebleu_scores.csv')

# Calculate statistics
summary = {
    "total_samples_evaluated": len(df),
    "codebleu_scores": {
        "average": float(df['codebleu'].mean()),
        "std_dev": float(df['codebleu'].std()),
        "min": float(df['codebleu'].min()),
        "max": float(df['codebleu'].max()),
        "median": float(df['codebleu'].median())
    },
    "component_scores": {
        "bleu": {
            "average": float(df['bleu'].mean()),
            "std_dev": float(df['bleu'].std())
        },
        "syntax": {
            "average": float(df['syntax'].mean()),
            "std_dev": float(df['syntax'].std())
        },
        "dataflow": {
            "average": float(df['dataflow'].mean()),
            "std_dev": float(df['dataflow'].std())
        }
    },
    "perfect_matches": {
        "count": int((df['codebleu'] == 1.0).sum()),
        "percentage": float((df['codebleu'] == 1.0).sum() / len(df) * 100)
    }
}

# Save to JSON
with open('results/codebleu_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print("=" * 80)
print("📊 CODEBLEU EVALUATION SUMMARY")
print("=" * 80)
print(f"\n✅ Total Samples Evaluated: {summary['total_samples_evaluated']}")
print(f"\n🎯 AVERAGE CODEBLEU SCORE: {summary['codebleu_scores']['average']:.4f}")
print(f"\n📈 CodeBLEU Statistics:")
print(f"   Mean:    {summary['codebleu_scores']['average']:.4f}")
print(f"   Std Dev: {summary['codebleu_scores']['std_dev']:.4f}")
print(f"   Median:  {summary['codebleu_scores']['median']:.4f}")
print(f"   Min:     {summary['codebleu_scores']['min']:.4f}")
print(f"   Max:     {summary['codebleu_scores']['max']:.4f}")
print(f"\n🎖️  Perfect Matches (CodeBLEU = 1.0):")
print(f"   Count:      {summary['perfect_matches']['count']}")
print(f"   Percentage: {summary['perfect_matches']['percentage']:.2f}%")
print(f"\n📊 Component Scores (Averages):")
print(f"   BLEU:     {summary['component_scores']['bleu']['average']:.4f}")
print(f"   Syntax:   {summary['component_scores']['syntax']['average']:.4f}")
print(f"   Dataflow: {summary['component_scores']['dataflow']['average']:.4f}")
print(f"\n💾 Results saved to:")
print(f"   - results/codebleu_scores.csv (detailed scores)")
print(f"   - results/codebleu_summary.json (summary statistics)")
print("=" * 80)
