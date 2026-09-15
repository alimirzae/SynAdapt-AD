import pandas as pd

def export_results_to_markdown_and_latex(df, name_prefix="benchmark_table"):
    md_str = df.to_markdown(index=False)
    with open(f"results/tables/{name_prefix}.md", "w", encoding="utf-8") as f:
        f.write(md_str)
    
    latex_str = df.to_latex(index=False, float_format="%.2f")
    with open(f"results/tables/{name_prefix}.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"✅ Exported tables to results/tables/{name_prefix}.md and .tex")
