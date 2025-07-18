import json
from pathlib import Path
from typing import List

def build_prompt(entry: dict) -> str:
    """Constructs a robust prompt from a theorem entry."""
    prompt_lines = []

    # Basic context
    prompt_lines.append(f"Theorem: {entry.get('natural_language_statement', '').strip()}")
    prompt_lines.append(f"Formal logic: {entry.get('formal_logic', '').strip()}")

    # Proof prefix
    prefix = entry.get("partial_proof_prefix", [])
    if prefix:
        prompt_lines.append("Partial proof:")
        for step in prefix:
            prompt_lines.append(f"- {step.strip()}")

    # Tags and dependencies
    method_tags = ", ".join(entry.get("method_tags", []))
    if method_tags:
        prompt_lines.append(f"Proof techniques: {method_tags}")

    dependencies = ", ".join(entry.get("dependencies", []))
    if dependencies:
        prompt_lines.append(f"Dependencies: {dependencies}")

    # (Optional) GNN stats
    gnn_fields = []
    for stat in ['degree', 'centrality', 'pagerank']:
        if stat in entry:
            gnn_fields.append(f"{stat}: {entry[stat]}")
    if gnn_fields:
        prompt_lines.append(f"GNN stats | " + ", ".join(gnn_fields))

    return "\n".join(prompt_lines).strip()

def convert_dataset(
    input_path: str = "proof_tree_LLM/prooftree_impactful.jsonl",
    output_path: str = "proof_tree_LLM/llama_prompt_dataset.jsonl"
):
    """Converts a structured theorem dataset to prompt-target JSONL format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    assert input_path.exists(), f"File not found: {input_path}"

    print(f"Converting from {input_path} → {output_path}")

    converted = []
    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            data = json.loads(line)

            prompt = build_prompt(data)
            target = data.get("next_proof_step", data.get("target", "")).strip()

            if prompt and target:
                converted_sample = {
                    "prompt": prompt,
                    "target": target
                }
                outfile.write(json.dumps(converted_sample, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0:
                print(f"Processed {i+1} lines...")

    print(f"✅ Conversion complete. {i+1} samples written to: {output_path}")

if __name__ == "__main__":
    convert_dataset()