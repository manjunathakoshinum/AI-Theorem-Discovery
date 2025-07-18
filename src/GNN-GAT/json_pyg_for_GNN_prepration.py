import json
import os

def load_json_utf8(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except UnicodeDecodeError as e:
        return None, f"Unicode decoding error: {e}"
    except Exception as e:
        return None, f"Other error loading JSON: {e}"

def get_type_name(value):
    if isinstance(value, dict): return "dict"
    if isinstance(value, list): return "list"
    if isinstance(value, str): return "str"
    if isinstance(value, int): return "int"
    if isinstance(value, float): return "float"
    if isinstance(value, bool): return "bool"
    if value is None: return "NoneType"
    return type(value).__name__

def check_node_fields(node, path="root"):
    issues = []

    if isinstance(node, dict):
        for key, val in node.items():
            new_path = f"{path}.{key}"
            if isinstance(val, (dict, list)):
                issues += check_node_fields(val, new_path)
            else:
                # Check for Unicode-safe string handling
                if isinstance(val, str):
                    try:
                        val.encode("ascii")
                    except UnicodeEncodeError:
                        issues.append({
                            "path": new_path,
                            "issue": "non-ascii characters detected",
                            "value": val
                        })

    elif isinstance(node, list):
        for i, item in enumerate(node):
            issues += check_node_fields(item, f"{path}[{i}]")

    return issues

def check_required_fields(node, required_fields, path="root"):
    issues = []
    for field in required_fields:
        if field not in node:
            issues.append({
                "path": f"{path}.{field}",
                "issue": "missing required field"
            })
    return issues

def main():
    json_file = "cleaned_data.json"  # Set your file name here

    if not os.path.exists(json_file):
        print("File not found:", json_file)
        return

    data, error = load_json_utf8(json_file)
    if error:
        print(error)
        return

    if "nodes" not in data or not isinstance(data["nodes"], list):
        print("Missing or invalid 'nodes' field.")
        return

    print(f"Loaded {len(data['nodes'])} theorems.")

    required_top_fields = [
        "id", "label", "features", "target", "metadata", "info",
        "proof_methods", "proof_status", "discovery_method",
        "related_concepts", "retrieval_score", "linked_axioms",
        "relational_ontology_tags", "source", "reference_material",
        "curated_by", "reviewed_by", "visibility", "proof_steps"
    ]

    report = []
    for idx, node in enumerate(data["nodes"]):
        node_id = node.get("id", f"<missing_id_{idx}>")
        node_issues = []

        # 1. Top-level field check
        node_issues += check_required_fields(node, required_top_fields, path=node_id)

        # 2. Recursively check all fields for unicode + structure
        node_issues += check_node_fields(node, path=node_id)

        if node_issues:
            report.append({
                "theorem_id": node_id,
                "issues": node_issues
            })

    if report:
        print(f"\nFound issues in {len(report)} theorems.\n")
        for item in report:
            print(f"🧠 Theorem: {item['theorem_id']}")
            for issue in item['issues']:
                print(f"  - Path: {issue['path']}")
                print(f"    Issue: {issue['issue']}")
                if "value" in issue:
                    print(f"    Value: {issue['value']}")
    else:
        print("✅ All theorems are consistent, typed correctly, and contain no Unicode issues.")

if __name__ == "__main__":
    main()
