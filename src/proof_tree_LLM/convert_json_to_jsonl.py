import json
import networkx as nx
import numpy as np
from collections import defaultdict

# Load the cleaned data JSON with UTF-8 encoding
def load_cleaned_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Process nodes (theorems) and edges (relationships)
def process_nodes_and_edges(data):
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    # Create an empty graph
    G = nx.DiGraph()  # Directed graph since relations are directional (DEPENDS_ON, EQUIVALENT_TO, etc.)
    
    # Add nodes to the graph with relevant features
    for node in nodes:
        node_id = node['id']
        features = node['features']
        G.add_node(node_id, **features)
        
        # Add theorem description as an additional attribute for reference
        theorem_info = {
            'statement': node['info']['statement'],
            'latex': node['info']['latex_logic'],
            'symbolic_logic': node['info']['symbolic_logic'],
            'first_order_logic': node['info']['first_order_logic'],
            'proof_methods': node['proof_methods'],
            'related_concepts': node['related_concepts']
        }
        G.nodes[node_id].update(theorem_info)
    
    # Add edges (relationships between theorems)
    for edge in edges:
        source = edge['source']
        target = edge['target']
        relation = edge['relation']
        weight = edge.get('weight', 1.0)
        
        G.add_edge(source, target, relation=relation, weight=weight)
    
    return G

# Generate tokenized input from the graph data (nodes and edges)
def generate_tokenized_input(graph):
    tokenized_data = []
    
    for node in graph.nodes(data=True):
        node_id, attributes = node
        statement = attributes['statement']
        
        # Tokenizing statement (simple whitespace split as an example)
        tokens = tokenize_expression(statement)  # Replace with a more advanced tokenization (e.g., LaTeX to tokens)
        
        # Create a tokenized entry for the node (theorem)
        tokenized_entry = {
            'theorem_id': node_id,
            'tokens': tokens,
            'proof_methods': attributes['proof_methods'],
            'related_concepts': attributes['related_concepts']
        }
        tokenized_data.append(tokenized_entry)
        
    return tokenized_data

# Advanced tokenization for special characters and LaTeX math expressions
def tokenize_expression(expression):
    """
    Tokenizes a mathematical expression (supports special characters and LaTeX).
    This can be extended to better handle LaTeX symbols and mathematical expressions.
    """
    # Example: Replace LaTeX-style symbols with tokens
    latex_map = {
        '\\cosh': 'cosh',
        '\\sinh': 'sinh',
        '\\cos': 'cos',
        '\\frac': 'frac',
        '\\sin': 'sin',
        '\\forall': 'forall',
        '\\exists': 'exists'
    }
    
    # Replace known LaTeX symbols with tokens
    for symbol, token in latex_map.items():
        expression = expression.replace(symbol, token)
    
    # Split the expression by non-alphanumeric characters
    tokens = expression.split()  # You can improve this with more sophisticated tokenizers
    
    # Handle special characters like Greek symbols or math functions
    # For example: "α" could be tokenized as "alpha"
    tokens = [token.replace("α", "alpha").replace("β", "beta") for token in tokens]
    
    return tokens

# Create a dictionary for storing tokenized relationships (edges)
def generate_tokenized_relationships(graph):
    tokenized_relationships = []
    
    for edge in graph.edges(data=True):
        source, target, attributes = edge
        relation = attributes['relation']
        
        # Tokenize the relation (just as an example here, it can be extended)
        tokenized_relationship = {
            'source': source,
            'target': target,
            'relation': relation
        }
        
        tokenized_relationships.append(tokenized_relationship)
    
    return tokenized_relationships

# Main function to execute the process
def main(json_file):
    # Load and process the data
    data = load_cleaned_data(json_file)
    
    # Process nodes and edges into a graph
    graph = process_nodes_and_edges(data)
    
    # Generate tokenized input from the graph nodes (theorems)
    tokenized_theorems = generate_tokenized_input(graph)
    
    # Generate tokenized relationships (edges)
    tokenized_relationships = generate_tokenized_relationships(graph)
    
    # Print some tokenized data for verification
    print("Tokenized Theorems:")
    for theorem in tokenized_theorems[:5]:  # Show first 5 tokenized theorems
        print(f"Theorem ID: {theorem['theorem_id']}, Tokens: {theorem['tokens']}")
    
    print("\nTokenized Relationships:")
    for relationship in tokenized_relationships[:5]:  # Show first 5 relationships
        print(f"Source: {relationship['source']}, Target: {relationship['target']}, Relation: {relationship['relation']}")

    # Optionally save tokenized data back to JSON with UTF-8 encoding
    with open("proof_tree_LLM/tokenized_data.json", 'w', encoding='utf-8') as outfile:
        json.dump({
            'theorems': tokenized_theorems,
            'relationships': tokenized_relationships
        }, outfile, ensure_ascii=False, indent=4)

# Run the main function with your cleaned data JSON file
if __name__ == "__main__":
    # Specify the path to your cleaned data JSON file
    json_file = 'GNN-GAT\cleaned_data.json'
    main(json_file)
