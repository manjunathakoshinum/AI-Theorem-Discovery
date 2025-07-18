import json
import uuid
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import random
import math
from collections import defaultdict
import sys
from pathlib import Path
from datetime import datetime

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'mcts_theorem_analyzer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Data structures
@dataclass
class Theorem:
    id: str
    name: str
    domain: str
    importance_score: float
    proof_success_probability: float
    mathematical_depth: float
    latex_logic: str
    related_concepts: List[str]
    linked_axioms: List[str]
    features: Dict
    metadata: Dict
    info: Dict
    is_placeholder: bool = False  # Flag for auto-generated theorems

@dataclass
class Relationship:
    source: str
    target: str
    relation: str
    weight: float

@dataclass
class MCTSNode:
    theorem_id: str
    parent: Optional['MCTSNode']
    children: List['MCTSNode']
    visits: int
    value: float
    unexplored_relations: Set[Tuple[str, str]]  # (target_id, relation_type)
    depth: int = 0  # Track depth for normalization

class TheoremGraph:
    def __init__(self):
        self.theorems: Dict[str, Theorem] = {}
        self.relations: List[Relationship] = []
        self.adj_list: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

    def add_theorem(self, theorem: Theorem):
        self.theorems[theorem.id] = theorem

    def add_relation(self, relation: Relationship):
        self.relations.append(relation)
        self.adj_list[relation.source].append((relation.target, relation.relation, relation.weight))

    def get_related_theorems(self, theorem_id: str) -> List[Tuple[str, str, float]]:
        return self.adj_list.get(theorem_id, [])

    def create_placeholder_theorem(self, theorem_id: str) -> Theorem:
        """Create a placeholder theorem for missing IDs."""
        logger.warning(f"Creating placeholder theorem for ID: {theorem_id}")
        return Theorem(
            id=theorem_id,
            name=f"Placeholder_{theorem_id}",
            domain="Unknown",
            importance_score=0.5,
            proof_success_probability=0.5,
            mathematical_depth=0.5,
            latex_logic="",
            related_concepts=[],
            linked_axioms=[],
            features={},
            metadata={},
            info={},
            is_placeholder=True
        )

class MCTS:
    def __init__(self, graph: TheoremGraph, exploration_constant: float = 0.7, max_depth: int = 10):
        self.graph = graph
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.root: Optional[MCTSNode] = None
        self.total_rewards: List[float] = []  # For reward normalization

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select node using adaptive UCB1 with depth consideration."""
        while node.children and not node.unexplored_relations:
            node = max(node.children, key=lambda c: self._ucb1(c))
        return node

    def _ucb1(self, node: MCTSNode) -> float:
        """Calculate UCB1 score with depth-weighted exploration."""
        if node.visits == 0:
            return float('inf')
        parent_visits = node.parent.visits if node.parent else 1
        exploration = self.exploration_constant * math.sqrt(math.log(parent_visits) / node.visits)
        exploitation = (node.value / node.visits) if node.visits > 0 else 0
        depth_factor = 1 / (1 + node.depth * 0.1)  # Reduce exploration at deeper levels
        return exploitation + exploration * depth_factor

    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand node by adding a child from unexplored relations."""
        if not node.unexplored_relations:
            return None
        target_id, relation_type = node.unexplored_relations.pop()
        if target_id not in self.graph.theorems:
            logger.warning(f"Target theorem {target_id} not found during expansion")
            return None
        child = MCTSNode(
            theorem_id=target_id,
            parent=node,
            children=[],
            visits=0,
            value=0.0,
            unexplored_relations=self._get_unexplored_relations(target_id),
            depth=node.depth + 1
        )
        node.children.append(child)
        return child

    def _get_unexplored_relations(self, theorem_id: str) -> Set[Tuple[str, str]]:
        return {(target, rel) for target, rel, _ in self.graph.get_related_theorems(theorem_id)}

    def simulate(self, node: MCTSNode) -> float:
        """Simulate a rollout with enhanced reward function."""
        current_id = node.theorem_id
        depth = node.depth
        theorem = self.graph.theorems[current_id]
        
        # Enhanced reward: combine multiple features
        reward = (
            0.4 * theorem.importance_score +
            0.3 * theorem.features.get('novelty_score', 0.5) +
            0.2 * theorem.proof_success_probability +
            0.1 * theorem.mathematical_depth
        )
        visited = {current_id}

        while depth < self.max_depth:
            related = self.graph.get_related_theorems(current_id)
            if not related:
                break
            next_id, relation, weight = random.choice(related)
            if next_id not in self.graph.theorems or next_id in visited:
                break
            visited.add(next_id)
            theorem = self.graph.theorems[next_id]
            # Incremental reward
            inc_reward = (
                0.4 * theorem.importance_score +
                0.3 * theorem.features.get('novelty_score', 0.5) +
                0.2 * theorem.proof_success_probability +
                0.1 * theorem.mathematical_depth
            ) * weight
            if theorem.is_placeholder:
                inc_reward *= 0.8
            if relation == "CONTRADICTS":
                inc_reward *= 0.5
            reward += inc_reward
            current_id = next_id
            depth += 1

        # Normalize reward
        normalized_reward = reward / (depth + 1) if depth > 0 else reward
        self.total_rewards.append(normalized_reward)
        if len(self.total_rewards) > 1000:
            self.total_rewards.pop(0)
        avg_reward = sum(self.total_rewards) / len(self.total_rewards) if self.total_rewards else 1
        return normalized_reward / max(avg_reward, 1e-6)

    def backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate normalized reward."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def search(self, root_theorem_id: str, iterations: int = 2000) -> List[Tuple[str, float, int]]:
        """Run MCTS with enhanced iterations and result collection."""
        if root_theorem_id not in self.graph.theorems:
            logger.error(f"Root theorem {root_theorem_id} not found")
            return []

        self.root = MCTSNode(
            theorem_id=root_theorem_id,
            parent=None,
            children=[],
            visits=0,
            value=0.0,
            unexplored_relations=self._get_unexplored_relations(root_theorem_id),
            depth=0
        )

        for i in range(iterations):
            node = self.select(self.root)
            child = self.expand(node)
            reward = self.simulate(child if child else node)
            self.backpropagate(child if child else node, reward)
            if i % 500 == 0:
                logger.debug(f"Completed {i} iterations")

        # Collect results with visit counts
        results = []
        visited_nodes = set()

        def collect_results(node: MCTSNode):
            if node.theorem_id not in visited_nodes and node.visits > 0:
                avg_value = node.value / node.visits
                results.append((node.theorem_id, avg_value, node.visits))
                visited_nodes.add(node.theorem_id)
            for child in node.children:
                collect_results(child)

        collect_results(self.root)
        return sorted(results, key=lambda x: x[1], reverse=True)

def load_theorems(file_path: str, create_placeholders: bool = True) -> TheoremGraph:
    """Load theorems with advanced error handling and placeholder creation."""
    graph = TheoremGraph()
    missing_theorem_ids: Set[str] = set()
    problematic_fields: Dict[str, List[str]] = defaultdict(list)  # Track fields with issues
    processed_theorems = 0
    skipped_theorems = 0
    processed_edges = 0
    skipped_edges = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File {file_path} not found")
        return graph
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return graph
    except UnicodeDecodeError as e:
        logger.error(f"UTF-8 decoding error in {file_path}: {e}")
        return graph

    # Process nodes (theorems)
    for node in data.get('nodes', []):
        try:
            theorem_id = node.get('id', '')
            if not theorem_id:
                logger.warning("Skipping node with missing ID")
                skipped_theorems += 1
                continue

            features = node.get('features', {})
            metadata = node.get('metadata', {})
            info = node.get('info', {})

            # Validate and provide defaults for critical fields
            def get_float_field(field_name: str, default: float) -> float:
                try:
                    value = features.get(field_name)
                    if value is None:
                        problematic_fields[theorem_id].append(f"{field_name}: None")
                        return default
                    return float(value)
                except (TypeError, ValueError) as e:
                    problematic_fields[theorem_id].append(f"{field_name}: {str(e)}")
                    return default

            importance_score = get_float_field('importance_score', 0.5)
            proof_success_probability = get_float_field('proof_success_probability', 0.5)
            mathematical_depth = get_float_field('mathematical_depth', 0.5)

            # Handle other fields with defaults
            latex_logic = info.get('latex_logic', '')
            related_concepts = info.get('related_concepts', [])
            linked_axioms = info.get('linked_axioms', [])
            name = info.get('name', f"Unnamed_{theorem_id}")
            domain = metadata.get('domain', 'Unknown')

            theorem = Theorem(
                id=theorem_id,
                name=name,
                domain=domain,
                importance_score=importance_score,
                proof_success_probability=proof_success_probability,
                mathematical_depth=mathematical_depth,
                latex_logic=latex_logic,
                related_concepts=related_concepts,
                linked_axioms=linked_axioms,
                features=features,
                metadata=metadata,
                info=info
            )
            graph.add_theorem(theorem)
            processed_theorems += 1
        except Exception as e:
            logger.warning(f"Unexpected error processing theorem {node.get('id', 'unknown')}: {e}")
            skipped_theorems += 1

    # Process relationships (edges)
    for edge in data.get('edges', []):
        try:
            source = edge.get('source', '')
            target = edge.get('target', '')
            relation = edge.get('relation', '')
            weight = float(edge.get('weight', 1.0))

            if not source or not target or not relation:
                logger.warning(f"Skipping edge with missing fields: source={source}, target={target}, relation={relation}")
                skipped_edges += 1
                continue

            # Track missing theorem IDs
            if source not in graph.theorems:
                missing_theorem_ids.add(source)
                if create_placeholders:
                    graph.add_theorem(graph.create_placeholder_theorem(source))
            if target not in graph.theorems:
                missing_theorem_ids.add(target)
                if create_placeholders:
                    graph.add_theorem(graph.create_placeholder_theorem(target))

            # Add relation if both theorems exist
            if source in graph.theorems and target in graph.theorems:
                graph.add_relation(Relationship(source, target, relation, weight))
                processed_edges += 1
            else:
                logger.warning(f"Skipping edge with invalid source {source} or target {target}")
                skipped_edges += 1
        except (TypeError, ValueError) as e:
            logger.warning(f"Error processing edge {source} -> {target}: {e}")
            skipped_edges += 1

    # Log detailed summary
    logger.info(f"Processed {processed_theorems} theorems, skipped {skipped_theorems}")
    logger.info(f"Processed {processed_edges} relations, skipped {skipped_edges}")
    if missing_theorem_ids:
        logger.warning(f"Found {len(missing_theorem_ids)} missing theorem IDs: {', '.join(sorted(missing_theorem_ids))}")
        if create_placeholders:
            logger.info(f"Created placeholders for {len(missing_theorem_ids)} missing theorems")
    if problematic_fields:
        logger.warning("Theorems with problematic fields:")
        for theorem_id, issues in problematic_fields.items():
            logger.warning(f"  {theorem_id}: {'; '.join(issues)}")
    else:
        logger.info("No problematic fields found in theorems")

    return graph

def validate_json(file_path: str) -> Tuple[int, int, Set[str]]:
    """Validate theorems.json and return counts and missing IDs."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Cannot validate {file_path}: {e}")
        return 0, 0, set()

    theorem_ids = {node.get('id', '') for node in data.get('nodes', []) if node.get('id')}
    missing_ids = set()
    valid_edges = 0
    invalid_edges = 0

    for edge in data.get('edges', []):
        source = edge.get('source', '')
        target = edge.get('target', '')
        if source in theorem_ids and target in theorem_ids:
            valid_edges += 1
        else:
            invalid_edges += 1
            if source not in theorem_ids:
                missing_ids.add(source)
            if target not in theorem_ids:
                missing_ids.add(target)

    return valid_edges, invalid_edges, missing_ids

def main():
    file_path = 'theorems_cleaned.json'
    
    # Validate JSON before processing
    valid_edges, invalid_edges, missing_ids = validate_json(file_path)
    logger.info(f"Validation: {valid_edges} valid edges, {invalid_edges} invalid edges")
    if missing_ids:
        logger.warning(f"Validation found {len(missing_ids)} missing IDs (sample): {', '.join(sorted(list(missing_ids))[:10])}")

    # Load theorems with placeholders
    graph = load_theorems(file_path, create_placeholders=True)
    if not graph.theorems:
        logger.error("No theorems loaded. Exiting.")
        return

    # Initialize MCTS with optimized parameters
    mcts = MCTS(graph, exploration_constant=0.7, max_depth=10)

    # Choose starting theorem
    root_theorem_id = 'HYPERBOLIC_LAW_OF_COSINES'
    if root_theorem_id not in graph.theorems:
        logger.error(f"Root theorem {root_theorem_id} not found. Using first available theorem.")
        root_theorem_id = next(iter(graph.theorems.keys()), None)
        if not root_theorem_id:
            logger.error("No theorems available to start MCTS.")
            return

    # Run MCTS search
    logger.info(f"Running MCTS search starting from {root_theorem_id} with 2000 iterations")
    results = mcts.search(root_theorem_id, iterations=2000)

    # Output results
    print("\nMCTS Search Results (Theorem ID, Average Value, Visits):")
    for theorem_id, value, visits in results[:10]:  # Top 10
        theorem = graph.theorems.get(theorem_id)
        if theorem:
            placeholder_note = " (Placeholder)" if theorem.is_placeholder else ""
            print(f"- {theorem.name} ({theorem_id}): {value:.4f}, Visits: {visits}{placeholder_note} (Importance: {theorem.importance_score:.2f})")
        else:
            print(f"- {theorem_id}: {value:.4f}, Visits: {visits} (Theorem data missing)")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"Unexpected error in main: {e}", exc_info=True)
        sys.exit(1)