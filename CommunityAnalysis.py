import os
import networkx as nx
import community as community_louvain
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from itertools import combinations

# Function to calculate coverage and performance
def calculate_coverage_performance(G, communities):
    intra_edges, total_edges = 0, G.number_of_edges()
    for community in communities:
        intra_edges += sum(1 for _ in combinations(community, 2) if G.has_edge(*_))
    inter_edges = total_edges - intra_edges
    return intra_edges / total_edges, 1 - (inter_edges / total_edges)

# Function to add community to graph nodes
def add_community_to_graph(G, partition):
    for node, community_id in partition.items():
        G.nodes[node]['community'] = str(community_id)
    return G

# Function to save graph with communities to .net file
def save_graph_with_communities(G, path):
    nx.write_gexf(G, path)


def process_community_detection(algorithm_name, G, save_filename, mu):
    if algorithm_name == 'CNM':
        communities = list(greedy_modularity_communities(G))
        partition = {node: idx for idx, comm in enumerate(communities) for node in comm}
    elif algorithm_name == 'Louvain':
        partition = community_louvain.best_partition(G)
        communities = {}
        for node, community_id in partition.items():
            communities.setdefault(community_id, []).append(node)
        communities = list(communities.values())
    else:
        raise ValueError("Unsupported algorithm name")

    # Calculate metrics
    modularity = nx.algorithms.community.quality.modularity(G, communities)
    coverage, performance = calculate_coverage_performance(G, communities)
    
    # Add community info to graph
    G_with_community = add_community_to_graph(G.copy(), partition)
    
    # Save graph with community info
    save_graph_with_communities(G_with_community, os.path.join(network_dir, save_filename))
    
    return {
        'Modularity': modularity,
        'Coverage': coverage,
        'Performance': performance,
    }

# Function to evaluate community detection algorithms
def evaluate_community_detection(G, true_partitions, mu):
    results = {}
    node_to_community = {node: idx for idx, community in enumerate(true_partitions) for node in community}
    true_labels = [node_to_community[node] for node in G.nodes()]
    
    # Process CNM algorithm
    cnm_results = process_community_detection('CNM', G, f'network_cnm_mu_{mu}.gexf', mu)
    cnm_predicted_labels = [G.nodes[node]['community'] for node in G.nodes()]
    cnm_results['Rand Index'] = adjusted_rand_score(true_labels, cnm_predicted_labels)
    cnm_results['NMI'] = normalized_mutual_info_score(true_labels, cnm_predicted_labels)
    results['CNM'] = cnm_results
    
    # Process Louvain algorithm
    louvain_results = process_community_detection('Louvain', G, f'network_louvain_mu_{mu}.gexf', mu)
    louvain_predicted_labels = [G.nodes[node]['community'] for node in G.nodes()]
    louvain_results['Rand Index'] = adjusted_rand_score(true_labels, louvain_predicted_labels)
    louvain_results['NMI'] = normalized_mutual_info_score(true_labels, louvain_predicted_labels)
    results['Louvain'] = louvain_results
    
    # Save results to Markdown file
    save_results_to_file(results, mu)

def save_results_to_file(results, mu):
    with open('results.md', 'a') as f:
        f.write(f'\n## Mu = {mu}\n')
        f.write('| Algorithm | Modularity | Coverage | Performance | Rand Index | NMI |\n')
        f.write('|-----------|------------|----------|-------------|------------|-----|\n')
        for algo, metrics in results.items():
            f.write(f'| {algo} | {metrics["Modularity"]:.4f} | {metrics["Coverage"]:.4f} | {metrics["Performance"]:.4f} | {metrics["Rand Index"]:.4f} | {metrics["NMI"]:.4f} |\n')

if __name__ == "__main__":
    # Create Network directory if it doesn't exist
    network_dir = 'Network'
    if not os.path.exists(network_dir):
        os.makedirs(network_dir)

    # Main example of generating LFR benchmark graph and evaluating it
    mu_values = [0.1, 0.3, 0.5]  # Example mixing parameter values
    for mu in mu_values:
        G = nx.LFR_benchmark_graph(n=1000, tau1=3, tau2=1.5,average_degree = 5, min_degree=None, max_degree=None, min_community=20,max_community=30, mu=mu, seed=42)
        true_partitions = {frozenset(G.nodes[v]['community']) for v in G}
        print(f"Evaluating for mu={mu}")
        evaluate_community_detection(G, true_partitions, mu)
