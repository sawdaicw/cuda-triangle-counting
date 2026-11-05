"""Small-graph oracle and oriented-graph reference; not a GPU benchmark."""
import argparse
import gzip
import itertools
import json


def normalize(edges, vertices=()):
    nodes = set(vertices)
    clean = set()
    for u, v in edges:
        nodes.update((u, v))
        if u != v:
            clean.add((min(u, v), max(u, v)))
    return sorted(nodes), clean


def count_bruteforce(edges, vertices=()):
    nodes, clean = normalize(edges, vertices)
    return sum((a, b) in clean and (a, c) in clean and (b, c) in clean
               for a, b, c in itertools.combinations(nodes, 3))


def forward_graph(edges, vertices=()):
    nodes, clean = normalize(edges, vertices)
    degree = dict.fromkeys(nodes, 0)
    for u, v in clean:
        degree[u] += 1
        degree[v] += 1
    forward = {u: set() for u in nodes}
    for u, v in clean:
        if (degree[u], u) > (degree[v], v):
            u, v = v, u
        forward[u].add(v)
    return forward


def count_forward(edges, vertices=()):
    forward = forward_graph(edges, vertices)
    return sum(len(forward[u] & forward[v])
               for u in forward for v in forward[u])


def read_edges(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith(('#', '%')):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f'line {line_number}: expected two integer endpoints')
            yield int(fields[0]), int(fields[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='two-column text edge list, optionally .gz')
    parser.add_argument('--oracle', action='store_true', help='also run small-graph enumeration')
    args = parser.parse_args()
    nodes, edges = normalize(read_edges(args.input))
    output = {'vertices': len(nodes), 'undirected_edges': len(edges),
              'triangles': count_forward(edges)}
    if args.oracle:
        if len(nodes) > 100:
            parser.error('--oracle is restricted to at most 100 vertices')
        output['oracle'] = count_bruteforce(edges)
        if output['oracle'] != output['triangles']:
            raise RuntimeError('oracle mismatch')
    print(json.dumps(output, ensure_ascii=False))
