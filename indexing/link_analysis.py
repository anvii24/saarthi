import json

def build_graph(verses, links):
    graph = {verse["id"]: set() for verse in verses}

    for link in links:
        source = link["from"]
        target = link["to"]

        graph[source].add(target)

    return graph


def load_graph(
    verses_path="data/verses.json",
    links_path="data/links.json"
):
    with open(verses_path, "r", encoding="utf-8") as file:
        verses = json.load(file)

    with open(links_path, "r", encoding="utf-8") as file:
        links = json.load(file)

    return build_graph(verses, links)

def pagerank(graph, damping=0.85, max_iterations=100, tolerance=1e-8):
    if not 0 <= damping < 1:
        raise ValueError("Damping must be between 0 and 1")

    if max_iterations <= 0:
        raise ValueError("Maximum iterations must be positive")

    if tolerance <= 0:
        raise ValueError("Tolerance must be positive")

    nodes = list(graph)
    n = len(nodes)

    if n == 0:
        return {}

    scores = {node: 1 / n for node in nodes}

    for _ in range(max_iterations):
        new_scores = {
            node: (1 - damping) / n
            for node in nodes
        }

        dangling_sum = sum(
            scores[node]
            for node in nodes
            if not graph[node]
        )

        for node in nodes:
            new_scores[node] += damping * dangling_sum / n

        for source in nodes:
            targets = graph[source]

            if targets:
                contribution = damping * scores[source] / len(targets)

                for target in targets:
                    new_scores[target] += contribution

        difference = sum(
            abs(new_scores[node] - scores[node])
            for node in nodes
        )

        scores = new_scores

        if difference < tolerance:
            break

    return scores


def topic_pagerank(
    graph,
    theme_verses,
    damping=0.85,
    max_iterations=100,
    tolerance=1e-8
):
    if not 0 <= damping < 1:
        raise ValueError("Damping must be between 0 and 1")

    if max_iterations <= 0:
        raise ValueError("Maximum iterations must be positive")

    if tolerance <= 0:
        raise ValueError("Tolerance must be positive")

    nodes = list(graph)
    n = len(nodes)

    if n == 0:
        return {}

    seeds = set(theme_verses) & set(nodes)

    if not seeds:
        return pagerank(
            graph,
            damping=damping,
            max_iterations=max_iterations,
            tolerance=tolerance
        )

    teleport = {
        node: 1 / len(seeds) if node in seeds else 0.0
        for node in nodes
    }

    scores = teleport.copy()

    for _ in range(max_iterations):
        new_scores = {
            node: (1 - damping) * teleport[node]
            for node in nodes
        }

        dangling_sum = sum(
            scores[node]
            for node in nodes
            if not graph[node]
        )

        for node in nodes:
            new_scores[node] += damping * dangling_sum * teleport[node]

        for source in nodes:
            targets = graph[source]

            if targets:
                contribution = damping * scores[source] / len(targets)

                for target in targets:
                    new_scores[target] += contribution

        difference = sum(
            abs(new_scores[node] - scores[node])
            for node in nodes
        )

        scores = new_scores

        if difference < tolerance:
            break

    return scores

def hits(graph, max_iterations=100, tolerance=1e-8):
    if max_iterations <= 0:
        raise ValueError("Maximum iterations must be positive")

    if tolerance <= 0:
        raise ValueError("Tolerance must be positive")

    nodes = list(graph)

    if not nodes:
        return {}, {}

    reverse_graph = {node: set() for node in nodes}

    for source, targets in graph.items():
        for target in targets:
            reverse_graph[target].add(source)

    hubs = {node: 1.0 for node in nodes}
    authorities = {node: 1.0 for node in nodes}

    for _ in range(max_iterations):
        new_authorities = {
            node: sum(hubs[source] for source in reverse_graph[node])
            for node in nodes
        }

        authority_norm = sum(
            score ** 2 for score in new_authorities.values()
        ) ** 0.5

        if authority_norm:
            new_authorities = {
                node: score / authority_norm
                for node, score in new_authorities.items()
            }

        new_hubs = {
            node: sum(new_authorities[target] for target in graph[node])
            for node in nodes
        }

        hub_norm = sum(
            score ** 2 for score in new_hubs.values()
        ) ** 0.5

        if hub_norm:
            new_hubs = {
                node: score / hub_norm
                for node, score in new_hubs.items()
            }

        difference = sum(
            abs(new_authorities[node] - authorities[node])
            + abs(new_hubs[node] - hubs[node])
            for node in nodes
        )

        authorities = new_authorities
        hubs = new_hubs

        if difference < tolerance:
            break

    return hubs, authorities