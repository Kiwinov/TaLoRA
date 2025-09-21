from collections import deque

edges = [[0, 1], [2, 3]]


def countComponents(edges) -> int:
    if not edges:
        return 0
    adj = {}
    for u, v in edges:
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append(v)
        adj[v].append(u)

    def dfs(node, visited):
        for i in adj[node]:
            if i not in visited:
                visited.add(i)
                dfs(i, visited)

    cnt = 0
    visited = set()
    for i in adj:
        if i not in visited:
            cnt += 1
            dfs(i, visited)
    return cnt


print(countComponents(edges))
