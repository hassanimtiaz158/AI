from collections import deque

def dfs(maze,start):
    rows=len(maze)
    cols=len(maze[0])
    queue=deque([(start,[start)]])
    visited=set()
    while(queue):
        (r,c),path=queue.popleft()
        if (r,c) in visited:
            continue
        visited.add((r,c))
        if maze[r][c] == 'E':
            return path
        
