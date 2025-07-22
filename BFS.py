from collections import deque

def bfs(maze,start,goal):
    rows=len(maze)
    cols=len(maze[0])
    queue=deque([start])
    visited=set()
    directions=[(-1,0),(1,0),(0,1),(0,-1)]
    while(queue):
        node=queue.popleft()
        if node not in visited:
            print(node)
            visited.add(node)
        for dr,dc in directions:
            new_r=row
