from collections import deque

def bfs(maze):
    rows = len(maze)
    cols = len(maze[0])
    
    # Locate Start (S) and Goal (G)
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 'S':
                start = (i, j)
            elif maze[i][j] == 'G':
                goal = (i, j)
    
    # BFS setup
    queue = deque([start])
    visited = set([start])
    parent = {start: None}
    
    directions = [(-1,0), (1,0), (0,-1), (0,1)]  # Up, Down, Left, Right
    
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if (0 <= nx < rows and 0 <= ny < cols and
                maze[nx][ny] in ['.', 'G'] and (nx, ny) not in visited):
                visited.add((nx, ny))
                parent[(nx, ny)] = current
                queue.append((nx, ny))
    
    # Reconstruct path
    path = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = parent[curr]
    path.reverse()
    
    # Mark the path
    steps = len(path)
    for x, y in path:
        if maze[x][y] != 'G':
            maze[x][y] = '*'
    
    return maze, steps

# Example maze
maze = [
    ['#', '#', '#', '#', '#', '#', '#'],
    ['#', 'S', '#', '.', '.', '.', '#'],
    ['#', '.', '#', '.', '#', 'G', '#'],
    ['#', '.', '.', '.', '.', '.', '#'],
    ['#', '#', '#', '#', '#', '#', '#']
]

solved_maze, total_steps = bfs(maze)

# Display result
for row in solved_maze:
    print(' '.join(row))
print("Total steps:", total_steps)
