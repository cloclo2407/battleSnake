# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing
import copy
from collections import deque
import heapq


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    width = game_state['board']['width']
    height = game_state['board']['height']

    # Initialize grid: 0 = free, 1 = blocked
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for snake in game_state['board']['snakes']:
        for body in snake['body']:
            grid[body['y']][body['x']] = 1  

    moves = ["up", "down", "left", "right"]
    open_space_count_dict = {}
    best_move = "up" # Default move

    # Directions mapped to (dx, dy)
    directions = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0)
    }

    head_x = game_state['you']['body'][0]['x']
    head_y = game_state['you']['body'][0]['y']

    
    for move in moves:
        dx, dy = directions[move]
        start_x, start_y = head_x + dx, head_y + dy

        # Check bounds and collisions
        if not (0 <= start_x < width and 0 <= start_y < height):
            open_space_count_dict[move] = 0
            continue
        if grid[start_y][start_x] == 1:
            open_space_count_dict[move] = 0
            continue

        flooded_grid = copy.deepcopy(grid)
        queue = deque()
        open_space_count = 0

        queue.append((start_x, start_y))
        while queue:
            x, y = queue.popleft()
            if not (0 <= x < width and 0 <= y < height):
                continue
            if flooded_grid[y][x] != 0:
                continue

            open_space_count += 1
            flooded_grid[y][x] = -1  # mark visited

            for dx, dy in directions.values():
                queue.append((x + dx, y + dy))

        open_space_count_dict[move] = open_space_count

    print(open_space_count_dict)

    # Choose the move with the maximum open space count
    if open_space_count_dict:
        best_move = sorted(open_space_count_dict.items(), key=lambda item: item[1], reverse=True)[0][0]
    else:
        best_move = "up"

    
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"
    
    

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head[
            "x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # TODO: Step 1 - Prevent your Battlesnake from moving out of bounds
    # board_width = game_state['board']['width']
    # board_height = game_state['board']['height']

    # TODO: Step 2 - Prevent your Battlesnake from colliding with itself
    # my_body = game_state['you']['body']

    # TODO: Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    # opponents = game_state['board']['snakes']

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(
            f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Choose a random move from the safe ones
    next_move = random.choice(safe_moves)

    # TODO: Step 4 - Move towards food instead of random, to regain health and survive longer
    # food = game_state['board']['food']
    next_move = best_move
    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    width = len(grid[0])
    height = len(grid)

    open_set = []
    heapq.heappush(open_set, (0, start, [start]))  # (f_score, node, path_so_far)
    g_score = {start: 0}
    visited = set()

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while open_set:
        _, current, path = heapq.heappop(open_set)

        if current == goal:
            return path  # shortest path found

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)

            if not (0 <= nx < width and 0 <= ny < height):
                continue  # out of bounds
            if grid[ny][nx] != 0:
                continue  # blocked

            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor, path + [neighbor]))

    return None  # no path found




# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
