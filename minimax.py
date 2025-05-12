import copy
import time
import random
from collections import deque

DIRECTIONS = ["up", "down", "left", "right"]
MOVE_DELTAS = {
    "up":    (0, 1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1, 0),
}

#Node class for Minimax
class BattleSnakeNode:
    def __init__(self, state, current_snake_id, depth, maximizing):
        self.state = state
        self.current_snake_id = current_snake_id
        self.depth = depth
        self.maximizing = maximizing

    # If maximizing, generate all possible moves for the snake
    # If minimizing, generate some random moves for the opponent snakes among possible ones
    def get_children(self):
        children = []

        if self.maximizing:
            # Your snake's turn: explore all moves
            for move in DIRECTIONS:
                new_state = simulate_move(self.state, self.current_snake_id, move) #Check if the move is valid
                if new_state:
                    child = BattleSnakeNode(new_state, self.current_snake_id, self.depth - 1, False)
                    children.append((move, child))
        else:
            # Opponent snakes' turn: treat them as one opponent player
            # Generate random moves for opponent snakes to reduce branching factor
            opponent_ids = [s["id"] for s in self.state["board"]["snakes"] if s["id"] != self.current_snake_id]
            for _ in range(3):  # Generate 3 random combinations of opponent moves
                move_set = {}
                for oid in opponent_ids:
                    valid_moves = get_valid_moves(self.state, oid) # Get valid moves for the opponent snake
                    if valid_moves:
                        move_set[oid] = random.choice(valid_moves)
                    else: 
                        move_set[oid] = random.choice(DIRECTIONS)  # If no valid moves, pick a random one

                new_state = simulate_multiple_moves(self.state, move_set) # New state with all opponent moves
                if new_state:
                    child = BattleSnakeNode(new_state, self.current_snake_id, self.depth - 1, True)
                    children.append((None, child))  # Move is not relevant for opponents

        return children

# Check if the move is valid
# A move is valid if it doesn't lead to a wall or a snake body
# Simulate the move and return the new state
# If the move is invalid, return None
def simulate_move(state, snake_id, move):
    # Find the snake to move
    snake = next((s for s in state["board"]["snakes"] if s["id"] == snake_id), None)
    if not snake or not snake["health"] > 0:
        return None  # Snake is dead or missing

    dx, dy = MOVE_DELTAS[move]
    head = snake["body"][0]
    new_head = {"x": head["x"] + dx, "y": head["y"] + dy}

    # Check wall bounds
    if not (0 <= new_head["x"] < state["board"]["width"] and 0 <= new_head["y"] < state["board"]["height"]):
        return None

    # Check collision with snake bodies
    for s in state["board"]["snakes"]:
        if new_head in s["body"]:
            return None

    # --- At this point, the move is valid ---

    # Now create a new state with the move applied
    new_state = copy_state(state)
    new_snake = next((s for s in new_state["board"]["snakes"] if s["id"] == snake_id), None)

    new_snake["body"].insert(0, new_head)

    if new_head in new_state["board"]["food"]:
        new_state["board"]["food"].remove(new_head)
        new_snake["health"] = 100
    else:
        new_snake["body"].pop()
        new_snake["health"] -= 1

    return new_state

# Get valid moves for a snake
# A move is valid if it doesn't lead to a wall or a snake body
def get_valid_moves(state, snake_id):
    valid = []
    for move in DIRECTIONS:
        if simulate_move(state, snake_id, move):
            valid.append(move)
    return valid

# Simulate multiple moves for all snakes
# Return the new state after all moves
def simulate_multiple_moves(state, move_set):
    new_state = copy_state(state)

    # Apply all moves (record intended positions)
    heads = {}
    for snake in new_state["board"]["snakes"]:
        if snake["id"] not in move_set:
            continue
        move = move_set[snake["id"]]
        dx, dy = MOVE_DELTAS[move]
        head = snake["body"][0]
        new_head = {"x": head["x"] + dx, "y": head["y"] + dy}
        heads[snake["id"]] = new_head

    # Collision detection
    board_width = new_state["board"]["width"]
    board_height = new_state["board"]["height"]
    occupied = {tuple(segment.values()) for s in new_state["board"]["snakes"] for segment in s["body"]}

    for snake in new_state["board"]["snakes"]:
        if snake["id"] not in heads:
            continue
        new_head = heads[snake["id"]]

        # Update body
        snake["body"].insert(0, new_head)
        if new_head in new_state["board"]["food"]:
            new_state["board"]["food"].remove(new_head)
            snake["health"] = 100
        else:
            snake["body"].pop()
            snake["health"] -= 1

        # Wall or body collision
        if not (0 <= new_head["x"] < board_width and 0 <= new_head["y"] < board_height):
            snake["health"] = 0  # Mark as dead
            continue
        if tuple(new_head.values()) in occupied:
            snake["health"] = 0
            continue

    # Remove dead snakes
    new_state["board"]["snakes"] = [s for s in new_state["board"]["snakes"] if s["health"] > 0]

    return new_state

def heuristic(state, my_snake_id):
    my_snake = next((s for s in state["board"]["snakes"] if s["id"] == my_snake_id), None)
    if not my_snake:
        return -1000  # You're dead

    score = 0
    head = my_snake["body"][0]

    # Favor staying alive
    score += 100

    # Favor longer length
    score += len(my_snake["body"]) * 2

    # Favor high health
    score += my_snake["health"]

    score += flood_fill(state, my_snake)

    if my_snake["health"] == 100:
        score += 30  # Bonus for eating food

    # Prefer being closer to food
    min_food_dist = float('inf')
    for food in state["board"]["food"]:
        dist = abs(food["x"] - head["x"]) + abs(food["y"] - head["y"])
        if dist < min_food_dist:
            min_food_dist = dist
    if min_food_dist < float('inf') :
        score += max(0, 20 - min_food_dist)


    # Penalize being near other snake heads
    for other in state["board"]["snakes"]:
        if other["id"] != my_snake_id:
            other_head = other["body"][0]
            dist = abs(head["x"] - other_head["x"]) + abs(head["y"] - other_head["y"])
            if dist == 2 & len(other["body"]) >= len(my_snake["body"]):
                score -= 20  
            elif dist == 1 & len(other["body"]) >= len(my_snake["body"]):
                score -= 1000 #Probably going to die
            elif dist <=2 & len(other["body"]) < len(my_snake["body"]): # Bonus if you can kill
                score += 10

    # Penalize going into a snake's body
    for other in state["board"]["snakes"]:
        for segment in other["body"]:
            dist = abs(head["x"] - segment["x"]) + abs(head["y"] - segment["y"])
            if dist <= 2:
                score -= 10   

    return score

def flood_fill(state, my_snake):
    width = state['board']['width']
    height = state['board']['height']

    # Initialize grid: 0 = free, 1 = blocked
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for snake in state['board']['snakes']:
        for body in snake['body']:
            grid[body['y']][body['x']] = 1  

    # Directions mapped to (dx, dy)
    directions = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0)
    }

    head_x = my_snake['body'][0]['x']
    head_y = my_snake['body'][0]['y']

    start_x, start_y = head_x, head_y

    # Check bounds and collisions
    if not (0 <= start_x < width and 0 <= start_y < height):
        return 0
    if grid[start_y][start_x] == 1:
        return 0

    flooded_grid = [row[:] for row in grid]
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

    return open_space_count

# Classical Minimax algorithm with alpha-beta pruning for two players
def minimax(node, alpha, beta, depth, start_time, my_snake_id, time_limit=0.045, memo=None):
    if time.time() - start_time > time_limit:
        return heuristic(node.state, my_snake_id), None

    state_key = hash_state(node.state, node.current_snake_id, node.depth, node.maximizing)
    if memo is not None and state_key in memo:
        return memo[state_key]

    if depth == 0:
        return heuristic(node.state, my_snake_id), None

    best_move = None

    if node.maximizing:
        max_eval = float('-inf')
        for move, child in node.get_children():
            eval, _ = minimax(child, alpha, beta, depth - 1, start_time, my_snake_id, time_limit, memo)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        result = (max_eval, best_move)
    else:
        min_eval = float('inf')
        for move, child in node.get_children():
            eval, _ = minimax(child, alpha, beta, depth - 1, start_time, my_snake_id, time_limit, memo)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        result = (min_eval, best_move)

    if memo is not None:
        memo[state_key] = result

    return result

# Main function to choose the best move
def choose_best_move(root, my_snake_id, time_limit=0.15):
    import time
    start_time = time.time()
    depth = 1
    best_move = None
    move = None

    while True:
        best_move_table = {}
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break

        score, move = minimax(
            root,
            alpha=float('-inf'),
            beta=float('inf'),
            depth=depth,
            start_time=start_time,
            my_snake_id=my_snake_id,
            time_limit=time_limit,
            memo=best_move_table
        )

        if move is not None:
            best_move = move

        depth += 1

    return best_move, depth - 1  # Return depth-1 as the last completed depth

# Hashing function for memoization
def hash_state(state, current_snake_id, depth, maximizing):
    snake_tuples = tuple((s['id'], tuple((seg['x'], seg['y']) for seg in s['body']), s['health']) for s in state['board']['snakes'])
    food_tuples = tuple((f['x'], f['y']) for f in state['board']['food'])
    return (snake_tuples, food_tuples, current_snake_id, depth, maximizing)

# Copy the state of the game (to avoid deep copying)
def copy_state(state):
    new_snakes = [dict(snake, body=snake["body"][:]) for snake in state["board"]["snakes"]]
    return {
        "board": {
            "width": state["board"]["width"],
            "height": state["board"]["height"],
            "food": list(state["board"]["food"]),
            "snakes": new_snakes
        }
    }
