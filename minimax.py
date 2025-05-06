import copy
import time
import random

DIRECTIONS = ["up", "down", "left", "right"]
MOVE_DELTAS = {
    "up":    (0, 1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1, 0),
}


class BattleSnakeNode:
    def __init__(self, state, current_snake_id, depth, maximizing):
        self.state = state
        self.current_snake_id = current_snake_id
        self.depth = depth
        self.maximizing = maximizing

    def get_children(self):
        children = []
        for move in DIRECTIONS:
            new_state = simulate_move(self.state, self.current_snake_id, move)
            if new_state:
                child = BattleSnakeNode(new_state, self.current_snake_id, self.depth - 1, not self.maximizing)
                children.append((move, child))
        return children


def simulate_move(state, snake_id, move):
    new_state = copy.deepcopy(state)
    snakes = new_state["board"]["snakes"]
    snake = next((s for s in snakes if s["id"] == snake_id), None)

    if not snake or not snake["health"] > 0:
        return None  # Snake is dead

    dx, dy = MOVE_DELTAS[move]
    head = snake["body"][0]
    new_head = {"x": head["x"] + dx, "y": head["y"] + dy}

    # Check wall collision
    if not (0 <= new_head["x"] < state["board"]["width"] and 0 <= new_head["y"] < state["board"]["height"]):
        return None

    # Check self collision or with any other snake body
    for s in new_state["board"]["snakes"]:
        if new_head in s["body"]:
            return None

    # Update snake body
    snake["body"].insert(0, new_head)

    if new_head in new_state["board"]["food"]:
        new_state["board"]["food"].remove(new_head)
        snake["health"] = 100  # Ate food
    else:
        snake["body"].pop()  # Move forward without growing
        snake["health"] -= 1  # Lose health

    return new_state


def heuristic(state, my_snake_id):
    my_snake = next((s for s in state["board"]["snakes"] if s["id"] == my_snake_id), None)
    if not my_snake:
        return -9999  # You're dead

    score = 0
    head = my_snake["body"][0]

    # Favor staying alive
    score += 100

    # Favor longer length
    score += len(my_snake["body"]) * 2

    # Favor high health
    score += my_snake["health"]
    
    if my_snake["health"] == 100:
        score += 50  # Bonus for being at max health

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
            if dist <= 2:
                score -= 10  
    
    # Penalize going into a snake's body
    for other in state["board"]["snakes"]:
        for segment in other["body"]:
            dist = abs(head["x"] - segment["x"]) + abs(head["y"] - segment["y"])
            if dist < 2:
                score -= 10   

    return score


def minimax(node, alpha, beta, depth, start_time, my_snake_id, time_limit=0.045, memo=None):
    if time.time() - start_time > time_limit:
        return heuristic(node.state, my_snake_id), None

    state_key = str(node.state) + str(node.current_snake_id) + str(node.depth) + str(node.maximizing)
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

