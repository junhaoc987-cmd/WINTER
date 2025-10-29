# JSIC WinterHack 2026 - Frontier Selection Technical Explanation

## Overview

This document describes the frontier selection strategy implemented for autonomous robot navigation in unknown maze environments. The solution enables efficient exploration while maintaining goal-directed behavior.

## Core Algorithm: Heuristic Cost-Based Selection

The implemented strategy uses a **cost minimization approach** inspired by A* pathfinding algorithm, selecting frontiers that minimize the total estimated cost to reach the final goal.

### Cost Function

For each candidate frontier, the total cost is computed as:

```
total_cost = g_cost + h_cost + direction_penalty
```

Where:
- **g_cost**: Actual BFS distance from robot to frontier (known cost)
- **h_cost**: Euclidean distance from frontier to goal (heuristic estimate)
- **direction_penalty**: Angular deviation penalty from optimal direction

## Three Key Components

### 1. Actual Path Cost (g_cost)

```python
g_cost = frontier_distances.get(candidate, float('inf'))
```

**Purpose**: Represents the real traversable distance from the robot's current position to the frontier cell.

**Why BFS Distance**: 
- Unlike Euclidean distance, BFS distance accounts for obstacles and walls
- Provides the actual number of grid cells the robot must traverse
- More realistic cost estimation in maze environments

### 2. Heuristic Estimate (h_cost)

```python
h_cost = math.hypot(cx - tx, cy - ty)
```

**Purpose**: Estimates the remaining distance from the frontier to the final goal.

**Rationale**:
- Euclidean distance provides an admissible heuristic (never overestimates)
- Guides exploration toward the goal region
- Similar to A* algorithm's heuristic function

### 3. Direction Deviation Penalty

```python
cos_angle = (vec_to_target · vec_to_candidate) / (||vec_to_target|| × ||vec_to_candidate||)
direction_penalty = (1.0 - cos_angle) × 0.5
```

**Purpose**: Penalizes frontiers that deviate from the direct path toward the goal.

**How it Works**:
- Computes the angle between "robot→target" and "robot→candidate" vectors
- Uses cosine similarity: 1.0 = aligned, 0.0 = perpendicular, -1.0 = opposite
- Penalty ranges from 0 (perfect alignment) to 1.0 (opposite direction)
- Weight of 0.5 ensures direction is important but not dominant

## Decision Process

### Step 1: Frontier Detection
The system calls `detect_frontiers()` which uses BFS to find all reachable frontier cells (free cells adjacent to unknown regions).

### Step 2: Cost Evaluation
For each frontier candidate:
1. Retrieve actual BFS distance (g_cost)
2. Calculate straight-line distance to goal (h_cost)
3. Compute angular deviation penalty
4. Sum to get total cost

### Step 3: Special Cases

**Goal Cell Bonus**:
```python
if candidate == target_cell:
    total_cost -= 100.0
```
If the goal cell itself becomes a frontier (i.e., becomes reachable), it receives a massive cost reduction, ensuring immediate selection.

**Fallback Strategy**:
If no frontier can be evaluated (all have infinite g_cost), select the one with minimum BFS distance as a safe fallback.

### Step 4: Path Planning and Validation

After selecting the best frontier, the system:

1. **First Priority**: Try to plan a direct path to the final goal
   - If successful, navigate directly to goal (skip frontier exploration)
   
2. **Second Priority**: Plan path to selected frontier
   - Execute if direct goal path is not possible
   
3. **Fallback**: If selected frontier is unreachable
   - Try up to 5 nearest frontiers (by BFS distance)
   - Select first reachable alternative

## Key Advantages

### 1. Goal-Directed Exploration
- The h_cost component ensures exploration trends toward the goal
- Not just exploring blindly, but strategically moving closer to target

### 2. Realistic Path Costs
- Using BFS distances accounts for actual maze structure
- Avoids selecting frontiers that appear close but require long detours

### 3. Balanced Trade-offs
- g_cost prevents choosing distant frontiers
- h_cost ensures goal-oriented progress
- direction_penalty reduces unnecessary turns and zigzagging

### 4. Robust Fallback Mechanisms
- Multiple validation layers ensure a path is always found
- Gracefully handles edge cases (no frontiers, unreachable cells)

## Design Philosophy

### Why Not Weighted Scoring?

Traditional approaches use normalized scores (0-1) with weighted averaging:
```
score = w1×factor1 + w2×factor2 + w3×factor3
```

**Our approach differs**:
- Uses absolute costs in meaningful units (grid cells, meters)
- No need for normalization or weight tuning
- More intuitive: "minimize total path length"
- Similar to proven algorithms (A*, Dijkstra)

### Exploration vs Exploitation

The algorithm balances:
- **Exploration**: Frontiers far from robot may be selected if they're much closer to goal
- **Exploitation**: Nearby frontiers are preferred when goal distance is similar
- **Direction**: Slight preference for frontiers in the goal direction

## Performance Characteristics

### Time Complexity
- O(N) where N = number of frontiers (typically small, < 50)
- Each frontier requires constant-time cost calculation

### Space Complexity
- O(1) additional space
- Only stores current best frontier and cost

### Behavior in Different Scenarios

**Random Maze**:
- Efficiently navigates around obstacles
- Selects frontiers that maintain progress toward goal
- Adapts to discovered openings

**Snake Maze**:
- Follows corridor structure naturally
- The direction penalty helps maintain forward momentum
- Low oscillation between frontiers

## Implementation Details

### Numerical Stability
- All divisions check for zero denominators (threshold: 1e-6)
- Use of `float('inf')` for unreachable frontiers
- Proper handling of edge cases (no frontiers, goal already reached)

### Exception Handling
- Path planning wrapped in try-except blocks
- Graceful degradation if primary frontier unreachable
- Maintains robot operation even in unexpected states

## Conclusion

This frontier selection strategy provides a **simple yet effective** approach to unknown environment exploration. By framing the problem as cost minimization rather than score maximization, and leveraging heuristic search principles, the robot achieves:

- Efficient goal-directed exploration
- Robust operation in complex mazes
- Predictable and stable behavior

The implementation is self-contained, computationally efficient, and requires no parameter tuning, making it suitable for real-time robotic applications.

---

**Author's Note**: This implementation draws inspiration from classical pathfinding algorithms (A*) and applies those principles to the frontier selection problem, resulting in a principled and effective navigation strategy.

