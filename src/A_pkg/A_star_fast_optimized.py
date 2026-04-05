"""
SE(2) A*路径规划算法 - 优化精简版

优化改进:
1. 提取公共基类 - 减少重复代码
2. 使用配置类 - 简化参数管理
3. 定义常量类 - 消除魔法数字
4. 添加输入验证 - 提高健壮性
5. 优化数据结构 - 提高可读性
"""

import numpy as np
import heapq
from typing import Tuple, List, Optional, Dict, Set
import sys
import os

# 添加se2模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from C_space_pkg.se2 import (
    SE2ConfigurationSpace, RobotShape, create_rectangle_robot
)

# 导入基类（使用绝对导入）
from A_pkg.A_star_base import (
    BaseSE2Planner, ReedsSheppSegment, SearchNode,
    Constants, PlannerConfig, normalize_angle, compute_distance
)


# ============== 快速SE(2) A*规划器 ==============

class FastSE2AStarPlanner(BaseSE2Planner):
    """快速SE(2) A*路径规划器 - 跳跃点搜索优化"""
    
    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float]) -> Optional[List[Tuple[float, float, float]]]:
        """规划路径 - 使用跳跃点搜索优化"""
        # 清空缓存
        self._collision_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 验证输入
        try:
            self._validate_inputs(start, goal)
        except ValueError as e:
            if self.config.verbose:
                print(f"输入验证失败: {e}")
            return None
        
        # 初始化
        start_node = SearchNode(
            cost=0.0, pose=start, g_cost=0.0,
            h_cost=self._heuristic(start, goal)
        )
        
        open_list: List[SearchNode] = [start_node]
        closed_set: Set[Tuple[int, int, int]] = set()
        visited: Dict[Tuple[int, int, int], float] = {}
        
        iterations = 0
        while open_list and iterations < self.config.max_iterations:
            iterations += 1
            current = heapq.heappop(open_list)
            
            if self._is_goal(current.pose, goal):
                if self.config.verbose:
                    print(f"找到路径! 迭代次数: {iterations}")
                    total_checks = self._cache_hits + self._cache_misses
                    if total_checks > 0:
                        print(f"缓存命中率: {self._cache_hits/total_checks*100:.1f}%")
                return self._reconstruct_path(current)
            
            current_key = self._discretize_pose(current.pose)
            if current_key in closed_set:
                continue
            
            closed_set.add(current_key)
            
            # 跳跃点搜索
            jump_success = self._try_jump(current, goal, closed_set, visited, open_list)
            
            # 常规扩展
            if not jump_success:
                self._expand_neighbors(current, goal, closed_set, visited, open_list)
        
        if self.config.verbose:
            print(f"未找到路径，迭代次数: {iterations}")
        return None
    
    def _try_jump(self, current: SearchNode, goal: Tuple[float, float, float],
                  closed_set: Set, visited: Dict, open_list: List) -> bool:
        """尝试跳跃点搜索"""
        if current.direction is None:
            return False
        
        jump_poses = self._jump(current.pose, current.direction, goal)
        if not jump_poses:
            return False
        
        for jump_pose, jump_cost in jump_poses:
            jump_key = self._discretize_pose(jump_pose)
            if jump_key in closed_set or self._check_collision_cached(jump_pose):
                continue
            
            g_cost = current.g_cost + jump_cost
            if jump_key not in visited or visited[jump_key] > g_cost:
                visited[jump_key] = g_cost
                h_cost = self._heuristic(jump_pose, goal)
                jump_node = SearchNode(
                    cost=g_cost + h_cost, pose=jump_pose,
                    g_cost=g_cost, h_cost=h_cost,
                    parent=current, direction=current.direction
                )
                heapq.heappush(open_list, jump_node)
                return True
        
        return False
    
    def _expand_neighbors(self, current: SearchNode, goal: Tuple[float, float, float],
                         closed_set: Set, visited: Dict, open_list: List) -> None:
        """扩展邻居节点"""
        for neighbor_pose, motion, cost in self._get_neighbors(current.pose):
            neighbor_key = self._discretize_pose(neighbor_pose)
            
            if neighbor_key in closed_set or self._check_collision_cached(neighbor_pose):
                continue
            
            g_cost = current.g_cost + cost
            if neighbor_key in visited and visited[neighbor_key] <= g_cost:
                continue
            
            visited[neighbor_key] = g_cost
            h_cost = self._heuristic(neighbor_pose, goal)
            direction = (neighbor_pose[0] - current.pose[0],
                        neighbor_pose[1] - current.pose[1],
                        neighbor_pose[2] - current.pose[2])
            
            neighbor_node = SearchNode(
                cost=g_cost + h_cost, pose=neighbor_pose,
                g_cost=g_cost, h_cost=h_cost,
                parent=current, direction=direction
            )
            heapq.heappush(open_list, neighbor_node)
    
    def _jump(self, pose: Tuple[float, float, float],
              direction: Tuple[float, float, float],
              goal: Tuple[float, float, float]) -> Optional[List[Tuple[Tuple[float, float, float], float]]]:
        """跳跃点搜索"""
        x, y, theta = pose
        dx, dy, dtheta = direction
        
        # 归一化方向
        dist = np.sqrt(dx**2 + dy**2)
        if dist < Constants.MIN_DIRECTION_DIST:
            return None
        
        dx_norm, dy_norm = dx / dist, dy / dist
        
        # 自适应跳跃距离
        dist_to_goal = compute_distance(pose, goal)
        if self.config.adaptive_jump:
            if dist_to_goal < Constants.DIST_CLOSE:
                jump_multipliers = Constants.JUMP_CLOSE
            elif dist_to_goal < Constants.DIST_MEDIUM:
                jump_multipliers = Constants.JUMP_MEDIUM
            else:
                jump_multipliers = Constants.JUMP_FAR
        else:
            jump_multipliers = [2, 3, 5]
        
        jump_poses = []
        for jump_mult in jump_multipliers:
            jump_dist = jump_mult * self.resolution
            new_x = x + dx_norm * jump_dist
            new_y = y + dy_norm * jump_dist
            new_theta = normalize_angle(theta + dtheta)
            
            # 高精度碰撞检测
            if self._check_jump_path(x, y, theta, dx_norm, dy_norm, dtheta, 
                                    jump_dist, jump_mult):
                new_dist = compute_distance((new_x, new_y, new_theta), goal)
                if new_dist < dist_to_goal:
                    jump_poses.append(((new_x, new_y, new_theta), jump_dist))
        
        return jump_poses if jump_poses else None
    
    def _check_jump_path(self, x: float, y: float, theta: float,
                        dx_norm: float, dy_norm: float, dtheta: float,
                        jump_dist: float, jump_mult: int) -> bool:
        """检查跳跃路径是否可行"""
        num_samples = jump_mult * self.config.collision_samples
        for i in range(1, num_samples + 1):
            sample_dist = jump_dist * i / num_samples
            sample_x = x + dx_norm * sample_dist
            sample_y = y + dy_norm * sample_dist
            sample_theta = normalize_angle(theta + dtheta * (i / num_samples))
            
            if self._check_collision_cached((sample_x, sample_y, sample_theta)):
                return False
        return True
    
    def _reconstruct_path(self, node: SearchNode) -> List[Tuple[float, float, float]]:
        """重建路径"""
        path = []
        current = node
        while current is not None:
            path.append(current.pose)
            current = current.parent
        path.reverse()
        
        # 路径插值
        if self.config.path_interpolation and len(path) > 1:
            path = self._interpolate_path(path)
        
        return path
    
    def _interpolate_path(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """路径插值"""
        interpolated = [path[0]]
        
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            dist = compute_distance(p1, p2)
            
            if dist > self.resolution * Constants.INTERPOLATION_THRESHOLD:
                num_insertions = min(
                    int(dist / (self.resolution * 0.5)),
                    Constants.MAX_INTERPOLATION_POINTS
                )
                
                for j in range(1, num_insertions + 1):
                    t = j / (num_insertions + 1)
                    interp_x = p1[0] + (p2[0] - p1[0]) * t
                    interp_y = p1[1] + (p2[1] - p1[1]) * t
                    interp_theta = normalize_angle(p1[2] + (p2[2] - p1[2]) * t)
                    interpolated.append((interp_x, interp_y, interp_theta))
            
            interpolated.append(p2)
        
        return interpolated


# ============== 双向A*规划器 ==============

class BidirectionalSE2AStarPlanner(BaseSE2Planner):
    """双向SE(2) A*规划器"""
    
    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float]) -> Optional[List[Tuple[float, float, float]]]:
        """双向搜索"""
        self._collision_cache.clear()
        
        # 验证输入
        try:
            self._validate_inputs(start, goal)
        except ValueError as e:
            if self.config.verbose:
                print(f"输入验证失败: {e}")
            return None
        
        # 前向和反向搜索初始化
        forward_open = [SearchNode(0.0, start, 0.0, self._heuristic(start, goal))]
        backward_open = [SearchNode(0.0, goal, 0.0, self._heuristic(goal, start))]
        
        forward_closed: Set[Tuple[int, int, int]] = set()
        backward_closed: Set[Tuple[int, int, int]] = set()
        forward_visited: Dict[Tuple[int, int, int], SearchNode] = {}
        backward_visited: Dict[Tuple[int, int, int], SearchNode] = {}
        
        iterations = 0
        while (forward_open or backward_open) and iterations < self.config.max_iterations:
            iterations += 1
            
            # 前向扩展
            if forward_open:
                result = self._bidirectional_step(
                    forward_open, forward_closed, forward_visited,
                    backward_visited, start, goal, True
                )
                if result:
                    if self.config.verbose:
                        print(f"双向搜索相遇! 迭代次数: {iterations}")
                    return result
            
            # 反向扩展
            if backward_open:
                result = self._bidirectional_step(
                    backward_open, backward_closed, backward_visited,
                    forward_visited, goal, start, False
                )
                if result:
                    if self.config.verbose:
                        print(f"双向搜索相遇! 迭代次数: {iterations}")
                    return result
        
        if self.config.verbose:
            print(f"未找到路径，迭代次数: {iterations}")
        return None
    
    def _bidirectional_step(self, open_list: List, closed_set: Set,
                           visited: Dict, other_visited: Dict,
                           start: Tuple, goal: Tuple, is_forward: bool):
        """双向搜索单步"""
        current = heapq.heappop(open_list)
        current_key = self._discretize_pose(current.pose)
        
        if current_key in closed_set:
            return None
        
        closed_set.add(current_key)
        visited[current_key] = current
        
        # 检查是否相遇
        if current_key in other_visited:
            if is_forward:
                return self._merge_paths(current, other_visited[current_key])
            else:
                return self._merge_paths(other_visited[current_key], current)
        
        # 扩展邻居
        for neighbor_pose, _, cost in self._get_neighbors(current.pose):
            neighbor_key = self._discretize_pose(neighbor_pose)
            if neighbor_key not in closed_set and not self._check_collision_cached(neighbor_pose):
                g_cost = current.g_cost + cost
                h_cost = self._heuristic(neighbor_pose, goal)
                neighbor_node = SearchNode(
                    g_cost + h_cost, neighbor_pose, g_cost, h_cost, current
                )
                heapq.heappush(open_list, neighbor_node)
        
        return None
    
    def _merge_paths(self, forward_node: SearchNode, 
                     backward_node: SearchNode) -> List[Tuple[float, float, float]]:
        """合并双向路径"""
        # 前向路径
        forward_path = []
        current = forward_node
        while current:
            forward_path.append(current.pose)
            current = current.parent
        forward_path.reverse()
        
        # 反向路径
        backward_path = []
        current = backward_node
        while current:
            backward_path.append(current.pose)
            current = current.parent
        
        return forward_path + backward_path[1:]


# ============== 使用示例 ==============

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("优化版A*算法测试")
    print("=" * 60)
    
    # 创建测试地图
    map_size = 50
    obstacle_map = np.zeros((map_size, map_size), dtype=np.uint8)
    obstacle_map[20:25, 20:25] = 1
    obstacle_map[30:35, 30:35] = 1
    
    c_space = SE2ConfigurationSpace(obstacle_map, resolution=1.0)
    rover = create_rectangle_robot(length=1.5, width=1.0)
    
    # 创建配置
    config = PlannerConfig(
        max_iterations=100000,
        goal_tolerance=1.5,
        verbose=True
    )
    
    # 创建规划器
    planner = FastSE2AStarPlanner(c_space, rover, config=config)
    
    # 规划路径
    start = (5.0, 5.0, 0.0)
    goal = (40.0, 40.0, np.pi/4)
    
    t0 = time.time()
    path = planner.plan(start, goal)
    elapsed = time.time() - t0
    
    if path:
        print(f"\n路径节点数: {len(path)}")
        print(f"规划时间: {elapsed:.4f}s")
    else:
        print("\n未找到路径")
    
    print("\n✅ 测试完成！")

    
