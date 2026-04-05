"""
SE(2) A*路径规划算法 - 快速优化版

优化策略:
1. 跳跃点搜索(JPS) - 跳过不必要的中间节点
2. 双向搜索 - 从起点和目标同时搜索
3. 碰撞检测缓存 - 避免重复计算
4. 运动基元优化 - 减少邻居扩展数量
5. 启发式加权 - 加速收敛

精度优化:
1. 高精度运动基元 - 增加转向角度分辨率
2. 自适应跳跃距离 - 根据环境动态调整
3. 路径插值 - 在跳跃点之间插入中间点
4. 精确碰撞检测 - 多采样点检测
"""

import numpy as np
import heapq
from typing import Tuple, List, Optional, Dict, Set
import sys
import os

# 添加se2模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, src_dir)

from C_space_pkg.se2 import (
    SE2ConfigurationSpace, RobotShape, create_rectangle_robot
)

# 导入基类
from A_star_base import (
    BaseSE2Planner, ReedsSheppSegment, SearchNode,
    Constants, PlannerConfig, normalize_angle, compute_distance
)


# ============== 快速SE(2) A*规划器 ==============

class FastSE2AStarPlanner(BaseSE2Planner):
    """快速SE(2) A*路径规划器 - 跳跃点搜索优化"""

    def __init__(self, c_space: SE2ConfigurationSpace,
                 robot: RobotShape,
                 min_radius: float = 1.5,
                 resolution: float = 1.0,
                 theta_resolution: int = 16):
        # 创建配置
        config = PlannerConfig(
            max_iterations=5000000,
            goal_tolerance=2.0,
            theta_tolerance=np.pi / 8,
            heuristic_weight=1.2,
            adaptive_jump=True,
            collision_samples=3,
            high_precision_mode=True,
            path_interpolation=True,
            verbose=True
        )
        
        super().__init__(c_space, robot, min_radius, resolution, theta_resolution, config)

    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float]) -> Optional[List[Tuple[float, float, float]]]:
        """
        规划路径 - 使用跳跃点搜索优化
        """
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
            cost=0.0,
            pose=start,
            g_cost=0.0,
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

            # 跳跃点搜索：尝试跳跃到更远的节点
            jump_success = False
            if current.direction is not None:
                jump_poses = self._jump(current.pose, current.direction, goal)
                if jump_poses:
                    for jump_pose, jump_cost in jump_poses:
                        jump_key = self._discretize_pose(jump_pose)
                        if jump_key not in closed_set and not self._check_collision_cached(jump_pose):
                            g_cost = current.g_cost + jump_cost
                            if jump_key not in visited or visited[jump_key] > g_cost:
                                visited[jump_key] = g_cost
                                h_cost = self._heuristic(jump_pose, goal)
                                jump_node = SearchNode(
                                    cost=g_cost + h_cost,
                                    pose=jump_pose,
                                    g_cost=g_cost,
                                    h_cost=h_cost,
                                    parent=current,
                                    direction=current.direction
                                )
                                heapq.heappush(open_list, jump_node)
                                jump_success = True

            # 如果跳跃失败，使用常规扩展
            if not jump_success:
                for neighbor_pose, motion, cost in self._get_neighbors(current.pose):
                    neighbor_key = self._discretize_pose(neighbor_pose)

                    if neighbor_key in closed_set:
                        continue

                    if self._check_collision_cached(neighbor_pose):
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
                        cost=g_cost + h_cost,
                        pose=neighbor_pose,
                        g_cost=g_cost,
                        h_cost=h_cost,
                        parent=current,
                        direction=direction
                    )

                    heapq.heappush(open_list, neighbor_node)

        if self.config.verbose:
            print(f"未找到路径，迭代次数: {iterations}")
        return None

    def _jump(self, pose: Tuple[float, float, float],
              direction: Tuple[float, float, float],
              goal: Tuple[float, float, float]) -> Optional[List[Tuple[Tuple[float, float, float], float]]]:
        """
        跳跃点搜索：沿方向跳跃多个步长（高精度版本）
        
        优化：
        1. 自适应跳跃距离 - 根据到目标的距离调整
        2. 多采样点碰撞检测 - 提高安全性
        3. 角度连续性检查 - 确保路径平滑
        
        返回: [(跳跃点位置, 代价), ...] 或 None
        """
        jump_poses = []
        x, y, theta = pose
        dx, dy, dtheta = direction
        
        # 归一化方向
        dist = np.sqrt(dx**2 + dy**2)
        if dist < Constants.MIN_DIRECTION_DIST:
            return None
        
        dx_norm = dx / dist
        dy_norm = dy / dist
        
        # 自适应跳跃距离：根据到目标的距离调整
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
        
        # 尝试不同跳跃距离
        for jump_mult in jump_multipliers:
            jump_dist = jump_mult * self.resolution
            new_x = x + dx_norm * jump_dist
            new_y = y + dy_norm * jump_dist
            new_theta = normalize_angle(theta + dtheta)
            
            # 高精度碰撞检测：多采样点
            valid = True
            num_samples = jump_mult * self.config.collision_samples
            for i in range(1, num_samples + 1):
                sample_dist = jump_dist * i / num_samples
                sample_x = x + dx_norm * sample_dist
                sample_y = y + dy_norm * sample_dist
                sample_theta = normalize_angle(theta + dtheta * (i / num_samples))
                
                if self._check_collision_cached((sample_x, sample_y, sample_theta)):
                    valid = False
                    break
            
            if valid and not self._check_collision_cached((new_x, new_y, new_theta)):
                new_dist = compute_distance((new_x, new_y, new_theta), goal)
                if new_dist < dist_to_goal:
                    jump_poses.append(((new_x, new_y, new_theta), jump_dist))
        
        return jump_poses if jump_poses else None

    def _reconstruct_path(self, node: SearchNode) -> List[Tuple[float, float, float]]:
        """重建路径（带插值优化）"""
        path = []
        current = node

        while current is not None:
            path.append(current.pose)
            current = current.parent

        path.reverse()
        
        # 路径插值：在跳跃点之间插入中间点
        if self.config.path_interpolation and len(path) > 1:
            path = self._interpolate_path(path)
        
        return path
    
    def _interpolate_path(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        路径插值：在相邻点之间插入中间点，提高路径平滑度
        
        Args:
            path: 原始路径
            
        Returns:
            插值后的路径
        """
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
    """双向SE(2) A*规划器 - 从起点和目标同时搜索"""

    def __init__(self, c_space: SE2ConfigurationSpace,
                 robot: RobotShape,
                 min_radius: float = 1.5,
                 resolution: float = 1.0,
                 theta_resolution: int = 16):
        # 创建配置
        config = PlannerConfig(
            max_iterations=5000000,
            goal_tolerance=2.0,
            theta_tolerance=np.pi / 8,
            heuristic_weight=1.0,
            verbose=True
        )
        
        super().__init__(c_space, robot, min_radius, resolution, theta_resolution, config)

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

        # 前向搜索
        forward_open = [SearchNode(
            cost=0.0, pose=start, g_cost=0.0,
            h_cost=self._heuristic(start, goal)
        )]
        forward_closed: Set[Tuple[int, int, int]] = set()
        forward_visited: Dict[Tuple[int, int, int], SearchNode] = {}

        # 反向搜索
        backward_open = [SearchNode(
            cost=0.0, pose=goal, g_cost=0.0,
            h_cost=self._heuristic(goal, start)
        )]
        backward_closed: Set[Tuple[int, int, int]] = set()
        backward_visited: Dict[Tuple[int, int, int], SearchNode] = {}

        iterations = 0
        while (forward_open or backward_open) and iterations < self.config.max_iterations:
            iterations += 1

            # 前向扩展
            if forward_open:
                current = heapq.heappop(forward_open)
                current_key = self._discretize_pose(current.pose)

                if current_key not in forward_closed:
                    forward_closed.add(current_key)
                    forward_visited[current_key] = current

                    # 检查是否与反向搜索相遇
                    if current_key in backward_visited:
                        if self.config.verbose:
                            print(f"双向搜索相遇! 迭代次数: {iterations}")
                        return self._reconstruct_bidirectional_path(
                            current, backward_visited[current_key]
                        )

                    # 扩展邻居
                    for neighbor_pose, motion, cost in self._get_neighbors(current.pose):
                        neighbor_key = self._discretize_pose(neighbor_pose)
                        if neighbor_key not in forward_closed and not self._check_collision_cached(neighbor_pose):
                            g_cost = current.g_cost + cost
                            h_cost = self._heuristic(neighbor_pose, goal)
                            neighbor_node = SearchNode(
                                cost=g_cost + h_cost,
                                pose=neighbor_pose,
                                g_cost=g_cost,
                                h_cost=h_cost,
                                parent=current
                            )
                            heapq.heappush(forward_open, neighbor_node)

            # 反向扩展
            if backward_open:
                current = heapq.heappop(backward_open)
                current_key = self._discretize_pose(current.pose)

                if current_key not in backward_closed:
                    backward_closed.add(current_key)
                    backward_visited[current_key] = current

                    # 检查是否与前向搜索相遇
                    if current_key in forward_visited:
                        if self.config.verbose:
                            print(f"双向搜索相遇! 迭代次数: {iterations}")
                        return self._reconstruct_bidirectional_path(
                            forward_visited[current_key], current
                        )

                    # 扩展邻居（反向）
                    for neighbor_pose, motion, cost in self._get_neighbors(current.pose):
                        neighbor_key = self._discretize_pose(neighbor_pose)
                        if neighbor_key not in backward_closed and not self._check_collision_cached(neighbor_pose):
                            g_cost = current.g_cost + cost
                            h_cost = self._heuristic(neighbor_pose, start)
                            neighbor_node = SearchNode(
                                cost=g_cost + h_cost,
                                pose=neighbor_pose,
                                g_cost=g_cost,
                                h_cost=h_cost,
                                parent=current
                            )
                            heapq.heappush(backward_open, neighbor_node)

        if self.config.verbose:
            print(f"未找到路径，迭代次数: {iterations}")
        return None

    def _reconstruct_bidirectional_path(self, forward_node: SearchNode,
                                        backward_node: SearchNode) -> List[Tuple[float, float, float]]:
        """重建双向路径"""
        # 前向路径
        forward_path = []
        current = forward_node
        while current is not None:
            forward_path.append(current.pose)
            current = current.parent
        forward_path.reverse()

        # 反向路径
        backward_path = []
        current = backward_node
        while current is not None:
            backward_path.append(current.pose)
            current = current.parent

        # 合并路径
        return forward_path + backward_path[1:]


# ============== 可视化工具 ==============

class FastPathVisualizer:
    """快速路径规划可视化工具"""

    @staticmethod
    def visualize_path(c_space: SE2ConfigurationSpace,
                       path: List[Tuple[float, float, float]],
                       start: Tuple[float, float, float],
                       goal: Tuple[float, float, float],
                       robot: RobotShape,
                       title: str = "Path Planning Result",
                       output_file: Optional[str] = None,
                       show_direction: bool = True):
        """
        可视化路径规划结果

        Args:
            c_space: 配置空间
            path: 路径点列表
            start: 起点位姿
            goal: 目标位姿
            robot: 机器人形状
            title: 图表标题
            output_file: 输出文件路径
            show_direction: 是否显示方向箭头
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制障碍物地图
        extent = [c_space.origin[0],
                  c_space.origin[0] + c_space.width * c_space.resolution,
                  c_space.origin[1],
                  c_space.origin[1] + c_space.height * c_space.resolution]

        ax.imshow(c_space.obstacle_map, cmap='gray_r', origin='lower',
                  extent=extent, alpha=0.6)

        # 绘制路径
        if path:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2.5, 
                   label='Path', zorder=3)

            # 绘制方向箭头
            if show_direction:
                arrow_interval = max(1, len(path) // 15)
                for i in range(0, len(path), arrow_interval):
                    x, y, theta = path[i]
                    dx = 1.5 * np.cos(theta)
                    dy = 1.5 * np.sin(theta)
                    ax.arrow(x, y, dx, dy, head_width=0.8, head_length=0.4,
                            fc='blue', ec='blue', alpha=0.7, zorder=4)

        # 绘制起点
        ax.plot(start[0], start[1], 'go', markersize=15, label='Start', zorder=5)
        robot_verts = robot.get_vertices_at_pose(start[0], start[1], start[2])
        robot_patch = Polygon(robot_verts, fill=True, facecolor='green',
                             edgecolor='darkgreen', linewidth=2, alpha=0.5)
        ax.add_patch(robot_patch)

        # 绘制目标
        ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal', zorder=5)
        robot_verts = robot.get_vertices_at_pose(goal[0], goal[1], goal[2])
        robot_patch = Polygon(robot_verts, fill=True, facecolor='red',
                             edgecolor='darkred', linewidth=2, alpha=0.5)
        ax.add_patch(robot_patch)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Path visualization saved to: {output_file}")

        plt.show()
        plt.close()

    @staticmethod
    def compare_paths(c_space: SE2ConfigurationSpace,
                      paths: Dict[str, List[Tuple[float, float, float]]],
                      start: Tuple[float, float, float],
                      goal: Tuple[float, float, float],
                      robot: RobotShape,
                      metrics: Optional[Dict[str, Dict[str, float]]] = None,
                      output_file: Optional[str] = None):
        """
        对比多个路径规划结果

        Args:
            c_space: 配置空间
            paths: 路径字典 {算法名称: 路径点列表}
            start: 起点位姿
            goal: 目标位姿
            robot: 机器人形状
            metrics: 性能指标字典
            output_file: 输出文件路径
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        n_paths = len(paths)
        fig, axes = plt.subplots(1, n_paths + 1, figsize=(6 * (n_paths + 1), 6))

        if n_paths == 0:
            print("No paths to compare!")
            return

        extent = [c_space.origin[0],
                  c_space.origin[0] + c_space.width * c_space.resolution,
                  c_space.origin[1],
                  c_space.origin[1] + c_space.height * c_space.resolution]

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        # 绘制各个路径
        for idx, (name, path) in enumerate(paths.items()):
            ax = axes[idx] if n_paths > 1 else axes
            ax.imshow(c_space.obstacle_map, cmap='gray_r', origin='lower',
                     extent=extent, alpha=0.5)

            if path:
                path_array = np.array(path)
                color = colors[idx % len(colors)]
                ax.plot(path_array[:, 0], path_array[:, 1], 
                       color=color, linewidth=2.5, label=name)

                # 方向箭头
                arrow_interval = max(1, len(path) // 10)
                for i in range(0, len(path), arrow_interval):
                    x, y, theta = path[i]
                    dx = 1.2 * np.cos(theta)
                    dy = 1.2 * np.sin(theta)
                    ax.arrow(x, y, dx, dy, head_width=0.6, head_length=0.3,
                            fc=color, ec=color, alpha=0.6)

            ax.plot(start[0], start[1], 'go', markersize=12, zorder=5)
            ax.plot(goal[0], goal[1], 'r*', markersize=15, zorder=5)

            title = f'{name}'
            if metrics and name in metrics:
                m = metrics[name]
                title += f"\nTime: {m.get('time', 0):.4f}s | Nodes: {m.get('nodes', 0)}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)

        # 绘制对比图
        ax_compare = axes[-1] if n_paths > 1 else axes
        ax_compare.imshow(c_space.obstacle_map, cmap='gray_r', origin='lower',
                         extent=extent, alpha=0.3)

        for idx, (name, path) in enumerate(paths.items()):
            if path:
                path_array = np.array(path)
                color = colors[idx % len(colors)]
                linestyle = '-' if idx == 0 else '--'
                ax_compare.plot(path_array[:, 0], path_array[:, 1],
                               color=color, linestyle=linestyle, 
                               linewidth=2.5, label=name, alpha=0.8)

        ax_compare.plot(start[0], start[1], 'go', markersize=12, zorder=5)
        ax_compare.plot(goal[0], goal[1], 'r*', markersize=15, zorder=5)

        ax_compare.set_title('Path Comparison', fontsize=12, fontweight='bold')
        ax_compare.set_xlabel('X (m)', fontsize=10)
        ax_compare.set_ylabel('Y (m)', fontsize=10)
        ax_compare.set_aspect('equal')
        ax_compare.grid(True, alpha=0.3)
        ax_compare.legend(loc='upper left', fontsize=10)

        plt.suptitle('Path Planning Algorithm Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Comparison saved to: {output_file}")

        plt.show()
        plt.close()

    @staticmethod
    def plot_performance_metrics(metrics: Dict[str, Dict[str, float]],
                                 output_file: Optional[str] = None):
        """
        绘制性能指标对比图

        Args:
            metrics: 性能指标字典 {算法名称: {指标名: 数值}}
            output_file: 输出文件路径
        """
        import matplotlib.pyplot as plt

        if not metrics:
            print("No metrics to plot!")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        algorithms = list(metrics.keys())
        colors = ['steelblue', 'coral', 'green', 'orange', 'purple']

        # 规划时间对比
        ax = axes[0]
        times = [metrics[alg].get('time', 0) for alg in algorithms]
        bars = ax.bar(algorithms, times, color=[colors[i % len(colors)] for i in range(len(algorithms))])
        ax.set_ylabel('Time (seconds)', fontsize=11)
        ax.set_title('Planning Time Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{time_val:.4f}', ha='center', va='bottom', fontsize=9)

        # 路径节点数对比
        ax = axes[1]
        nodes = [metrics[alg].get('nodes', 0) for alg in algorithms]
        bars = ax.bar(algorithms, nodes, color=[colors[i % len(colors)] for i in range(len(algorithms))])
        ax.set_ylabel('Number of Nodes', fontsize=11)
        ax.set_title('Path Nodes Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, node_val in zip(bars, nodes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{node_val}', ha='center', va='bottom', fontsize=9)

        # 加速比对比
        ax = axes[2]
        if len(algorithms) > 1:
            baseline_time = times[0] if times[0] > 0 else 1.0
            speedups = [baseline_time / t if t > 0 else 0 for t in times]
            bars = ax.bar(algorithms, speedups, color=[colors[i % len(colors)] for i in range(len(algorithms))])
            ax.set_ylabel('Speedup Factor', fontsize=11)
            ax.set_title('Speedup Comparison', fontsize=12, fontweight='bold')
            ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Baseline')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=9)
            for bar, speedup in zip(bars, speedups):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.suptitle('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Metrics plot saved to: {output_file}")

        plt.show()
        plt.close()


# ============== 测试 ==============

if __name__ == "__main__":
    import time
    from C_space_pkg.se2 import SE2ConfigurationSpace, create_rectangle_robot

    print("=" * 60)
    print("快速A*算法性能测试")
    print("=" * 60)

    # 创建测试地图
    map_size = 50
    obstacle_map = np.zeros((map_size, map_size), dtype=np.uint8)
    obstacle_map[20:25, 20:25] = 1
    obstacle_map[30:35, 30:35] = 1

    c_space = SE2ConfigurationSpace(obstacle_map, resolution=1.0)
    rover = create_rectangle_robot(length=1.5, width=1.0)

    start = (5.0, 5.0, 0.0)
    goal = (40.0, 40.0, np.pi/4)

    # 测试原始A*
    print("\n[测试1] 原始A*算法")
    from A_pkg.scripts.A_star import SE2AStarPlanner
    planner1 = SE2AStarPlanner(c_space, rover, min_radius=1.5, resolution=0.5, theta_resolution=32)
    planner1.goal_tolerance = 0.5
    
    t0 = time.time()
    path1 = planner1.plan(start, goal)
    time1 = time.time() - t0

    if path1:
        print(f"路径长度: {len(path1)}个节点")
    print(f"耗时: {time1:.4f}秒")

    # 测试跳跃A*
    print("\n[测试2] 跳跃A*算法")
    planner2 = FastSE2AStarPlanner(c_space, rover, min_radius=1.5, resolution=0.5, theta_resolution=32)
    planner2.goal_tolerance = 0.5
    
    t0 = time.time()
    path2 = planner2.plan(start, goal)
    time2 = time.time() - t0

    if path2:
        print(f"路径长度: {len(path2)}个节点")
    print(f"耗时: {time2:.4f}秒")

    # 测试双向A*
    print("\n[测试3] 双向A*算法")
    planner3 = BidirectionalSE2AStarPlanner(c_space, rover, min_radius=1.5, resolution=0.5, theta_resolution=32)
    planner3.goal_tolerance = 0.5
    
    t0 = time.time()
    path3 = planner3.plan(start, goal)
    time3 = time.time() - t0

    if path3:
        print(f"路径长度: {len(path3)}个节点")
    print(f"耗时: {time3:.4f}秒")

    # 性能对比
    print("\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)
    print(f"原始A*:  {time1:.4f}秒 (基准)")
    if time2 > 0:
        print(f"跳跃A*:  {time2:.4f}秒 (加速 {time1/time2:.2f}x)")
    if time3 > 0:
        print(f"双向A*:  {time3:.4f}秒 (加速 {time1/time3:.2f}x)")

    # 可视化结果
    print("\n" + "=" * 60)
    print("生成可视化结果...")
    print("=" * 60)

    # 收集路径和指标
    paths = {}
    metrics = {}

    if path1:
        paths['Original A*'] = path1
        metrics['Original A*'] = {
            'time': time1,
            'nodes': len(path1)
        }

    if path2:
        paths['Jump A*'] = path2
        metrics['Jump A*'] = {
            'time': time2,
            'nodes': len(path2)
        }

    if path3:
        paths['Bidirectional A*'] = path3
        metrics['Bidirectional A*'] = {
            'time': time3,
            'nodes': len(path3)
        }

    # 生成可视化
    if paths:
        # 1. 路径对比图
        FastPathVisualizer.compare_paths(
            c_space=c_space,
            paths=paths,
            start=start,
            goal=goal,
            robot=rover,
            metrics=metrics,
            output_file='fast_astar_comparison.png'
        )

        # 2. 性能指标图
        FastPathVisualizer.plot_performance_metrics(
            metrics=metrics,
            output_file='fast_astar_metrics.png'
        )

        # 3. 单个路径可视化（跳跃A*）
        if path2:
            FastPathVisualizer.visualize_path(
                c_space=c_space,
                path=path2,
                start=start,
                goal=goal,
                robot=rover,
                title='Jump A* Path Planning Result',
                output_file='fast_astar_jump_path.png',
                show_direction=True
            )

    print("\n✅ 可视化完成！")
    print("生成的文件:")
    print("  - fast_astar_comparison.png (路径对比)")
    print("  - fast_astar_metrics.png (性能指标)")
    print("  - fast_astar_jump_path.png (跳跃A*路径)")
