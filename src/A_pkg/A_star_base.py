"""
SE(2) A*路径规划算法 - 基类模块

提供公共基类、常量、配置和数据结构
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import sys
import os

# 添加se2模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, src_dir)

from C_space_pkg.se2 import (
    SE2ConfigurationSpace, RobotShape, create_rectangle_robot
)


# ============== 常量定义 ==============

class Constants:
    """算法常量"""
    # 角度
    PI = np.pi
    TWO_PI = 2 * np.pi
    
    # 跳跃距离配置
    JUMP_CLOSE = [1, 2]      # 距离 < 5m
    JUMP_MEDIUM = [2, 3]     # 距离 5-15m
    JUMP_FAR = [2, 3, 4]     # 距离 > 15m
    
    # 距离阈值
    DIST_CLOSE = 5.0
    DIST_MEDIUM = 15.0
    
    # 插值参数
    MAX_INTERPOLATION_POINTS = 5
    INTERPOLATION_THRESHOLD = 1.5
    
    # 代价权重
    STEER_COST_MULTIPLIER = 1.1
    
    # 最小距离阈值
    MIN_DIRECTION_DIST = 0.01


# ============== 配置类 ==============

@dataclass
class PlannerConfig:
    """规划器配置"""
    # 搜索参数
    max_iterations: int = 100000
    goal_tolerance: float = 2.0
    theta_tolerance: float = np.pi / 8
    
    # 启发式参数
    heuristic_weight: float = 1.2
    
    # 跳跃点搜索参数
    adaptive_jump: bool = True
    collision_samples: int = 3
    
    # 精度参数
    high_precision_mode: bool = True
    path_interpolation: bool = True
    
    # 调试参数
    verbose: bool = True


# ============== 数据结构 ==============

@dataclass
class ReedsSheppSegment:
    """Reeds-Shepp路径段"""
    steer: float      # 转向: -1(右转), 0(直行), 1(左转)
    gear: int         # 档位: 1(前进), -1(倒车)
    length: float     # 长度


@dataclass(order=True)
class SearchNode:
    """A*搜索节点"""
    cost: float
    pose: Tuple[float, float, float] = field(compare=False)
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    parent: Optional['SearchNode'] = field(compare=False, default=None)
    direction: Optional[Tuple[float, float, float]] = field(compare=False, default=None)


# ============== 工具函数 ==============

def normalize_angle(angle: float) -> float:
    """将角度归一化到[-π, π]"""
    while angle > Constants.PI:
        angle -= Constants.TWO_PI
    while angle < -Constants.PI:
        angle += Constants.TWO_PI
    return angle


def compute_distance(pose1: Tuple[float, float, float], 
                     pose2: Tuple[float, float, float]) -> float:
    """计算两个位姿之间的欧氏距离"""
    return np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)


# ============== 基类定义 ==============

class BaseSE2Planner(ABC):
    """SE(2)规划器基类"""
    
    def __init__(self, c_space: SE2ConfigurationSpace,
                 robot: RobotShape,
                 min_radius: float = 1.5,
                 resolution: float = 1.0,
                 theta_resolution: int = 16,
                 config: Optional[PlannerConfig] = None):
        self.c_space = c_space
        self.robot = robot
        self.min_radius = min_radius
        self.resolution = resolution
        self.theta_resolution = theta_resolution
        self.config = config or PlannerConfig()
        
        # 碰撞检测缓存
        self._collision_cache: Dict[Tuple[int, int, int], bool] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 运动基元
        self.motion_primitives = self._generate_motion_primitives()
    
    def _generate_motion_primitives(self) -> List[ReedsSheppSegment]:
        """生成运动基元"""
        primitives = []
        
        # 直行（前进和倒车）
        for gear in [1, -1]:
            primitives.append(ReedsSheppSegment(0, gear, self.resolution))
            if self.config.high_precision_mode:
                primitives.append(ReedsSheppSegment(0, gear, self.resolution * 0.5))
        
        # 转向运动
        arc_length = self.min_radius * (Constants.TWO_PI / self.theta_resolution)
        for steer in [-1, 1]:
            primitives.append(ReedsSheppSegment(steer, 1, arc_length))
            if self.config.high_precision_mode:
                primitives.append(ReedsSheppSegment(steer, 1, arc_length * 0.5))
            primitives.append(ReedsSheppSegment(steer, -1, arc_length * 0.5))
        
        return primitives
    
    def _apply_motion(self, x: float, y: float, theta: float,
                      motion: ReedsSheppSegment) -> Tuple[float, float, float]:
        """应用运动基元"""
        if motion.steer == 0:
            # 直行
            new_x = x + motion.gear * motion.length * np.cos(theta)
            new_y = y + motion.gear * motion.length * np.sin(theta)
            new_theta = theta
        else:
            # 转向
            delta_theta = motion.steer * motion.length / self.min_radius
            new_theta = theta + delta_theta
            
            # 计算圆心
            if motion.steer > 0:
                center_x = x - self.min_radius * np.sin(theta)
                center_y = y + self.min_radius * np.cos(theta)
            else:
                center_x = x + self.min_radius * np.sin(theta)
                center_y = y - self.min_radius * np.cos(theta)
            
            # 计算新位置
            angle_to_center = np.arctan2(y - center_y, x - center_x)
            new_angle = angle_to_center + delta_theta
            new_x = center_x + self.min_radius * np.cos(new_angle)
            new_y = center_y + self.min_radius * np.sin(new_angle)
        
        return new_x, new_y, normalize_angle(new_theta)
    
    def _discretize_pose(self, pose: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """离散化位姿"""
        x, y, theta = pose
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        gtheta = int((theta + Constants.PI) / (Constants.TWO_PI / self.theta_resolution)) % self.theta_resolution
        return (gx, gy, gtheta)
    
    def _check_collision_cached(self, pose: Tuple[float, float, float]) -> bool:
        """带缓存的碰撞检测"""
        key = self._discretize_pose(pose)
        
        if key in self._collision_cache:
            self._cache_hits += 1
            return self._collision_cache[key]
        
        self._cache_misses += 1
        result = self.c_space.check_collision(self.robot, pose[0], pose[1], pose[2])
        self._collision_cache[key] = result
        return result
    
    def _is_goal(self, pose: Tuple[float, float, float],
                 goal: Tuple[float, float, float]) -> bool:
        """检查是否到达目标"""
        dist = compute_distance(pose, goal)
        dtheta = abs(normalize_angle(pose[2] - goal[2]))
        return (dist < self.config.goal_tolerance and
                dtheta < self.config.theta_tolerance)
    
    def _heuristic(self, pose: Tuple[float, float, float],
                   goal: Tuple[float, float, float]) -> float:
        """启发式函数"""
        dist = compute_distance(pose, goal)
        dtheta = abs(normalize_angle(goal[2] - pose[2]))
        turn_cost = dtheta * self.min_radius
        return (dist + turn_cost) * self.config.heuristic_weight
    
    def _get_neighbors(self, pose: Tuple[float, float, float]) -> List[Tuple[Tuple[float, float, float], ReedsSheppSegment, float]]:
        """获取邻居节点"""
        neighbors = []
        x, y, theta = pose
        
        for motion in self.motion_primitives:
            new_x, new_y, new_theta = self._apply_motion(x, y, theta, motion)
            cost = motion.length
            if motion.steer != 0:
                cost *= Constants.STEER_COST_MULTIPLIER
            neighbors.append(((new_x, new_y, new_theta), motion, cost))
        
        return neighbors
    
    def _validate_inputs(self, start: Tuple[float, float, float],
                        goal: Tuple[float, float, float]) -> None:
        """验证输入参数"""
        # 检查起点和目标是否碰撞
        if self._check_collision_cached(start):
            raise ValueError("Start position is in collision")
        if self._check_collision_cached(goal):
            raise ValueError("Goal position is in collision")
    
    @abstractmethod
    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float]) -> Optional[List[Tuple[float, float, float]]]:
        """规划路径（抽象方法）"""
        pass
