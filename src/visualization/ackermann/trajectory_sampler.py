"""
轨迹采样模块
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pydrake.trajectories import BsplineTrajectory

from ackermann_gcs_pkg.ackermann_data_structures import VehicleParams
from ackermann_gcs_pkg.flat_output_mapper import compute_flat_output_mapping


@dataclass
class TrajectoryData:
    """轨迹数据容器
    
    存储轨迹的所有采样数据
    """
    # 时间数据
    t_samples: np.ndarray  # 采样时间点 (N,)
    start_time: float  # 起始时间
    end_time: float  # 终止时间
    
    # 位置和姿态数据
    position: np.ndarray  # 位置 (2, N)
    heading: np.ndarray  # 航向角 (N,)
    
    # 速度和加速度数据
    velocity: np.ndarray  # 速度 (N,)
    acceleration: np.ndarray  # 加速度 (N,)
    
    # 曲率和转向角数据
    curvature: np.ndarray  # 曲率 (N,)
    steering_angle: np.ndarray  # 转向角 (N,)
    
    # 可选：路径长度
    path_length: Optional[np.ndarray] = None  # 累积路径长度 (N,)
    
    def validate(self) -> bool:
        """验证数据完整性
        
        Returns:
            bool: 数据是否完整
        """
        n = len(self.t_samples)
        
        checks = [
            self.position.shape == (2, n),
            self.heading.shape == (n,),
            self.velocity.shape == (n,),
            self.acceleration.shape == (n,),
            self.curvature.shape == (n,),
            self.steering_angle.shape == (n,),
        ]
        
        if self.path_length is not None:
            checks.append(self.path_length.shape == (n,))
        
        return all(checks)


class TrajectorySampler:
    """轨迹采样器
    
    负责对轨迹进行采样并计算相关数据
    """
    
    def __init__(self, num_samples: int = 200):
        """初始化采样器
        
        Args:
            num_samples: 采样点数
        """
        self.num_samples = num_samples
        self._cache = {}  # 缓存字典
    
    def sample(
        self,
        trajectory: BsplineTrajectory,
        vehicle_params: VehicleParams,
        use_cache: bool = True
    ) -> TrajectoryData:
        """采样轨迹
        
        Args:
            trajectory: B样条轨迹
            vehicle_params: 车辆参数
            use_cache: 是否使用缓存
            
        Returns:
            TrajectoryData: 轨迹数据
        """
        # 检查缓存
        cache_key = self._get_cache_key(trajectory, vehicle_params)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 计算平坦输出映射
        mapping = compute_flat_output_mapping(
            trajectory, vehicle_params, num_samples=self.num_samples
        )
        
        # 采样时间点
        t_samples = np.linspace(
            trajectory.start_time(),
            trajectory.end_time(),
            self.num_samples
        )
        
        # 计算路径长度
        position = mapping["position"]
        path_length = self._compute_path_length(position)
        
        # 处理数据形状（确保是1D数组）
        heading = mapping["heading"].flatten() if mapping["heading"].ndim > 1 else mapping["heading"]
        velocity = mapping["velocity"].flatten() if mapping["velocity"].ndim > 1 else mapping["velocity"]
        acceleration = mapping["acceleration"].flatten() if mapping["acceleration"].ndim > 1 else mapping["acceleration"]
        curvature = mapping["curvature"].flatten() if mapping["curvature"].ndim > 1 else mapping["curvature"]
        steering_angle = mapping["steering_angle"].flatten() if mapping["steering_angle"].ndim > 1 else mapping["steering_angle"]
        
        # 创建轨迹数据
        data = TrajectoryData(
            t_samples=t_samples,
            start_time=trajectory.start_time(),
            end_time=trajectory.end_time(),
            position=position,
            heading=heading,
            velocity=velocity,
            acceleration=acceleration,
            curvature=curvature,
            steering_angle=steering_angle,
            path_length=path_length
        )
        
        # 验证数据
        if not data.validate():
            raise ValueError("轨迹数据验证失败")
        
        # 存入缓存
        if use_cache:
            self._cache[cache_key] = data
        
        return data
    
    def _compute_path_length(self, position: np.ndarray) -> np.ndarray:
        """计算累积路径长度
        
        Args:
            position: 位置数组 (2, N)
            
        Returns:
            np.ndarray: 累积路径长度 (N,)
        """
        # 计算相邻点之间的距离
        diff = np.diff(position, axis=1)
        distances = np.sqrt(np.sum(diff**2, axis=0))
        
        # 累积求和
        path_length = np.zeros(position.shape[1])
        path_length[1:] = np.cumsum(distances)
        
        return path_length
    
    def _get_cache_key(
        self,
        trajectory: BsplineTrajectory,
        vehicle_params: VehicleParams
    ) -> Tuple:
        """生成缓存键
        
        Args:
            trajectory: 轨迹对象
            vehicle_params: 车辆参数
            
        Returns:
            Tuple: 缓存键
        """
        # 使用轨迹对象的id和参数作为缓存键
        return (
            id(trajectory),
            trajectory.start_time(),
            trajectory.end_time(),
            vehicle_params.wheelbase,
            self.num_samples
        )
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
    
    def sample_at_time(
        self,
        trajectory: BsplineTrajectory,
        vehicle_params: VehicleParams,
        t: float
    ) -> Dict[str, np.ndarray]:
        """在指定时间点采样
        
        Args:
            trajectory: 轨迹对象
            vehicle_params: 车辆参数
            t: 时间点
            
        Returns:
            Dict: 采样数据字典
        """
        # 使用flat_output_mapper计算单个时间点的数据
        mapping = compute_flat_output_mapping(
            trajectory, vehicle_params, num_samples=1
        )
        
        return {
            'position': mapping['position'][:, 0],
            'heading': mapping['heading'][0],
            'velocity': mapping['velocity'][0],
            'acceleration': mapping['acceleration'][0],
            'curvature': mapping['curvature'][0],
            'steering_angle': mapping['steering_angle'][0]
        }
