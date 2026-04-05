"""
控制点提取器模块

从轨迹对象中提取控制点坐标
"""

import numpy as np
from typing import Optional
import logging

from config.visualization import ControlPointData

# 配置日志
logger = logging.getLogger(__name__)


class ControlPointExtractor:
    """控制点提取器
    
    从PyDrake B样条轨迹对象中提取控制点坐标
    """
    
    def __init__(self, trajectory):
        """初始化提取器
        
        Args:
            trajectory: PyDrake轨迹对象（BsplineTrajectory）
            
        Raises:
            ValueError: 轨迹对象无效时抛出异常
        """
        self._trajectory = trajectory
        self._validate_trajectory()
    
    def extract(self) -> ControlPointData:
        """提取控制点
        
        Returns:
            ControlPointData: 控制点数据对象
            
        Raises:
            RuntimeError: 提取失败时抛出异常
        """
        try:
            # 提取B样条控制点
            points = self._extract_bspline_control_points()
            
            # 创建控制点数据对象
            control_point_data = ControlPointData.from_array(points)
            
            logger.info(f"成功提取 {control_point_data.num_points} 个控制点")
            return control_point_data
            
        except Exception as e:
            logger.error(f"控制点提取失败: {str(e)}")
            raise RuntimeError(f"控制点提取失败: {str(e)}") from e
    
    def _extract_bspline_control_points(self) -> np.ndarray:
        """从B样条轨迹提取控制点
        
        Returns:
            np.ndarray: 控制点坐标数组，形状为(N, 3)
        """
        # 检查是否是BezierTrajectory包装类
        if hasattr(self._trajectory, 'path_traj'):
            # 提取path_traj的控制点
            logger.info("检测到BezierTrajectory包装类，提取path_traj的控制点")
            trajectory = self._trajectory.path_traj
        else:
            trajectory = self._trajectory
        
        # 检查是否是CompositeTrajectory或PiecewiseTrajectory
        if hasattr(trajectory, 'get_number_of_segments'):
            # 分段轨迹，提取每段的控制点
            return self._extract_piecewise_control_points(trajectory)
        elif hasattr(trajectory, 'control_points'):
            # 单个B样条轨迹
            return self._extract_single_bspline_control_points(trajectory)
        else:
            raise ValueError("轨迹对象不支持控制点提取")
    
    def _extract_single_bspline_control_points(self, trajectory) -> np.ndarray:
        """从单个B样条轨迹提取控制点
        
        Args:
            trajectory: B样条轨迹对象
            
        Returns:
            np.ndarray: 控制点坐标数组，形状为(N, 3)
        """
        # 获取控制点
        control_points = trajectory.control_points()
        
        # 检查control_points的类型
        if isinstance(control_points, list):
            # control_points是list，每个元素是numpy数组
            points_list = []
            for point in control_points:
                # point是numpy数组，展平为一维
                point_array = np.asarray(point).flatten()
                points_list.append(point_array)
            
            # 堆叠为numpy数组
            points = np.vstack(points_list)
            
        elif hasattr(control_points, 'size'):
            # control_points是PyDrake对象，有size()方法
            num_control_points = control_points.size()
            points_list = []
            
            for i in range(num_control_points):
                # 获取第i个控制点
                point = control_points.at(i)
                # 转换为numpy数组并展平
                point_array = point.CopyToVector()
                points_list.append(point_array)
            
            # 堆叠为numpy数组
            points = np.vstack(points_list)
        else:
            raise ValueError(f"不支持的控制点类型: {type(control_points)}")
        
        return points
    
    def _extract_piecewise_control_points(self, trajectory) -> np.ndarray:
        """从分段轨迹提取控制点
        
        Args:
            trajectory: 分段轨迹对象
            
        Returns:
            np.ndarray: 控制点坐标数组，形状为(N, 3)
        """
        num_segments = trajectory.get_number_of_segments()
        all_points = []
        
        for i in range(num_segments):
            segment = trajectory.segment(i)
            
            # 检查段是否有control_points方法
            if hasattr(segment, 'control_points'):
                control_points = segment.control_points()
                num_control_points = control_points.size()
                
                for j in range(num_control_points):
                    point = control_points.at(j)
                    point_array = point.CopyToVector()
                    all_points.append(point_array)
            else:
                logger.warning(f"段 {i} 不支持控制点提取，跳过")
        
        if len(all_points) == 0:
            raise ValueError("分段轨迹中没有可提取的控制点")
        
        # 堆叠为numpy数组
        points = np.vstack(all_points)
        return points
    
    def _validate_trajectory(self) -> None:
        """验证轨迹对象
        
        Raises:
            ValueError: 轨迹对象无效时抛出异常
        """
        if self._trajectory is None:
            raise ValueError("轨迹对象不能为None")
        
        # 检查是否是BezierTrajectory包装类
        if hasattr(self._trajectory, 'path_traj'):
            # BezierTrajectory包装类，检查path_traj
            trajectory = self._trajectory.path_traj
        else:
            trajectory = self._trajectory
        
        # 检查是否支持控制点提取
        # 1. 单个B样条轨迹：有control_points方法
        # 2. 分段轨迹：有get_number_of_segments方法
        has_control_points = hasattr(trajectory, 'control_points')
        is_piecewise = hasattr(trajectory, 'get_number_of_segments')
        
        if not (has_control_points or is_piecewise):
            raise ValueError(
                "轨迹对象不支持控制点提取。"
                "需要control_points方法（B样条轨迹）或get_number_of_segments方法（分段轨迹）"
            )


def extract_control_points(trajectory) -> ControlPointData:
    """便捷函数：提取轨迹控制点
    
    Args:
        trajectory: PyDrake轨迹对象
        
    Returns:
        ControlPointData: 控制点数据对象
        
    Raises:
        ValueError: 轨迹对象无效时抛出异常
        RuntimeError: 提取失败时抛出异常
    """
    extractor = ControlPointExtractor(trajectory)
    return extractor.extract()
