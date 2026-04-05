"""
可视化基类模块
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import matplotlib.pyplot as plt
import numpy as np
from config.visualization import PlotConfig
from .output_manager import VisualizationOutputManager


class BaseVisualizer(ABC):
    """可视化器基类
    
    所有可视化器的抽象基类，定义通用接口和功能
    """
    
    def __init__(
        self,
        config: Optional[PlotConfig] = None,
        output_manager: Optional[VisualizationOutputManager] = None
    ):
        """初始化可视化器
        
        Args:
            config: 可视化配置对象，如果为None则使用默认配置
            output_manager: 输出管理器对象，如果为None则使用单例实例
        """
        self.config = config if config is not None else PlotConfig()
        self.config.validate()
        
        # 初始化输出管理器
        self.output_manager = (
            output_manager or VisualizationOutputManager.get_instance()
        )
        
        self._fig = None
        self._axes = None
    
    @abstractmethod
    def visualize(self, *args, **kwargs) -> Any:
        """可视化方法（抽象方法）
        
        子类必须实现此方法以提供具体的可视化功能
        
        Returns:
            Any: 可视化结果，通常是matplotlib的Figure对象
        """
        pass
    
    def setup_figure(self, nrows: int = 1, ncols: int = 1, **kwargs) -> tuple:
        """设置图形和坐标轴
        
        Args:
            nrows: 子图行数
            ncols: 子图列数
            **kwargs: 传递给plt.subplots的额外参数
            
        Returns:
            tuple: (figure, axes) 元组
        """
        figsize = kwargs.pop('figsize', self.config.figsize)
        dpi = kwargs.pop('dpi', self.config.dpi)
        
        self._fig, self._axes = plt.subplots(
            nrows=nrows, 
            ncols=ncols, 
            figsize=figsize, 
            dpi=dpi,
            **kwargs
        )
        
        return self._fig, self._axes
    
    def save_figure(self, filename: str, **kwargs) -> None:
        """保存图形到文件
        
        Args:
            filename: 文件名
            **kwargs: 传递给plt.savefig的额外参数
        """
        if self._fig is None:
            raise RuntimeError("图形尚未创建，请先调用setup_figure()")
        
        format = kwargs.pop('format', self.config.save_format)
        bbox_inches = kwargs.pop('bbox_inches', self.config.bbox_inches)
        dpi = kwargs.pop('dpi', self.config.dpi)
        
        self._fig.savefig(
            filename, 
            format=format, 
            bbox_inches=bbox_inches, 
            dpi=dpi,
            **kwargs
        )
    
    def show(self) -> None:
        """显示图形"""
        if self._fig is None:
            raise RuntimeError("图形尚未创建，请先调用setup_figure()")
        plt.show()
    
    def close(self) -> None:
        """关闭图形"""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None
    
    def set_title(self, title: str, ax: Optional[Any] = None, **kwargs) -> None:
        """设置标题
        
        Args:
            title: 标题文本
            ax: 坐标轴对象，如果为None则使用主坐标轴
            **kwargs: 传递给ax.set_title的额外参数
        """
        ax = ax if ax is not None else self._axes
        if ax is None:
            raise RuntimeError("坐标轴尚未创建")
        
        fontsize = kwargs.pop('fontsize', self.config.title_font_size)
        ax.set_title(title, fontsize=fontsize, **kwargs)
    
    def set_labels(self, xlabel: str, ylabel: str, ax: Optional[Any] = None, **kwargs) -> None:
        """设置坐标轴标签
        
        Args:
            xlabel: x轴标签
            ylabel: y轴标签
            ax: 坐标轴对象，如果为None则使用主坐标轴
            **kwargs: 传递给ax.set_xlabel/set_ylabel的额外参数
        """
        ax = ax if ax is not None else self._axes
        if ax is None:
            raise RuntimeError("坐标轴尚未创建")
        
        fontsize = kwargs.pop('fontsize', self.config.label_font_size)
        ax.set_xlabel(xlabel, fontsize=fontsize, **kwargs)
        ax.set_ylabel(ylabel, fontsize=fontsize, **kwargs)
    
    def apply_style(self) -> None:
        """应用全局样式设置"""
        plt.rcParams['font.family'] = self.config.font_family
        plt.rcParams['font.size'] = self.config.font_size
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动关闭图形"""
        self.close()
        return False
    
    def _get_output_path(
        self,
        filename: str,
        dimension: str,
        run_id: Optional[str] = None
    ) -> str:
        """获取输出路径（使用输出管理器）
        
        Args:
            filename: 文件名（包含扩展名）
            dimension: 维度标识（2d, 3d, 4d）
            run_id: 运行实例标识，可选
            
        Returns:
            str: 完整的规范化输出路径
            
        Example:
            >>> path = self._get_output_path(
            ...     filename="trajectory_2d.png",
            ...     dimension="2d",
            ...     run_id="experiment_001"
            ... )
        """
        return self.output_manager.generate_output_path(
            filename=filename,
            dimension=dimension,
            run_id=run_id
        )
