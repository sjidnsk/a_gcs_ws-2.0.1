"""
可视化输出管理器核心类

本模块实现核心输出管理器，采用单例模式，提供统一的输出路径管理服务。
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import ClassVar, Dict, List, Optional, Tuple

from .exceptions import (
    DirectoryCreationError,
    InvalidDimensionError,
    InvalidRunIdError
)
from .models import OutputConfig, OutputFileInfo, RunInstanceInfo
from .path_builder import PathBuilder


# 配置日志
logger = logging.getLogger(__name__)


class VisualizationOutputManager:
    """可视化输出管理器
    
    采用单例模式，统一管理可视化模块的输出路径。
    提供路径生成、目录创建、文件查询等核心功能。
    
    Attributes:
        config: 输出配置对象
        _current_run_id: 当前运行实例标识
        _run_directories: 已创建的运行目录缓存
    """
    
    # 单例实例
    _instance: ClassVar[Optional['VisualizationOutputManager']] = None
    
    # 线程锁
    _lock: ClassVar[Lock] = Lock()
    
    def __init__(self, config: Optional[OutputConfig] = None):
        """初始化输出管理器
        
        Args:
            config: 输出配置对象，可选
            
        Note:
            此方法不应直接调用，请使用 get_instance() 方法。
        """
        self.config = config or OutputConfig()
        self.config.validate()
        
        self._current_run_id: Optional[str] = None
        self._run_directories: Dict[str, str] = {}
        
        logger.debug(
            f"输出管理器初始化完成: "
            f"output_root={self.config.output_root}, "
            f"default_dimension={self.config.default_dimension}"
        )
    
    @classmethod
    def get_instance(
        cls,
        config: Optional[OutputConfig] = None
    ) -> 'VisualizationOutputManager':
        """获取单例实例
        
        线程安全的单例模式实现。首次调用时初始化配置。
        
        Args:
            config: 输出配置对象，仅在首次调用时使用
            
        Returns:
            VisualizationOutputManager: 单例实例
            
        Raises:
            ValueError: 配置校验失败时抛出
            
        Example:
            >>> manager = VisualizationOutputManager.get_instance()
            >>> # 或使用自定义配置
            >>> config = OutputConfig(output_root="./my_output")
            >>> manager = VisualizationOutputManager.get_instance(config)
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
                    logger.info("输出管理器单例实例已创建")
        
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """重置单例实例
        
        用于测试或重新初始化。
        """
        with cls._lock:
            cls._instance = None
            logger.debug("输出管理器单例实例已重置")
    
    def generate_output_path(
        self,
        filename: str,
        dimension: Optional[str] = None,
        run_id: Optional[str] = None,
        auto_create_dir: bool = True
    ) -> str:
        """生成规范化输出路径
        
        按照 {output_root}/{dimension}/{run_id}/{filename} 的结构生成路径。
        
        Args:
            filename: 文件名（包含扩展名）
            dimension: 维度标识，可选，默认使用配置中的default_dimension
            run_id: 运行实例标识，可选，默认使用当前运行实例或自动生成时间戳
            auto_create_dir: 是否自动创建目录，默认True
            
        Returns:
            str: 完整的规范化输出路径
            
        Raises:
            InvalidDimensionError: 维度标识非法时抛出
            InvalidRunIdError: 运行实例标识非法时抛出
            DirectoryCreationError: 目录创建失败时抛出
            
        Example:
            >>> path = manager.generate_output_path(
            ...     filename="trajectory_2d.png",
            ...     dimension="2d",
            ...     run_id="experiment_001"
            ... )
            >>> # 返回: "./output/2d/experiment_001/trajectory_2d.png"
        """
        # 使用默认维度
        if dimension is None:
            dimension = self.config.default_dimension
        
        # 校验维度
        if dimension not in self.config.VALID_DIMENSIONS:
            raise InvalidDimensionError(
                dimension,
                self.config.VALID_DIMENSIONS
            )
        
        # 生成或使用run_id
        if run_id is None:
            run_id = self._current_run_id or self.generate_timestamp_id()
        
        # 校验run_id
        if not PathBuilder.is_valid_run_id(run_id):
            raise InvalidRunIdError(
                run_id,
                "只能包含字母、数字、下划线和连字符，长度1-100"
            )
        
        # 校验文件名（允许扩展名）
        PathBuilder.validate_path_component(filename, "filename", allow_extension=True)
        
        # 清理文件名
        sanitized_filename = PathBuilder.sanitize_filename(
            filename,
            max_length=self.config.max_filename_length
        )
        
        # 构建路径
        output_path = PathBuilder.build_path(
            self.config.output_root,
            dimension,
            run_id,
            sanitized_filename
        )
        
        # 自动创建目录
        if auto_create_dir and self.config.auto_mkdir:
            directory = PathBuilder.build_path(
                self.config.output_root,
                dimension,
                run_id
            )
            self._create_directory(directory)
        
        logger.debug(f"生成输出路径: {output_path}")
        
        return output_path
    
    def create_run_directory(
        self,
        dimension: str,
        run_id: Optional[str] = None
    ) -> str:
        """创建运行实例目录
        
        Args:
            dimension: 维度标识
            run_id: 运行实例标识，可选
            
        Returns:
            str: 创建的目录完整路径
            
        Raises:
            InvalidDimensionError: 维度标识非法时抛出
            InvalidRunIdError: 运行实例标识非法时抛出
            DirectoryCreationError: 目录创建失败时抛出
            
        Example:
            >>> dir_path = manager.create_run_directory("3d", "test_run")
            >>> # 创建: ./output/3d/test_run/
        """
        # 校验维度
        if dimension not in self.config.VALID_DIMENSIONS:
            raise InvalidDimensionError(
                dimension,
                self.config.VALID_DIMENSIONS
            )
        
        # 生成或使用run_id
        if run_id is None:
            run_id = self._current_run_id or self.generate_timestamp_id()
        
        # 校验run_id
        if not PathBuilder.is_valid_run_id(run_id):
            raise InvalidRunIdError(
                run_id,
                "只能包含字母、数字、下划线和连字符，长度1-100"
            )
        
        # 构建目录路径
        directory = PathBuilder.build_path(
            self.config.output_root,
            dimension,
            run_id
        )
        
        # 创建目录
        self._create_directory(directory)
        
        # 缓存目录
        cache_key = f"{dimension}/{run_id}"
        self._run_directories[cache_key] = directory
        
        return directory
    
    def get_output_files(
        self,
        dimension: Optional[str] = None,
        run_id: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> List[str]:
        """查询输出文件列表
        
        Args:
            dimension: 维度标识，可选，不指定则查询所有维度
            run_id: 运行实例标识，可选，不指定则查询所有运行实例
            pattern: 文件名匹配模式（glob格式），可选
            
        Returns:
            List[str]: 文件完整路径列表
            
        Example:
            >>> # 查询所有2D输出文件
            >>> files = manager.get_output_files(dimension="2d")
            >>> 
            >>> # 查询特定运行的PNG文件
            >>> files = manager.get_output_files(
            ...     dimension="2d",
            ...     run_id="experiment_001",
            ...     pattern="*.png"
            ... )
        """
        output_root = Path(self.config.output_root)
        
        # 如果输出根目录不存在，返回空列表
        if not output_root.exists():
            logger.debug(f"输出根目录不存在: {output_root}")
            return []
        
        files = []
        
        # 确定要搜索的维度
        dimensions = [dimension] if dimension else list(self.config.VALID_DIMENSIONS)
        
        for dim in dimensions:
            dim_path = output_root / dim
            
            if not dim_path.exists():
                continue
            
            # 确定要搜索的run_id
            if run_id:
                run_ids = [run_id]
            else:
                # 获取所有run_id目录
                run_ids = [
                    d.name for d in dim_path.iterdir()
                    if d.is_dir() and PathBuilder.is_valid_run_id(d.name)
                ]
            
            for rid in run_ids:
                run_path = dim_path / rid
                
                if not run_path.exists():
                    continue
                
                # 获取文件列表
                if pattern:
                    matched_files = list(run_path.glob(pattern))
                else:
                    matched_files = list(run_path.iterdir())
                
                # 过滤文件（排除目录）
                for f in matched_files:
                    if f.is_file():
                        files.append(str(f.absolute()))
        
        logger.debug(f"查询到 {len(files)} 个文件")
        
        return files
    
    def get_latest_outputs(
        self,
        dimension: str
    ) -> Tuple[str, List[str]]:
        """获取最新运行结果
        
        Args:
            dimension: 维度标识
            
        Returns:
            Tuple[str, List[str]]: (最新run_id, 文件路径列表)
            
        Raises:
            InvalidDimensionError: 维度标识非法时抛出
            ValueError: 指定维度不存在时抛出
            
        Example:
            >>> run_id, files = manager.get_latest_outputs("2d")
            >>> print(f"最新运行: {run_id}, 文件数: {len(files)}")
        """
        # 校验维度
        if dimension not in self.config.VALID_DIMENSIONS:
            raise InvalidDimensionError(
                dimension,
                self.config.VALID_DIMENSIONS
            )
        
        output_root = Path(self.config.output_root)
        dim_path = output_root / dimension
        
        if not dim_path.exists():
            raise ValueError(f"维度目录不存在: {dimension}")
        
        # 获取所有run_id目录及其修改时间
        run_dirs = []
        for d in dim_path.iterdir():
            if d.is_dir() and PathBuilder.is_valid_run_id(d.name):
                stat = d.stat()
                run_dirs.append((d.name, stat.st_mtime))
        
        if not run_dirs:
            raise ValueError(f"维度 {dimension} 下没有运行实例")
        
        # 按修改时间排序，获取最新的
        run_dirs.sort(key=lambda x: x[1], reverse=True)
        latest_run_id = run_dirs[0][0]
        
        # 获取文件列表
        files = self.get_output_files(dimension, latest_run_id)
        
        logger.debug(
            f"获取最新运行结果: dimension={dimension}, "
            f"run_id={latest_run_id}, files={len(files)}"
        )
        
        return (latest_run_id, files)
    
    def set_current_run_id(self, run_id: str) -> None:
        """设置当前运行实例标识
        
        Args:
            run_id: 运行实例标识
            
        Raises:
            InvalidRunIdError: 运行实例标识非法时抛出
            
        Example:
            >>> manager.set_current_run_id("experiment_002")
        """
        if not PathBuilder.is_valid_run_id(run_id):
            raise InvalidRunIdError(
                run_id,
                "只能包含字母、数字、下划线和连字符，长度1-100"
            )
        
        self._current_run_id = run_id
        logger.debug(f"设置当前运行实例标识: {run_id}")
    
    def generate_timestamp_id(self) -> str:
        """生成时间戳格式的运行实例标识
        
        Returns:
            str: 格式化的时间戳字符串（YYYYMMDD_HHMMSS）
            
        Example:
            >>> run_id = manager.generate_timestamp_id()
            >>> # 返回: "20260402_143025"
        """
        timestamp = datetime.now().strftime(self.config.timestamp_format)
        logger.debug(f"生成时间戳标识: {timestamp}")
        return timestamp
    
    def get_run_info(
        self,
        dimension: str,
        run_id: str,
        pattern: Optional[str] = None
    ) -> RunInstanceInfo:
        """获取运行实例信息
        
        Args:
            dimension: 维度标识
            run_id: 运行实例标识
            pattern: 文件名匹配模式，可选
            
        Returns:
            RunInstanceInfo: 运行实例信息对象
            
        Raises:
            InvalidDimensionError: 维度标识非法时抛出
            InvalidRunIdError: 运行实例标识非法时抛出
        """
        # 校验参数
        if dimension not in self.config.VALID_DIMENSIONS:
            raise InvalidDimensionError(
                dimension,
                self.config.VALID_DIMENSIONS
            )
        
        if not PathBuilder.is_valid_run_id(run_id):
            raise InvalidRunIdError(
                run_id,
                "只能包含字母、数字、下划线和连字符，长度1-100"
            )
        
        # 构建目录路径
        directory = PathBuilder.build_path(
            self.config.output_root,
            dimension,
            run_id
        )
        
        return RunInstanceInfo.from_directory(
            directory,
            dimension,
            run_id,
            pattern
        )
    
    def _create_directory(self, directory: str) -> None:
        """创建目录（内部方法）
        
        Args:
            directory: 目录路径
            
        Raises:
            DirectoryCreationError: 目录创建失败时抛出
        """
        try:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"目录创建成功: {directory}")
        except OSError as e:
            raise DirectoryCreationError(
                directory,
                str(e)
            )
        except Exception as e:
            raise DirectoryCreationError(
                directory,
                f"未知错误: {e}"
            )
