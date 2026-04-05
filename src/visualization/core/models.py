"""
可视化输出管理器数据模型定义

本模块定义了输出管理器相关的所有数据模型类，使用dataclass简化定义。
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class OutputConfig:
    """可视化输出配置
    
    管理输出路径相关的所有配置项。
    
    Attributes:
        output_root: 输出根目录路径
        default_dimension: 默认维度分类
        timestamp_format: 时间戳格式字符串
        auto_mkdir: 是否自动创建目录
        max_filename_length: 文件名最大长度
        VALID_DIMENSIONS: 合法的维度值元组
    """
    
    # 输出根目录
    output_root: str = "./output"
    
    # 默认维度分类
    default_dimension: str = "2d"
    
    # 时间戳格式
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # 是否自动创建目录
    auto_mkdir: bool = True
    
    # 文件名最大长度
    max_filename_length: int = 255
    
    # 合法的维度值
    VALID_DIMENSIONS: Tuple[str, ...] = field(
        default_factory=lambda: ("2d", "3d", "4d"),
        repr=False,
        compare=False
    )
    
    def validate(self) -> None:
        """校验配置项的合法性
        
        Raises:
            ValueError: 配置项不合法时抛出
        """
        # 校验output_root
        if not self.output_root or not isinstance(self.output_root, str):
            raise ValueError("output_root 必须为非空字符串")
        
        # 校验default_dimension
        if self.default_dimension not in self.VALID_DIMENSIONS:
            raise ValueError(
                f"default_dimension 必须为 {self.VALID_DIMENSIONS} 之一，"
                f"当前值: {self.default_dimension}"
            )
        
        # 校验timestamp_format
        try:
            datetime.now().strftime(self.timestamp_format)
        except Exception as e:
            raise ValueError(f"timestamp_format 格式非法: {e}")
        
        # 校验max_filename_length
        if not (0 < self.max_filename_length <= 255):
            raise ValueError("max_filename_length 必须在 1-255 之间")
    
    def update(self, **kwargs) -> None:
        """更新配置项
        
        Args:
            **kwargs: 要更新的配置项键值对
            
        Raises:
            ValueError: 未知配置项或配置校验失败时抛出
        """
        for key, value in kwargs.items():
            if hasattr(self, key) and key != 'VALID_DIMENSIONS':
                setattr(self, key, value)
            else:
                raise ValueError(f"未知配置项: {key}")
        self.validate()
    
    def get_output_path(self) -> Path:
        """获取输出根目录的Path对象
        
        Returns:
            Path: 输出根目录的Path对象
        """
        return Path(self.output_root)


@dataclass
class OutputFileInfo:
    """输出文件信息
    
    存储单个输出文件的完整信息。
    
    Attributes:
        full_path: 文件完整路径
        filename: 文件名
        dimension: 维度标识
        run_id: 运行实例标识
        size: 文件大小（字节）
        created_at: 创建时间
        extension: 文件扩展名
    """
    
    # 文件完整路径
    full_path: str
    
    # 文件名
    filename: str
    
    # 维度标识
    dimension: str
    
    # 运行实例标识
    run_id: str
    
    # 文件大小（字节）
    size: int
    
    # 创建时间
    created_at: datetime
    
    # 文件扩展名
    extension: str
    
    @classmethod
    def from_path(
        cls,
        full_path: str,
        dimension: str,
        run_id: str
    ) -> 'OutputFileInfo':
        """从路径创建文件信息对象
        
        Args:
            full_path: 文件完整路径
            dimension: 维度标识
            run_id: 运行实例标识
            
        Returns:
            OutputFileInfo: 文件信息对象
            
        Raises:
            FileNotFoundError: 文件不存在时抛出
        """
        path = Path(full_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {full_path}")
        
        stat = path.stat()
        
        return cls(
            full_path=str(path.absolute()),
            filename=path.name,
            dimension=dimension,
            run_id=run_id,
            size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            extension=path.suffix
        )


@dataclass
class RunInstanceInfo:
    """运行实例信息
    
    存储单次运行实例的完整信息。
    
    Attributes:
        run_id: 运行实例标识
        dimension: 维度标识
        directory: 目录路径
        file_count: 文件数量
        created_at: 创建时间
        files: 文件列表
    """
    
    # 运行实例标识
    run_id: str
    
    # 维度标识
    dimension: str
    
    # 目录路径
    directory: str
    
    # 文件数量
    file_count: int
    
    # 创建时间
    created_at: datetime
    
    # 文件列表
    files: List[str]
    
    @classmethod
    def from_directory(
        cls,
        directory: str,
        dimension: str,
        run_id: str,
        pattern: Optional[str] = None
    ) -> 'RunInstanceInfo':
        """从目录创建运行实例信息对象
        
        Args:
            directory: 目录路径
            dimension: 维度标识
            run_id: 运行实例标识
            pattern: 文件名匹配模式（glob格式），可选
            
        Returns:
            RunInstanceInfo: 运行实例信息对象
            
        Raises:
            FileNotFoundError: 目录不存在时抛出
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        # 获取文件列表
        if pattern:
            files = [str(f) for f in dir_path.glob(pattern) if f.is_file()]
        else:
            files = [str(f) for f in dir_path.iterdir() if f.is_file()]
        
        # 获取目录创建时间
        stat = dir_path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime)
        
        return cls(
            run_id=run_id,
            dimension=dimension,
            directory=str(dir_path.absolute()),
            file_count=len(files),
            created_at=created_at,
            files=files
        )
