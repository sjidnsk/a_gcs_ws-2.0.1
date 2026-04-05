"""
可视化输出管理器异常类定义

本模块定义了输出管理器相关的所有自定义异常类，提供明确的错误信息和上下文。
"""


class OutputManagerError(Exception):
    """输出管理器基础异常
    
    所有输出管理器相关的异常都继承自此类。
    """
    pass


class InvalidDimensionError(OutputManagerError):
    """非法维度标识异常
    
    当传入的维度标识不在合法枚举值范围内时抛出。
    
    Attributes:
        dimension: 非法的维度标识
        valid_dimensions: 合法的维度标识元组
    """
    
    def __init__(self, dimension: str, valid_dimensions: tuple):
        """
        初始化异常
        
        Args:
            dimension: 非法的维度标识
            valid_dimensions: 合法的维度标识元组
        """
        self.dimension = dimension
        self.valid_dimensions = valid_dimensions
        super().__init__(
            f"维度标识 '{dimension}' 非法，必须为 {valid_dimensions} 之一"
        )


class InvalidRunIdError(OutputManagerError):
    """非法运行实例标识异常
    
    当传入的运行实例标识包含非法字符或格式不正确时抛出。
    
    Attributes:
        run_id: 非法的运行实例标识
        reason: 非法原因描述
    """
    
    def __init__(self, run_id: str, reason: str):
        """
        初始化异常
        
        Args:
            run_id: 非法的运行实例标识
            reason: 非法原因描述
        """
        self.run_id = run_id
        self.reason = reason
        super().__init__(f"运行实例标识 '{run_id}' 非法: {reason}")


class DirectoryCreationError(OutputManagerError):
    """目录创建失败异常
    
    当无法创建指定的目录时抛出，通常由于权限不足或路径非法。
    
    Attributes:
        path: 无法创建的目录路径
        reason: 失败原因描述
    """
    
    def __init__(self, path: str, reason: str):
        """
        初始化异常
        
        Args:
            path: 无法创建的目录路径
            reason: 失败原因描述
        """
        self.path = path
        self.reason = reason
        super().__init__(f"无法创建目录 '{path}': {reason}")


class PathValidationError(OutputManagerError):
    """路径校验失败异常
    
    当路径组件包含非法字符或不符合规范时抛出。
    
    Attributes:
        component: 校验失败的路径组件
        reason: 失败原因描述
    """
    
    def __init__(self, component: str, reason: str):
        """
        初始化异常
        
        Args:
            component: 校验失败的路径组件
            reason: 失败原因描述
        """
        self.component = component
        self.reason = reason
        super().__init__(f"路径组件 '{component}' 校验失败: {reason}")


class FilenameTooLongError(OutputManagerError):
    """文件名过长异常
    
    当文件名超过系统限制长度时抛出。
    
    Attributes:
        filename: 过长的文件名
        max_length: 最大允许长度
    """
    
    def __init__(self, filename: str, max_length: int):
        """
        初始化异常
        
        Args:
            filename: 过长的文件名
            max_length: 最大允许长度
        """
        self.filename = filename
        self.max_length = max_length
        super().__init__(
            f"文件名 '{filename}' 长度为 {len(filename)}，"
            f"超过最大限制 {max_length}"
        )
