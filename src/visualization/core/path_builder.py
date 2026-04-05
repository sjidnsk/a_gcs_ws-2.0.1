"""
路径构建工具类

本模块提供路径构建相关的工具方法，无状态，可复用。
"""

import os
import re
from pathlib import Path
from typing import Optional

from .exceptions import PathValidationError, FilenameTooLongError


class PathBuilder:
    """路径构建工具类
    
    提供路径拼接、文件名处理、路径校验等静态工具方法。
    所有方法都是静态方法，无状态，可复用。
    """
    
    # 非法文件名字符（Windows + Unix）
    ILLEGAL_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'
    
    # 相对路径符号
    RELATIVE_PATH_SYMBOLS = ('..', '.')
    
    @staticmethod
    def build_path(*components: str) -> str:
        """拼接路径组件
        
        使用pathlib处理跨平台路径分隔符，自动规范化路径。
        
        Args:
            *components: 路径组件列表
            
        Returns:
            str: 拼接后的完整路径
            
        Example:
            >>> PathBuilder.build_path('./output', '2d', 'run001', 'file.png')
            './output/2d/run001/file.png'
        """
        if not components:
            return '.'
        
        # 使用Path处理路径拼接
        path = Path(components[0])
        for component in components[1:]:
            path = path / component
        
        return str(path)
    
    @staticmethod
    def sanitize_filename(
        filename: str,
        replacement: str = "_",
        max_length: Optional[int] = None
    ) -> str:
        """清理文件名中的非法字符
        
        替换或移除文件名中的非法字符，确保文件名在所有平台上合法。
        
        Args:
            filename: 原始文件名
            replacement: 替换字符，默认为下划线
            max_length: 最大长度限制，可选
            
        Returns:
            str: 清理后的文件名
            
        Raises:
            FilenameTooLongError: 文件名过长时抛出
            
        Example:
            >>> PathBuilder.sanitize_filename('file<name>.png')
            'file_name_.png'
        """
        if not filename:
            return filename
        
        # 替换非法字符
        sanitized = re.sub(
            PathBuilder.ILLEGAL_FILENAME_CHARS,
            replacement,
            filename
        )
        
        # 移除首尾空格和点
        sanitized = sanitized.strip(' .')
        
        # 确保文件名不为空
        if not sanitized:
            sanitized = 'unnamed'
        
        # 检查长度限制
        if max_length and len(sanitized) > max_length:
            raise FilenameTooLongError(sanitized, max_length)
        
        return sanitized
    
    @staticmethod
    def ensure_unique_filename(
        directory: str,
        filename: str,
        start_index: int = 1
    ) -> str:
        """确保文件名唯一
        
        如果目标目录中已存在同名文件，自动添加序号后缀避免冲突。
        
        Args:
            directory: 目标目录
            filename: 原始文件名
            start_index: 起始序号，默认为1
            
        Returns:
            str: 唯一的文件名（必要时添加序号）
            
        Example:
            >>> PathBuilder.ensure_unique_filename('./output', 'file.png')
            'file_1.png'  # 如果file.png已存在
        """
        dir_path = Path(directory)
        
        # 如果目录不存在，直接返回原文件名
        if not dir_path.exists():
            return filename
        
        # 检查文件是否存在
        file_path = dir_path / filename
        if not file_path.exists():
            return filename
        
        # 分离文件名和扩展名
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            name, ext = name_parts
            ext = '.' + ext
        else:
            name = filename
            ext = ''
        
        # 添加序号后缀
        index = start_index
        while True:
            new_filename = f"{name}_{index}{ext}"
            new_path = dir_path / new_filename
            if not new_path.exists():
                return new_filename
            index += 1
            
            # 防止无限循环
            if index > 10000:
                raise RuntimeError(f"无法生成唯一文件名: {filename}")
    
    @staticmethod
    def validate_path_component(
        component: str,
        name: str,
        allow_empty: bool = False,
        allow_extension: bool = False
    ) -> None:
        """校验路径组件合法性
        
        检查路径组件是否包含非法字符或相对路径符号。
        
        Args:
            component: 路径组件
            name: 组件名称（用于错误提示）
            allow_empty: 是否允许空字符串，默认为False
            allow_extension: 是否允许扩展名（文件名），默认为False
            
        Raises:
            PathValidationError: 组件不合法时抛出
            
        Example:
            >>> PathBuilder.validate_path_component('2d', 'dimension')
            # 通过校验
            
            >>> PathBuilder.validate_path_component('file.png', 'filename', allow_extension=True)
            # 通过校验
            
            >>> PathBuilder.validate_path_component('../', 'path')
            # 抛出 PathValidationError
        """
        # 检查空值
        if not component:
            if not allow_empty:
                raise PathValidationError(
                    component,
                    f"{name} 不能为空"
                )
            return
        
        # 检查类型
        if not isinstance(component, str):
            raise PathValidationError(
                str(component),
                f"{name} 必须为字符串类型"
            )
        
        # 检查相对路径符号
        # 如果允许扩展名（文件名），则不检查单独的'.'
        if allow_extension:
            # 对于文件名，只检查'..'，不检查单独的'.'
            if '..' in component:
                raise PathValidationError(
                    component,
                    f"{name} 不能包含相对路径符号 '..'"
                )
        else:
            # 对于路径组件，检查所有相对路径符号
            for symbol in PathBuilder.RELATIVE_PATH_SYMBOLS:
                if symbol in component:
                    raise PathValidationError(
                        component,
                        f"{name} 不能包含相对路径符号 '{symbol}'"
                    )
        
        # 检查非法字符
        if re.search(PathBuilder.ILLEGAL_FILENAME_CHARS, component):
            raise PathValidationError(
                component,
                f"{name} 包含非法字符"
            )
    
    @staticmethod
    def is_valid_run_id(run_id: str) -> bool:
        """检查运行实例标识是否合法
        
        合法的run_id只能包含字母、数字、下划线和连字符。
        
        Args:
            run_id: 运行实例标识
            
        Returns:
            bool: 是否合法
            
        Example:
            >>> PathBuilder.is_valid_run_id('experiment_001')
            True
            
            >>> PathBuilder.is_valid_run_id('run 001')
            False
        """
        if not run_id or not isinstance(run_id, str):
            return False
        
        # 检查长度
        if len(run_id) < 1 or len(run_id) > 100:
            return False
        
        # 检查字符：只允许字母、数字、下划线和连字符
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, run_id))
    
    @staticmethod
    def get_filename_extension(filename: str) -> tuple:
        """分离文件名和扩展名
        
        Args:
            filename: 文件名
            
        Returns:
            tuple: (文件名, 扩展名)
            
        Example:
            >>> PathBuilder.get_filename_extension('file.png')
            ('file', '.png')
        """
        path = Path(filename)
        name = path.stem
        ext = path.suffix
        return (name, ext)
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """规范化路径
        
        统一路径分隔符，移除多余的斜杠。
        
        Args:
            path: 原始路径
            
        Returns:
            str: 规范化后的路径
            
        Example:
            >>> PathBuilder.normalize_path('./output//2d/')
            './output/2d'
        """
        # 使用Path规范化路径
        normalized = str(Path(path))
        
        # 移除末尾的分隔符
        if normalized != '/' and normalized.endswith('/'):
            normalized = normalized[:-1]
        
        return normalized
