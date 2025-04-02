#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from typing import Optional, Union, BinaryIO, Dict, Any

import oss2
from oss2.exceptions import OssError

from app.core.config import settings

class AliyunOSS:
    """
    阿里云OSS操作工具类
    提供简单的文件上传、下载、删除等操作
    """
    
    def __init__(self):
        """
        初始化OSS客户端
        从应用配置中读取OSS配置信息
        """
        self.access_key = settings.OSS_ACCESS_KEY
        self.secret_key = settings.OSS_SECRET_KEY
        self.bucket_name = settings.OSS_BUCKET
        self.endpoint = settings.OSS_ENDPOINT
        self.region = settings.OSS_REGION
        
        # 验证必要的配置是否存在
        if not all([self.access_key, self.secret_key, self.bucket_name, self.endpoint]):
            logging.error("阿里云OSS配置不完整，请检查环境变量或配置文件")
            raise ValueError("阿里云OSS配置不完整")
        
        # 初始化认证和Bucket对象
        self.auth = oss2.Auth(self.access_key, self.secret_key)
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)
        logging.info(f"阿里云OSS客户端初始化完成，Bucket: {self.bucket_name}, Endpoint: {self.endpoint}")
    
    def upload_file(self, 
                   local_file_path: str, 
                   oss_file_path: Optional[str] = None,
                   headers: Optional[Dict[str, str]] = None) -> str:
        """
        上传本地文件到OSS
        
        Args:
            local_file_path: 本地文件路径
            oss_file_path: OSS上的文件路径，如果为None则使用本地文件名
            headers: 请求头，可以用来设置Content-Type等
            
        Returns:
            上传后的文件URL
            
        Raises:
            FileNotFoundError: 本地文件不存在
            OssError: OSS操作失败
        """
        logging.info(f"开始上传文件: {local_file_path}")
        logging.debug(f"OSS目标路径: {oss_file_path}")
        logging.debug(f"请求头: {headers}")
        
        if not os.path.exists(local_file_path):
            logging.error(f"要上传的文件不存在: {local_file_path}")
            raise FileNotFoundError(f"文件不存在: {local_file_path}")
        
        # 如果没有指定OSS文件路径，则使用本地文件名
        if not oss_file_path:
            oss_file_path = os.path.basename(local_file_path)
            logging.info(f"未指定OSS路径，使用本地文件名: {oss_file_path}")
        
        try:
            # 获取文件大小
            file_size = os.path.getsize(local_file_path)
            logging.info(f"文件大小: {file_size} 字节")
            
            # 上传文件
            logging.info(f"开始上传到OSS: {oss_file_path}")
            result = self.bucket.put_object_from_file(oss_file_path, local_file_path, headers=headers)
            
            # 生成访问URL
            file_url = f"https://{self.bucket_name}.{self.endpoint.replace('http://', '').replace('https://', '')}/{oss_file_path}"
            
            logging.info(f"文件上传成功: {local_file_path} -> {oss_file_path}")
            logging.info(f"ETag: {result.etag}")
            logging.info(f"访问URL: {file_url}")
            
            return file_url
            
        except OssError as e:
            logging.error(f"OSS上传失败: {e.code}, {e.message}, {e.request_id}")
            logging.error(f"错误详情: {e.details}")
            raise
        except Exception as e:
            logging.error(f"上传过程中发生未知错误: {str(e)}")
            raise e
    def file_exists(self, oss_file_path: str) -> bool:
        """
        检查OSS上是否存在指定文件
        
        Args:
            oss_file_path: OSS上的文件路径
            
        Returns:
            文件是否存在
        """
        return self.bucket.object_exists(oss_file_path)
    
    def get_file_info(self, oss_file_path: str) -> Dict[str, Any]:
        """
        获取OSS文件的元信息
        
        Args:
            oss_file_path: OSS上的文件路径
            
        Returns:
            文件元信息字典
        """
        try:
            meta = self.bucket.get_object_meta(oss_file_path)
            return {
                'last_modified': meta.last_modified,
                'etag': meta.etag,
                'content_length': meta.content_length,
                'content_type': meta.content_type
            }
        except OssError as e:
            logging.error(f"获取文件信息失败: {e.code}, {e.message}, {e.request_id}")
            raise

# 创建默认OSS实例，方便直接导入使用
oss_client = AliyunOSS()
