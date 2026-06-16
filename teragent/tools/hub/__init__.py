# teragent/tools/hub/__init__.py
"""工具市场（Tool Hub）子模块

提供工具市场的客户端接口，支持:
  - 搜索远程工具市场中的工具
  - 安装远程工具到本地注册表
  - 发布本地工具到远程市场
  - 查询已安装的市场工具

核心组件:
  - ToolHubClient: 工具市场客户端（异步 HTTP）
  - ToolHubEntry: 工具市场条目（搜索结果数据类）
  - ToolHubError: 工具市场异常
  - HubTool: 从市场安装的工具（BaseTool 子类，远程执行代理）
"""

from teragent.tools.hub.client import HubTool, ToolHubClient, ToolHubEntry, ToolHubError

__all__ = [
    "ToolHubClient",
    "ToolHubEntry",
    "ToolHubError",
    "HubTool",
]
