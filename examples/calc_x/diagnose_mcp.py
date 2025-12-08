#!/usr/bin/env python3
"""Diagnostic script to check MCP calculator setup."""

import asyncio
import os
import subprocess
import sys


def check_command(cmd: str) -> bool:
    """Check if a command is available."""
    try:
        result = subprocess.run(
            ["which", cmd],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✅ {cmd} found at: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {cmd} not found")
            return False
    except Exception as e:
        print(f"❌ Error checking {cmd}: {e}")
        return False


def check_uv_version():
    """Check uv version."""
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"   uv version: {result.stdout.strip()}")
    except Exception as e:
        print(f"   Cannot get uv version: {e}")


async def test_mcp_server():
    """Test MCP calculator server connection."""
    print("\n3. Testing MCP calculator server connection...")

    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        # 强烈建议先通过 uv 安装一次：
        #   uv tool install mcp-server-calculator
        # 然后这里改为直接调用可执行文件，避免 uvx 带来的 stdout 噪音
        server_params = StdioServerParameters(
            command="mcp-server-calculator",
            args=[],
            env={
                # 保守地尽量让子进程“安静”
                "PYTHONUNBUFFERED": "1",
                "NO_COLOR": "1",
                "TERM": "dumb",
                # uv 相关的静默&离线（如果该工具内部仍依赖 uv 运行时）
                "UV_LOG_LEVEL": "error",
                "UV_NO_SYNC": "1",
                "UV_PYTHON": sys.executable,
                # 取消可能影响 stdout 的代理
                "http_proxy": "",
                "https_proxy": "",
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
            },
        )

        print("   Starting MCP server (direct binary, no uvx)...")
        async with stdio_client(server_params) as (read, write):
            print("   Server started, initializing session...")
            async with ClientSession(read, write) as session:
                init_result = await asyncio.wait_for(
                    session.initialize(),
                    timeout=30.0,
                )
                print("✅ MCP server initialized successfully!")
                print(f"   Server info: {init_result}")

                # 调用工具
                result = await session.call_tool(
                    "calculate",
                    arguments={"expression": "2 + 2"},
                )
                print(f"✅ Test calculation successful: 2 + 2 = {result}")

    except asyncio.TimeoutError:
        print("❌ MCP server initialization timed out (>30s)")
        print("   Possible causes:")
        print("   - MCP server not responding")
        print("   - stdout 被非 JSON-RPC 内容污染（请确认未用 uvx）")
        print("   - 服务器未进入 server.run 循环或启动时阻塞")
    except ImportError as e:
        print(f"❌ Missing Python package: {e}")
        print("   Install with: pip install mcp")
    except FileNotFoundError:
        print("❌ 'mcp-server-calculator' not found in PATH")
        print("   Fix with: uv tool install mcp-server-calculator")
        print("   Then ensure ~/.local/bin is in your PATH")
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        import traceback
        traceback.print_exc()


async def main():
    print("=" * 70)
    print("MCP Calculator Environment Diagnostic")
    print("=" * 70)
    
    print("\n1. Checking required commands...")
    uv_ok = check_command("uv")
    uvx_ok = check_command("uvx")
    python_ok = check_command("python")
    
    if uv_ok:
        check_uv_version()
    
    print("\n2. Checking Python packages...")
    try:
        import mcp
        print(f"✅ mcp package installed (version: {mcp.__version__ if hasattr(mcp, '__version__') else 'unknown'})")
    except ImportError:
        print("❌ mcp package not installed")
        print("   Install with: pip install mcp")
    
    try:
        from autogen_ext.tools.mcp import McpWorkbench
        print("✅ autogen_ext.tools.mcp available")
    except ImportError:
        print("❌ autogen_ext.tools.mcp not available")
        print("   Install with: pip install 'autogen-ext[mcp]'")
    
    # Test MCP server
    await test_mcp_server()
    
    print("\n" + "=" * 70)
    print("Diagnostic complete!")
    print("=" * 70)
    
    print("\n📋 Installation instructions if needed:")
    print("1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
    print("2. Install mcp: pip install 'mcp>=1.10.0'")
    print("3. Install autogen-ext: pip install 'autogen-ext[mcp]'")
    print("4. Install MCP calculator: uv tool install mcp-server-calculator")
    print("   OR: uvx mcp-server-calculator (to test if it works)")


if __name__ == "__main__":
    asyncio.run(main())

