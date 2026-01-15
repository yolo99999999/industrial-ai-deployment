#!/usr/bin/env python3
"""
AI部署环境检查脚本
作者：你的学习计划
日期：2026-01-12
"""

import sys
import subprocess
import importlib
import os

def check_tool(name, command, success_msg):
    """通用工具检查函数"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {name}: {success_msg}")
            return True
        else:
            print(f"❌ {name}: 未安装或配置错误")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

def check_python():
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python: {version.major}.{version.minor}.{version.micro} (符合要求)")
        return True
    else:
        print(f"⚠️  Python: {version.major}.{version.minor}.{version.micro} (建议升级到3.11+)")
        return False

def check_pip_package(package_name):
    """检查Python包是否安装"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name}: 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name}: 未安装")
        return False

def main():
    print("=" * 50)
    print("🚀 AI部署环境检查开始")
    print("=" * 50)
    
    # 基础工具检查
    results = []
    results.append(check_python())
    results.append(check_tool("Git", "git --version", "已安装"))
    results.append(check_tool("Docker", "docker --version", "已安装"))
    
    # Python关键包检查
    key_packages = ["numpy", "cv2", "onnxruntime", "fastapi", "uvicorn"]
    for package in key_packages:
        # 处理包名中的连字符
        module_name = package.replace("-", "_")
        results.append(check_pip_package(module_name))
    
    # 额外检查
    print("\n📋 额外检查:")
    
    # 检查当前目录是否是git仓库
    if os.path.exists(".git"):
        print("✅ Git仓库: 已初始化")
        # 检查远程仓库
        try:
            result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
            if result.stdout.strip():
                print("✅ 远程仓库: 已关联")
            else:
                print("⚠️  远程仓库: 未关联")
        except:
            print("❌ 远程仓库: 检查失败")
    else:
        print("⚠️  Git仓库: 未初始化")
    
    # 检查README文件
    if os.path.exists("README.md"):
        print("✅ README: 已存在")
    else:
        print("⚠️  README: 不存在")
    
    # 总结
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📊 检查结果: {passed}/{total} 项通过")
    
    if passed == total:
        print("🎉 环境准备完成！可以开始写代码了")
    else:
        print("🔧 还有问题需要解决，请看上面的❌项")
        print("💡 建议按顺序修复：")
        print("   1. 安装缺少的工具")
        print("   2. pip install 缺少的包")
        print("   3. 初始化git仓库")

if __name__ == "__main__":
    main()