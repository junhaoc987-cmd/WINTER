@echo off
chcp 65001 >nul
echo ========================================
echo JSIC WinterHack 2026 - 自主导航模拟器
echo ========================================
echo.
echo 正在启动程序...
echo.

python main.py

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo 程序运行出错！
    echo ========================================
    echo.
    echo 可能的原因：
    echo 1. Python 未安装或未添加到 PATH
    echo 2. 缺少必要的 Python 库
    echo.
    echo 解决方法：
    echo 1. 确保已安装 Python 3.x
    echo 2. 运行以下命令安装依赖：
    echo    pip install numpy matplotlib
    echo.
    pause
) else (
    echo.
    echo ========================================
    echo 程序已正常结束
    echo ========================================
    echo.
)

pause

