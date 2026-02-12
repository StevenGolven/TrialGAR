@echo off
chcp 65001 >nul
echo ======================================
echo  TrialGPT-China Web Interface
echo  临床试验智能匹配系统
echo ======================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo [安装] 正在安装 Gradio...
    pip install gradio --break-system-packages
    if errorlevel 1 (
        echo [错误] Gradio 安装失败
        pause
        exit /b 1
    )
)

echo [2/3] 检查 API Key...
if not defined DEEPSEEK_API_KEY (
    echo [警告] 未设置 DEEPSEEK_API_KEY 环境变量
    echo [提示] 可以在 PowerShell 中运行: setx DEEPSEEK_API_KEY "sk-xxxx"
    echo [提示] 或在 Web 界面中手动输入
    echo.
)

echo [3/3] 启动服务...
echo.
echo ========================================
echo  正在启动 Web 界面...
echo  启动后请在浏览器中打开显示的 URL
echo  通常是: http://127.0.0.1:7860
echo ========================================
echo.

python app_gradio.py

pause
