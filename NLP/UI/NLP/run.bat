@echo off
chcp 65001 >nul
echo.
echo === NLP 文本智能分析平台 ===
echo.
echo 正在检查依赖...
python3 -c "import gradio" 2>nul
if errorlevel 1 (
    echo [安装] gradio 未安装，正在安装...
    pip install gradio -q
)
echo.
echo [启动] 正在启动 NLP UI 服务...
echo         访问地址: http://localhost:7865
echo.
python3 "%~dp0app.py" --port 7865
if errorlevel 1 (
    echo [错误] 启动失败，请手动运行: python app.py --port 7865
    pause
)
