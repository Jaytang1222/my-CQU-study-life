@echo off
setlocal

:: --- 配置参数 ---
SET MSVC_URL=https://cg.cs.tsinghua.edu.cn/jittor/assets/msvc.zip
SET CACHE_ROOT=%USERPROFILE%\.cache\jittor\msvc
SET DOWNLOAD_PATH=%CD%\msvc.zip

echo.
echo =======================================================
echo 🛠️ Jittor MSVC 依赖手动下载和配置脚本
echo =======================================================
echo.

:: 1. 创建目标缓存目录
echo 1. 正在创建目标缓存目录: %CACHE_ROOT%
mkdir "%CACHE_ROOT%" 2>nul
if not exist "%CACHE_ROOT%" (
    echo ❌ 目录创建失败。请检查权限。
    goto :end
)
echo ✅ 目录准备完成。

:: 2. 尝试使用 PowerShell 下载文件
echo.
echo 2. 尝试从 %MSVC_URL% 下载 msvc.zip...

:: 使用 PowerShell 的 Invoke-WebRequest 进行下载
powershell -Command "Invoke-WebRequest -Uri '%MSVC_URL%' -OutFile '%DOWNLOAD_PATH%'"
if errorlevel 1 (
    echo ❌ 自动下载失败，可能原因：网络问题或502错误。
    echo.
    echo -------------------------------------------------------
    echo ⚠️ 请尝试手动下载：
    echo 1. 在浏览器中打开： %MSVC_URL%
    echo 2. 下载完成后，将 msvc.zip 复制到当前目录。
    echo 3. 运行本批处理文件的第 3 步（直接运行脚本下一步）。
    echo -------------------------------------------------------
    goto :manual_check
)

echo ✅ msvc.zip 已成功下载到当前目录。

:manual_check
:: 3. 检查下载文件是否存在，并移动到 Jittor 缓存
echo.
echo 3. 正在将 msvc.zip 移动到 Jittor 缓存目录...

if not exist "%DOWNLOAD_PATH%" (
    echo ❌ 错误：在 %CD% 目录下找不到 msvc.zip 文件。
    echo 请确认您已手动下载并放置。
    goto :end
)

:: 移动文件 (如果目标已存在，则覆盖)
move /y "%DOWNLOAD_PATH%" "%CACHE_ROOT%\"
if errorlevel 0 (
    echo ✅ msvc.zip 已成功放置在：
    echo    %CACHE_ROOT%\msvc.zip
    echo.
    echo =======================================================
    echo 🎉 配置完成! 现在请运行您的训练脚本。
    echo =======================================================
) else (
    echo ❌ 文件移动失败。请检查 %CACHE_ROOT% 目录权限。
)

:end
echo.
pause
endlocal