"""
环境检查脚本：检测 Visual C++ (cl.exe), nvcc, nvidia-smi, Python 版本与已安装包（jittor、imageio 等）
用法：
  python tools/check_env.py [--unsafe-import-jittor]
默认不导入 jittor（避免触发自动编译），添加 --unsafe-import-jittor 则尝试 import jittor 并打印结果（可能触发编译/链接）。
"""
import sys
import os
import platform
import shutil
import argparse
import subprocess
import json

def check_executable(name):
    path = shutil.which(name)
    return path

def run_cmd(cmd_list):
    try:
        p = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

def check_python():
    info = {
        "executable": sys.executable,
        "version": platform.python_version(),
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "micro": sys.version_info.micro,
    }
    return info

def check_packages(packages):
    found = {}
    for p in packages:
        try:
                        try:
                            # Better detection: use importlib.metadata to get installed distribution
                            from importlib import metadata
                            ver = metadata.version(p)
                            found[p] = True
                        except Exception:
                            import importlib
                            spec = importlib.util.find_spec(p)
                            found[p] = bool(spec)
        except Exception:
            found[p] = False
    return found

def check_vs_vars():
    keys = ["VSINSTALLDIR", "VCINSTALLDIR", "VS150COMNTOOLS", "VSCMD_ARG_TGT_ARCH"]
    return {k: os.environ.get(k) for k in keys}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsafe-import-jittor", action="store_true", help="尝试导入 jittor（可能触发编译/链接）")
    args = parser.parse_args()

    report = {}
    report['python'] = check_python()
    report['cl'] = check_executable('cl')
    report['nvcc'] = check_executable('nvcc')
    report['nvidia_smi'] = check_executable('nvidia-smi')
    report['where_cl'] = run_cmd(["where", "cl"]) if platform.system() == 'Windows' else (None, None, None)
    report['packages'] = check_packages(['jittor', 'imageio', 'numpy', 'pillow', 'torch'])
    report['vs_env'] = check_vs_vars()
    report['cpu_count'] = os.cpu_count()
    # Check nvcc version if present
    if report['nvcc']:
        ret, out, err = run_cmd(['nvcc', '--version'])
        report['nvcc_version'] = out or err
    else:
        report['nvcc_version'] = None

    # If user wants to try import jittor, do it now
    jittor_import_result = None
    if args.unsafe_import_jittor:
        try:
            print('\n注意：正在尝试 import jittor，这可能触发编译，可能会输出很多日志，请耐心等待（或按 Ctrl+C 取消）。')
            import jittor as jt
            jittor_import_result = {
                'imported': True,
                'version': getattr(jt, '__version__', 'unknown'),
                'has_cuda': getattr(jt, 'has_cuda', False) if hasattr(jt, 'has_cuda') else getattr(jt, 'has_cudnn', False)
            }
        except Exception as e:
            jittor_import_result = {'imported': False, 'error': str(e)}
    report['jittor_import'] = jittor_import_result

    # If jittor is installed, fetch its package version using importlib.metadata
    try:
        from importlib import metadata
        jittor_ver = None
        try:
            jittor_ver = metadata.version('jittor')
        except metadata.PackageNotFoundError:
            jittor_ver = None
        report['jittor_version'] = jittor_ver
    except Exception:
        report['jittor_version'] = None

    print('\n--- 环境检测报告（可复制为 JSON）---')
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Basic suggestions based on the report
    print('\n--- 建议 ---')
    if not report['cl']:
        print('⚠️ 未检测到 MSVC 的 cl.exe。建议安装 Visual Studio Build Tools（包含“Desktop development with C++” workload）。')
    else:
        print(f'✅ 检测到 cl.exe: {report["cl"]}')

    if not report['nvcc'] and report['nvidia_smi']:
        print('⚠️ 未检测到 nvcc，但检测到 nvidia-smi。可能需要安装 CUDA Toolkit 并把 nvcc 加入 PATH（或检查安装路径）。')
    elif report['nvcc']:
        print(f'✅ 检测到 nvcc：{report["nvcc_version"]}')

    if not report['packages'].get('jittor'):
        print('⚠️ 未检测到 jittor 已安装。若你需要使用 Jittor，请参考 https://github.com/Jittor/jittor 安装步骤。')
    else:
        print(f"✅ 检测到 jittor 安装 (版本: {report.get('jittor_version', 'unknown')})。若你遇到导入时的 C++ 编译错误，请检查 Python 与 CUDA 的兼容性。")

    print('\n更多帮助：')
    print('- 若使用 Windows + GPU，建议确保: Visual Studio Build Tools, CUDA Toolkit 与 NVIDIA 驱动版本兼容，以及 Python 版本与 jittor wheel 匹配。')
    print('- 若你想让脚本也帮你尝试 `jittor` 导入并捕获更详细错误信息，可再次执行：')
    print(f'  {sys.executable} tools/check_env.py --unsafe-import-jittor')

if __name__ == '__main__':
    main()

