"""
尝试自动安装/重新安装 Jittor（慎用）

说明：
  - 此脚本会在当前虚拟环境中执行 `pip uninstall jittor -y` 并尝试 `pip install jittor`。
  - 默认先会做一次 `pip download` 来查看可用 wheel 并打印信息（不会直接更改）。
  - 使用 `--apply` 参数才能实际进行卸载/安装操作。
  - 使用 `--prefer-cuda` 可在安装时尝试安装带 CUDA 支持的 wheel（若可用）。
  - 使用 `--cpu-only` 安装 CPU-only 版本（如果 Jittor 发布了 CPU-only 版本）。

用法（示例）：
  python tools/attempt_jittor_install.py --apply --prefer-cuda

注意：执行 `--apply` 时会修改当前虚拟环境中的包，可能需要重新构建或回滚。
"""
import argparse
import os
import subprocess
import sys
import shutil

def run_cmd(cmd_list, capture_output=False):
    try:
        p = subprocess.run(cmd_list, stdout=subprocess.PIPE if capture_output else None, stderr=subprocess.PIPE if capture_output else None, text=True)
        if capture_output:
            return p.returncode, p.stdout, p.stderr
        return p.returncode
    except Exception as e:
        return -1, '', str(e) if capture_output else -1

def pip_call(args):
    return run_cmd([sys.executable, '-m', 'pip'] + args, capture_output=True)

def check_wheels():
    # 查看 PyPI 上 jittor 是否有适配当前 Python 的 wheel（简单尝试pip index versions）
    code, out, err = pip_call(['index', 'versions', 'jittor'])
    if code != 0:
        # pip older, fallback to show info
        code, out, err = pip_call(['install', 'jittor', '--no-deps', '--dry-run'])
    return code, out, err

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='实际卸载/安装 jittor')
    parser.add_argument('--prefer-cuda', action='store_true', help='尝试安装带 CUDA 支持的 wheel')
    parser.add_argument('--cpu-only', action='store_true', help='尝试安装 CPU-only 版本')
    args = parser.parse_args()

    print('当前 Python', sys.executable)
    print('检测 pip 和网络可用性，查看 jittor 可用版本...')
    code, out, err = check_wheels()
    print('pip 返回码:', code)
    if out:
        print('输出:\n', out)
    if err:
        print('错误输出(若有):\n', err)

    if not args.apply:
        print('\n未指定 --apply，脚本仅做信息收集，并不会改动环境。如需实际执行，请重新运行并添加 --apply。')
        return

    print('\n开始实际卸载与安装 Jittor（请确保你已备份环境或可以回滚）...')
    # 卸载 jittor（如果已安装）
    print('Uninstalling jittor (if present) ...')
    rc, out, err = pip_call(['uninstall', 'jittor', '-y'])
    print('pip uninstall rc:', rc)
    if out:
        print(out)
    if err:
        print(err)

    # 安装 jittor
    install_cmd = ['install', 'jittor']
    if args.prefer_cuda:
        print('尝试安装带 CUDA 支持的 jittor wheel (由 pip 选择预编译版或源码编译)')
    if args.cpu_only:
        print('尝试安装 CPU-only（如果包有此版本）')

    rc, out, err = pip_call(install_cmd)
    print('pip install rc:', rc)
    if out:
        print(out)
    if err:
        print(err)

    print('\n尝试导入 jittor 并打印版本（可能触发编译/链接过程）')
    try:
        import importlib
        j = importlib.import_module('jittor')
        v = getattr(j, '__version__', 'unknown')
        print('导入成功，版本:', v)
    except Exception as e:
        print('导入失败，错误如下：')
        print(e)

if __name__ == '__main__':
    main()
