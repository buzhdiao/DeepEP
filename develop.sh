# Change current directory into project root
# 中文说明：开发脚本——就地构建 C++ 扩展并把生成的 .so 软链到 deep_ep 包目录，
# 使源码树可以直接 import deep_ep 而无需安装 wheel。
# 先记住用户当前目录，结束后再回到那里
original_dir=$(pwd)
# 脚本所在目录的绝对路径（realpath 消除符号链接），用于切换到项目根目录
script_dir=$(realpath "$(dirname "$0")")
# 切换到项目根目录，保证后续命令以仓库根为工作目录
cd "$script_dir"

# Remove old dist file, build files, and build
# 删除旧的 dist 产物、build 中间文件与 egg-info 元数据后重新构建
rm -rf build dist
rm -rf *.egg-info
# 只执行 build 阶段（扩展 .so 生成在 build/ 下，不打包 wheel）
python setup.py build

# Find the .so file in build directory and create symlink in current directory
# 在 build 目录里找到编译出的 .so，并在当前目录建软链以便直接 import
so_file=$(find build -name "*.so" -type f | head -n 1)
if [ -n "$so_file" ]; then
    # 找到了：把相对路径的 .so 软链进 deep_ep/ 包目录
    ln -sf "../$so_file" deep_ep/
else
    # 没找到：报错并以非零状态退出
    echo "Error: No SO file found in build directory" >&2
    exit 1
fi

# Open users' original directory
# 恢复用户原来的工作目录
cd "$original_dir"
