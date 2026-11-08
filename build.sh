# Change current directory into project root
# 中文说明：构建脚本——清理旧产物并构建 DeepEP 的 wheel（不安装）。
# 先记住用户当前目录，结束后再回到那里
original_dir=$(pwd)
# 脚本所在目录的绝对路径（realpath 消除符号链接），用于切换到项目根目录
script_dir=$(realpath "$(dirname "$0")")
# 切换到项目根目录，保证后续命令以仓库根为工作目录
cd "$script_dir"

# Remove old dist file, build files, and install
# 删除旧的 dist 产物、build 中间文件与 egg-info 元数据
rm -rf build dist
rm -rf *.egg-info
# 调用 setup.py 打包出 wheel 文件（产物在 dist/ 下，可自行 pip 安装）
python setup.py bdist_wheel

# Open users' original directory
# 恢复用户原来的工作目录
cd "$original_dir"
