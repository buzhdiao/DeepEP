# Change current directory into project root
# 中文说明：安装脚本——构建 DeepEP 的 wheel 并 pip 安装。
# 先记住用户当前目录，结束后再回到那里
original_dir=$(pwd)
# 脚本所在目录（$0 为脚本路径），用于切换到项目根目录
script_dir=$(dirname "$0")
# 切换到项目根目录，保证后续命令以仓库根为工作目录
cd "$script_dir"

# Remove old dist file, build, and install
# 删除旧的 dist 产物，构建 wheel 并安装
rm -rf dist
# 调用 setup.py 打包出 wheel 文件
python setup.py bdist_wheel
# 用 pip 安装 dist 目录下新生成的 wheel
pip install dist/*.whl

# Open users' original directory
# 恢复用户原来的工作目录
cd "$original_dir"
