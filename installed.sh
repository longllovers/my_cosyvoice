# 先conda 一个环境，python3.8，激活一个环境
#最后面，chmod +x 这个sh文件，运行这个sh文件


# 设置错误处理
set -e

# 克隆仓库并更新子模块

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 判断是否在 my_cosyvoice 目录下
if [ "${SCRIPT_DIR##*/}" != "my_cosyvoice" ]; then
    echo "当前不在 my_cosyvoice 目录，切换到 my_cosyvoice 目录..."
    cd "$SCRIPT_DIR/my_cosyvoice" || { echo "无法切换到 my_cosyvoice 目录"; exit 1; }
else
    echo "已经在 my_cosyvoice 目录下"
fi


git submodule update --init --recursive || {
    echo "Failed to update submodules"
    exit 1
}

echo "Installing pynini..."
pip install  pynini==2.1.5  -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Installing Python packages..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchaudio==2.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

if [[ "$(lsb_release -is)" == "Ubuntu" ]]; then
    echo "Installing sox on Ubuntu..."
    sudo apt-get install -y sox libsox-dev
elif [[ "$(lsb_release -is)" == "CentOS" ]]; then
    echo "Installing sox on CentOS..."
    sudo yum install -y sox sox-devel
fi

# 创建预训练模型文件夹
mkdir -p pretrained_models


export PYTHONPATH=$(pwd)/third_party/Matcha-TTS


# 完成
echo "Setup complete. You can now use CosyVoice."



