#!/bin/bash
# 肝硬化CHE风险预测应用启动脚本

set -e

echo "=========================================="
echo "肝硬化CHE风险预测系统启动脚本"
echo "=========================================="

# 1. 激活 conda 环境
echo "1. 激活 conda 环境 hepatitis_che..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hepatitis_che

# 2. 验证依赖
echo "2. 验证依赖版本..."
python -c "
import streamlit
import numpy
import sklearn
print(f'✓ Streamlit: {streamlit.__version__}')
print(f'✓ NumPy: {numpy.__version__}')
print(f'✓ scikit-learn: {sklearn.__version__}')
"

# 3. 设置环境变量
echo "3. 设置环境变量..."
export AUTH_USER=${AUTH_USER:-admin}
export AUTH_PASSWORD=${AUTH_PASSWORD:-test123}
export APP_DB_PATH=${APP_DB_PATH:-data/app.db}

echo "   用户名: $AUTH_USER"
echo "   数据库路径: $APP_DB_PATH"

# 4. 创建数据目录
echo "4. 创建数据目录..."
mkdir -p data
mkdir -p "$(dirname $APP_DB_PATH)"

# 5. 检查模型文件
echo "5. 检查模型文件..."
MODEL_DIR="model/saved_models"
if [ ! -d "$MODEL_DIR" ]; then
    echo "错误: 模型目录不存在: $MODEL_DIR"
    exit 1
fi

MODEL_COUNT=$(ls -1 $MODEL_DIR/*.pkl 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -lt 11 ]; then
    echo "警告: 模型文件不完整(需要11个,找到${MODEL_COUNT}个)"
    echo "请确保以下文件存在:"
    echo "  - log_reg.pkl, rf.pkl, mlp.pkl, svm.pkl, xgb.pkl, lgb.pkl"
    echo "  - scaler.pkl, encoder.pkl, val_cols.pkl, cat_cols.pkl, model_features.pkl"
fi

# 6. 启动应用
echo "6. 启动 Streamlit 应用..."
echo "=========================================="
echo "访问地址: http://localhost:8501"
echo "用户名: $AUTH_USER"
echo "密码: $AUTH_PASSWORD"
echo "=========================================="
echo ""

streamlit run app/main.py --server.port 8501
