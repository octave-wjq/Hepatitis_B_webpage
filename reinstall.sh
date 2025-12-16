#!/bin/bash
# 重新安装依赖脚本

set -e

echo "=========================================="
echo "重新安装依赖(匹配训练环境)"
echo "=========================================="

# 激活环境
echo "1. 激活 conda 环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hepatitis_che

# 强制重新安装
echo "2. 重新安装依赖..."
pip install -r requirements.txt --force-reinstall -q

# 验证版本
echo "3. 验证安装版本..."
python -c "
import pandas, numpy, sklearn, scipy, xgboost, lightgbm
print('✓ pandas:', pandas.__version__)
print('✓ numpy:', numpy.__version__)
print('✓ scikit-learn:', sklearn.__version__)
print('✓ scipy:', scipy.__version__)
print('✓ xgboost:', xgboost.__version__)
print('✓ lightgbm:', lightgbm.__version__)
"

# 测试模型加载
echo ""
echo "4. 测试模型加载..."
python -c "
import joblib
import warnings
warnings.filterwarnings('ignore')

models = ['rf.pkl', 'log_reg.pkl', 'svm.pkl', 'xgb.pkl', 'lgb.pkl', 'mlp.pkl']
success = 0
for model_name in models:
    try:
        model = joblib.load(f'model/saved_models/{model_name}')
        print(f'  ✓ {model_name}')
        success += 1
    except Exception as e:
        print(f'  ✗ {model_name}: {str(e)[:50]}')

print(f'\n成功加载: {success}/6 个模型')
"

echo ""
echo "=========================================="
echo "安装完成!"
echo "启动命令: ./start_app.sh"
echo "=========================================="
