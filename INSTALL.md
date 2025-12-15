# 肝硬化CHE风险预测系统 - 安装与启动指南

## 环境要求

- Python 3.11
- Conda (Miniconda 或 Anaconda)
- 模型文件(model/saved_models/*.pkl)

---

## 快速开始

### 1. 创建 Conda 环境

```bash
# 创建环境
conda create -n hepatitis_che python=3.11 -y

# 激活环境
conda activate hepatitis_che
```

### 2. 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt
```

### 3. 设置环境变量

```bash
# 设置登录凭据
export AUTH_USER=admin
export AUTH_PASSWORD=your_secure_password

# 可选:设置数据库路径(默认 data/app.db)
export APP_DB_PATH=data/app.db
```

### 4. 启动应用

```bash
# 方式1: 使用启动脚本(推荐)
./start_app.sh

# 方式2: 直接启动
streamlit run app/main.py --server.port 8501
```

### 5. 访问应用

浏览器打开: **http://localhost:8501**

使用设置的用户名和密码登录。

---

## 依赖版本说明

| 包名 | 版本 | 说明 |
|------|------|------|
| streamlit | 1.33.0 | Web框架 |
| pandas | 2.2.2 | 数据处理 |
| numpy | 1.26.4 | 数值计算 |
| scikit-learn | 1.6.1 | 机器学习(匹配训练模型) |
| scipy | 1.11.4 | 科学计算 |
| xgboost | ≥3.0.0 | XGBoost模型 |
| lightgbm | ≥4.0.0 | LightGBM模型 |

---

## 模型文件检查

确保以下文件存在于 `model/saved_models/` 目录:

**必需文件(11个)**:
- `log_reg.pkl` - 逻辑回归模型
- `rf.pkl` - 随机森林模型
- `svm.pkl` - 支持向量机模型
- `xgb.pkl` - XGBoost模型
- `lgb.pkl` - LightGBM模型
- `mlp.pkl` - 多层感知机模型(可选,加载失败不影响其他模型)
- `scaler.pkl` - 数值特征标准化器
- `encoder.pkl` - 分类特征编码器
- `val_cols.pkl` - 数值列名
- `cat_cols.pkl` - 分类列名
- `model_features.pkl` - 模型特征列表

检查命令:
```bash
ls -lh model/saved_models/*.pkl | wc -l
# 应输出: 11
```

---

## 常见问题

### Q1: 模型加载失败

**错误**: `Failed to load mlp.pkl: MT19937 is not a known BitGenerator`

**解决**: MLP模型与当前numpy版本不兼容,但不影响其他5个模型使用。系统会自动跳过该模型。

### Q2: 登录后没有跳转

**解决**: 已修复,登录成功后会自动刷新页面。

### Q3: 缺少 xgboost 或 lightgbm

**解决**:
```bash
pip install xgboost lightgbm
```

### Q4: 数据库权限错误

**解决**:
```bash
mkdir -p data
chmod 755 data
```

---

## 生产部署建议

### 1. 使用环境变量文件

创建 `.env` 文件:
```bash
AUTH_USER=admin
AUTH_PASSWORD=your_secure_password_here
APP_DB_PATH=/var/lib/hepatitis_che/app.db
```

### 2. 使用 systemd 服务

创建 `/etc/systemd/system/hepatitis-che.service`:
```ini
[Unit]
Description=Hepatitis CHE Risk Prediction Service
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/Hepatitis_B_webpage
EnvironmentFile=/path/to/.env
ExecStart=/home/your_user/miniconda3/envs/hepatitis_che/bin/streamlit run app/main.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务:
```bash
sudo systemctl daemon-reload
sudo systemctl enable hepatitis-che
sudo systemctl start hepatitis-che
```

### 3. 使用 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 技术支持

- 项目目录: `/home/octave/workPlace/Hepatitis_B_webpage`
- 日志查看: `tail -f app.log`
- 测试覆盖率: `pytest --cov=app --cov-report=html`

---

## 更新日志

### v1.0.0 (2025-12-15)
- ✅ 用户认证系统
- ✅ 6模型集成预测(5个可用,MLP可选)
- ✅ SQLite数据持久化
- ✅ 测试覆盖率 96.69%
- ✅ 登录自动跳转修复
- ✅ 模型容错加载
