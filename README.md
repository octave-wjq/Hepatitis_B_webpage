# CHE Risk Prediction Web App
Streamlit 应用, 提供肝硬化 CHE 风险预测、轻量级登录与 SQLite 持久化。

## 快速开始
1. 准备 Python 3.9+ 环境, 安装依赖: `pip install -r requirements.txt`
2. 确保模型文件已存在于 `model/saved_models/*.pkl` (仓库已包含示例导出, 缺失会导致推理失败)。
3. 配置环境变量:
   - `AUTH_USER` / `AUTH_PASSWORD`: 登录用户名/密码, 默认 `admin` / `changeit`
   - `APP_DB_PATH`: SQLite 路径, 默认 `/data/app.db` (`scripts/run.sh` 会自动创建目录)
4. 启动应用:
   - 推荐: `./scripts/run.sh` (注入环境变量、创建数据库目录、健康检查后前台等待)
   - 直接运行: `streamlit run app/main.py --server.headless true --server.port 8501`
5. 浏览器访问 `http://localhost:8501` 登录并进行预测。

## 数据库与并发
- 默认数据库位于 `/data/app.db`, 单机演示友好; SQLite 采用文件锁, 多写并发和多实例场景会被阻塞。
- 需要更高并发或多节点部署时, 优先迁移到 PostgreSQL(或兼容的托管服务): 用等效的连接器替换 `app/db.py` 中的 SQLite 初始化, 更新仓储层使用 PostgreSQL 客户端(如 `psycopg2`/SQLAlchemy), 并通过环境变量提供 DSN。

## 配置说明
- Streamlit 服务器配置位于 `.streamlit/config.toml` (端口 8501、启用 headless、CORS 关闭、浅色主题)。
- 运行脚本会使用当前环境值启动, 如需覆盖端口可设置 `STREAMLIT_PORT` 再执行脚本。

## 测试
在具备依赖的环境中运行:  
`bash -c "pip install -r requirements.txt && streamlit run app/main.py --server.headless true --server.port 8501 & sleep 5 && curl -f http://localhost:8501 && pkill -f streamlit"`
