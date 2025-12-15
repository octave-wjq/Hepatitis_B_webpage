# 肝硬化CHE风险预测Web应用 - Development Plan

## Overview
基于 Streamlit 构建的肝硬化CHE风险预测系统,集成用户认证、SQLite持久化和机器学习推理能力。

## Task Breakdown

### Task 1: 数据库层与Schema
- **ID**: task-1
- **Description**: 实现 SQLite 数据库连接器、建表脚本(users/predictions表)、CRUD仓储层,确保线程安全
- **File Scope**: app/db.py, app/schema.sql, app/repo.py, tests/test_db.py
- **Dependencies**: None
- **Test Command**: `pytest tests/test_db.py --cov=app/db --cov=app/repo --cov-report=term --cov-fail-under=90`
- **Test Focus**:
  - 数据库连接初始化与关闭
  - users表CRUD操作(插入/查询/更新)
  - predictions表CRUD操作(插入/查询/分页)
  - 并发访问场景(多线程写入)
  - 异常处理(文件权限/SQL注入防护)

### Task 2: 认证与会话管理
- **ID**: task-2
- **Description**: 实现登录表单、会话状态管理、环境变量读取(AUTH_USER/AUTH_PASSWORD)、pbkdf2_hmac密码哈希验证
- **File Scope**: app/auth.py, app/main.py(认证部分), tests/test_auth.py
- **Dependencies**: depends on task-1
- **Test Command**: `pytest tests/test_auth.py --cov=app/auth --cov-report=term --cov-fail-under=90`
- **Test Focus**:
  - 环境变量读取与默认值处理
  - 密码哈希生成与验证(正确/错误密码)
  - 会话状态初始化与持久化
  - 登录表单渲染与提交逻辑
  - 未授权访问拦截

### Task 3: 推理集成
- **ID**: task-3
- **Description**: 提取预测逻辑为独立服务,修复 model/saved_models 路径依赖,集成数据库记录预测结果
- **File Scope**: app/predict.py, app/main.py(推理集成部分), tests/test_predict.py
- **Dependencies**: depends on task-1
- **Test Command**: `pytest tests/test_predict.py --cov=app/predict --cov-report=term --cov-fail-under=90`
- **Test Focus**:
  - 模型加载(路径解析/文件不存在处理)
  - 输入验证(缺失字段/类型错误/范围检查)
  - 推理结果格式(概率/风险等级)
  - 预测结果持久化到数据库
  - 异常场景(模型损坏/内存不足)

### Task 4: 部署配置
- **ID**: task-4
- **Description**: 完善 requirements.txt、编写 README.md(环境变量文档)、创建启动脚本、配置 Streamlit 服务器参数
- **File Scope**: requirements.txt, README.md, scripts/run.sh, .streamlit/config.toml
- **Dependencies**: depends on task-1, task-2, task-3
- **Test Command**: `bash -c "pip install -r requirements.txt && streamlit run app/main.py --server.headless true --server.port 8501 & sleep 5 && curl -f http://localhost:8501 && pkill -f streamlit"`
- **Test Focus**:
  - 依赖安装完整性(无版本冲突)
  - 启动脚本执行(环境变量注入/进程管理)
  - Streamlit 服务可访问性(端口监听/健康检查)
  - 配置文件加载(主题/端口/CORS设置)

## Acceptance Criteria
- [ ] 用户可通过环境变量配置的凭据登录系统
- [ ] 预测结果持久化到 SQLite 数据库(/data/app.db)
- [ ] 模型从 model/saved_models 正确加载并返回风险预测
- [ ] 所有单元测试通过
- [ ] 代码覆盖率 ≥90%
- [ ] 应用可通过 scripts/run.sh 一键启动
- [ ] README.md 包含完整的环境变量配置说明

## Technical Notes
- **架构决策**: 保持 Streamlit 单体架构,通过 app/ 包结构实现模块化,避免过度工程化
- **安全约束**: 密码使用 pbkdf2_hmac(sha256) 哈希,盐值固定(非生产级,需文档说明)
- **并发模型**: SQLite 默认序列化模式,写操作需加锁或使用 WAL 模式
- **路径依赖**: 模型路径使用相对于项目根目录的绝对路径解析,避免 cwd 依赖
- **测试策略**: Task 1/3 使用 pytest + 内存数据库,Task 2 使用 mock 环境变量,Task 4 使用集成测试
- **并行执行**: Task 1 和 Task 4(文档部分)可并行,Task 2/3 需等待 Task 1 完成
