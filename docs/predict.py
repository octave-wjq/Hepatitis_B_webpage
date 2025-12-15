import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# ============================================
# 1. 页面配置与加载资源
# ============================================
st.set_page_config(page_title="肝硬化CHE风险预测系统", layout="wide")

st.title("🏥 肝硬化患者 CHE 风险预测系统")
st.markdown("基于多模态机器学习模型的临床决策支持工具")

# 【修改1】删除了这里的 plt.savefig，因为它不属于网页运行逻辑

# 加载模型和工具的函数（使用缓存加快速度）
@st.cache_resource
def load_assets():
    # 确保你的 saved_models 文件夹和 app.py 在同一目录下
    models = {
        'Logistic Regression': joblib.load('saved_models/log_reg.pkl'),
        'Random Forest': joblib.load('saved_models/rf.pkl'),
        'MLP': joblib.load('saved_models/mlp.pkl'),
        'SVM': joblib.load('saved_models/svm.pkl'),
        'XGBoost': joblib.load('saved_models/xgb.pkl'),
        'GBM': joblib.load('saved_models/lgb.pkl')
    }
    scaler = joblib.load('saved_models/scaler.pkl')
    encoder = joblib.load('saved_models/encoder.pkl')
    val_cols = joblib.load('saved_models/val_cols.pkl')
    cat_cols = joblib.load('saved_models/cat_cols.pkl')
    model_features = joblib.load('saved_models/model_features.pkl')
    return models, scaler, encoder, val_cols, cat_cols, model_features

# 加载资源
try:
    models, scaler, encoder, val_cols, cat_cols, model_features = load_assets()
except FileNotFoundError:
    st.error("错误：找不到模型文件。请确保 'saved_models' 文件夹存在且包含 .pkl 文件。")
    st.stop()

# ============================================
# 2. 侧边栏：输入患者信息
# ============================================
st.sidebar.header("输入患者临床参数")

def user_input_features():
    inputs = {}

    # 数值型变量输入
    st.sidebar.subheader("数值指标")
    inputs['Blood Ammonia'] = st.sidebar.number_input("Blood Ammonia (血氨)", min_value=0.0, value=50.0)
    inputs['Albumin'] = st.sidebar.number_input("Albumin (白蛋白)", min_value=0.0, value=35.0)

    # 分类变量输入
    st.sidebar.subheader("临床特征")
    # 假设 0代表无/No，1代表有/Yes
    inputs['Tips'] = st.sidebar.selectbox("Tips", options=[0, 1])
    inputs['HBV'] = st.sidebar.selectbox("HBV Infection", options=[0, 1])
    inputs['Splenomegaly'] = st.sidebar.selectbox("Splenomegaly (脾肿大)", options=[0, 1])
    inputs['History of Hepatic Encephalopathy'] = st.sidebar.selectbox("History of HE (肝性脑病史)", options=[0, 1])

    return pd.DataFrame([inputs])

input_df = user_input_features()

# 展示输入数据
st.subheader("1. 患者当前参数")
st.dataframe(input_df)

# ============================================
# 3. 数据预处理与预测
# ============================================
if st.button("开始预测", type="primary"):
    try:
        # 1. 分离数值和分类
        input_val = input_df[val_cols]
        input_cat = input_df[cat_cols]

        # 2. 标准化数值特征
        input_val_scaled = scaler.transform(input_val)
        input_val_df = pd.DataFrame(input_val_scaled, columns=val_cols)

        # 3. 独热编码分类特征
        input_cat_encoded = encoder.transform(input_cat)
        input_cat_df = pd.DataFrame(input_cat_encoded, columns=encoder.get_feature_names_out(cat_cols))

        # 4. 拼接
        final_input = pd.concat([input_val_df, input_cat_df], axis=1)

        # 5. 补齐缺失列 (对齐 XGBoost/GBM 特征)
        for col in model_features:
            if col not in final_input.columns:
                final_input[col] = 0
        final_input = final_input[model_features]

        # 转换类型
        final_input = final_input.astype(float)

        # ============================================
        # 4. 模型预测与展示
        # ============================================
        st.subheader("2. 风险预测结果")

        col1, col2, col3 = st.columns(3)

        # 定义卡片展示函数
        def show_prediction(model_name, model, col):
            if model_name == 'GBM':
                prob = model.predict(final_input)[0]
            else:
                prob = model.predict_proba(final_input)[0][1]

            risk_percent = prob * 100

            # 颜色逻辑
            color = "green" if risk_percent < 30 else "orange" if risk_percent < 70 else "red"

            col.markdown(f"""
            <div style="padding:10px; border-radius:10px; border:1px solid #ddd; text-align:center; background-color: #f9f9f9;">
                <h4 style="margin:0; color: #333;">{model_name}</h4>
                <h2 style="color:{color}; margin:10px 0;">{risk_percent:.1f}%</h2>
                <p style="margin:0; color: #666;">患病概率</p>
            </div>
            """, unsafe_allow_html=True)

        # 展示主要模型
        show_prediction('Random Forest', models['Random Forest'], col1)
        show_prediction('XGBoost', models['XGBoost'], col2)
        show_prediction('GBM', models['GBM'], col3)

        # 折叠面板展示所有详情
        with st.expander("查看所有模型预测详情"):
            all_probs = {}
            for name, model in models.items():
                if name == 'GBM':
                    p = model.predict(final_input)[0]
                else:
                    p = model.predict_proba(final_input)[0][1]
                all_probs[name] = f"{p*100:.2f}%"

            st.table(pd.DataFrame(list(all_probs.items()), columns=['模型名称', '预测概率']))

    except Exception as e:
        st.error(f"预测过程中发生错误: {e}")
        st.info("请检查 saved_models 文件是否完整，或输入数据是否异常。")

# ============================================
# 5. 模型性能展示 (静态图)
# ============================================
st.markdown("---")
st.subheader("3. 模型性能评估 (Nature Style)")

tab1, tab2, tab3, tab4 = st.tabs(["ROC 曲线", "校准曲线", "DCA 决策曲线", "PR 曲线"])

# 【修改2】确保这里读取的是 .png 文件，而不是 .pdf
# 请确保你的 "模型对比" 文件夹里有这些 png 图片
# 如果图片不存在，会显示错误提示

with tab1:
    try:
        st.image("模型对比/roc_curves_test_nature.png", caption="测试集 ROC 曲线")
    except:
        st.warning("未找到图片: roc_curves_test_nature.png (请先在 Notebook 中保存为 PNG)")

with tab2:
    try:
        st.image("模型对比/calibration_curves_test_nature.png", caption="测试集 校准曲线")
    except:
        st.warning("未找到图片: calibration_curves_test_nature.png")

with tab3:
    try:
        st.image("模型对比/dca_curves_test_nature.png", caption="测试集 DCA 决策曲线")
    except:
        st.warning("未找到图片: dca_curves_test_nature.png")

with tab4:
    try:
        st.image("模型对比/pr_curves_test.png", caption="测试集 PR 曲线")
    except:
        st.warning("未找到图片: pr_curves_test.png")