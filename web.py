import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 修复NumPy bool弃用问题
if not hasattr(np, 'bool'):
    np.bool = bool

# 设置页面标题和布局
st.set_page_config(
    page_title="DKD Risk Prediction Model Based on Random Forest",
    page_icon="🏥",
    layout="wide"
)

# 定义全局变量
global feature_names, feature_dict, variable_descriptions

# 特征名称和描述 - DKD预测模型
feature_names = [
    'Smoking', 'Bcadmium', 'Uantimony', 'Ubarium', 'Uthallium',
    'PIR', 'BMI', 'ALB', 'Age', 'HbA1c'
]

feature_names_en = [
    'Smoking Status', 'Blood Cadmium', 'Urine Antimony', 'Urine Barium', 'Urine Thallium',
    'Poverty Income Ratio', 'Body Mass Index', 'Blood albumin', 'Age', 'Hemoglobin A1c'
]

feature_dict = dict(zip(feature_names, feature_names_en))

# 变量说明字典
variable_descriptions = {
    'Smoking': 'Smoking status (0=Never, 1=Former, 2=Current)',
    'Bcadmium': 'Blood cadmium concentration (μg/L)',
    'Uantimony': 'Urine antimony concentration (μg/L)',
    'Ubarium': 'Urine barium concentration (μg/L)',
    'Uthallium': 'Urine thallium concentration (μg/L)',
    'PIR': 'Poverty Income Ratio',
    'BMI': 'Body Mass Index (kg/m²)',
    'ALB': 'Blood albumin level (g/L)',
    'Age': 'Patient age in years',
    'HbA1c': 'Hemoglobin A1c (%)'
}

# 加载RF模型
@st.cache_resource
def load_model():
    # 加载随机森林模型
    model = joblib.load('RFmodel.dat')
    return model

# 主应用
def main():
    global feature_names, feature_dict, variable_descriptions
    
    # 侧边栏标题
    st.sidebar.title("DKD Risk Prediction Model Based on Random Forest")
    st.sidebar.image("https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg", width=200)
    
    # 添加系统说明到侧边栏
    st.sidebar.markdown("""
    # System Description

    ## About This System
    This is a Diabetic Kidney Disease (DKD) risk prediction system based on Random Forest algorithm, which predicts DKD risk by analyzing patient clinical indicators and environmental exposure factors.

    ## Prediction Results
    The system predicts:
    - DKD probability
    - No DKD probability
    - Risk assessment (low, medium, high risk)

    ## How to Use
    1. Fill in patient clinical indicators in the main interface
    2. Click the prediction button to generate prediction results
    3. View prediction results and feature importance analysis

    ## Important Notes
    - Please ensure accurate patient information input
    - All fields need to be filled
    - Numeric fields require number input
    - Selection fields require choosing from options
    """)
    
    # 添加变量说明到侧边栏
    with st.sidebar.expander("Variable Descriptions"):
        for feature in feature_names:
            st.markdown(f"**{feature_dict[feature]}**: {variable_descriptions[feature]}")
    
    # 主页面标题
    st.title("DKD Risk Prediction Model Based on Random Forest")
    st.markdown("### Diabetic Kidney Disease Risk Assessment")
    
    # 加载模型
    try:
        model = load_model()
        st.sidebar.success("RF Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Model loading failed: {e}")
        return
    
    # 创建输入表单
    st.sidebar.header("Patient Information Input")

    # 创建两列布局用于输入
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Information")
        age = st.number_input(f"{feature_dict['Age']} (years)", min_value=18, max_value=100, value=50)
        smoking = st.selectbox(f"{feature_dict['Smoking']}", options=[0, 1, 2], format_func=lambda x: "Never" if x == 0 else "Former" if x == 1 else "Current")
        bmi = st.number_input(f"{feature_dict['BMI']} (kg/m²)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        pir = st.number_input(f"{feature_dict['PIR']}", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
        hba1c = st.number_input(f"{feature_dict['HbA1c']} (%)", min_value=4.0, max_value=15.0, value=5.5, step=0.1)

    with col2:
        st.subheader("Laboratory Results")
        alb = st.number_input(f"{feature_dict['ALB']} (g/L)", min_value=1.0, max_value=200.0, value=40.0, step=0.1)
        bcadmium = st.number_input(f"{feature_dict['Bcadmium']} (μg/L)", min_value=0.0, max_value=50.0, value=2.5, step=0.01)
        uantimony = st.number_input(f"{feature_dict['Uantimony']} (μg/L)", min_value=0.0, max_value=1.0, value=0.05, step=0.001)
        ubarium = st.number_input(f"{feature_dict['Ubarium']} (μg/L)", min_value=0.0, max_value=100.0, value=2.0, step=0.01)
        uthallium = st.number_input(f"{feature_dict['Uthallium']} (μg/L)", min_value=0.0, max_value=2.0, value=0.2, step=0.001)
    
    # 创建预测按钮
    predict_button = st.button("Predict DKD Risk")

    if predict_button:
        # 收集所有输入特征 - 按照训练数据的顺序
        features = [smoking, bcadmium, uantimony, ubarium, uthallium, pir, bmi, alb, age, hba1c]

        # 转换为DataFrame
        input_df = pd.DataFrame([features], columns=feature_names)

        # 进行预测
        prediction = model.predict_proba(input_df)[0]
        no_dkd_prob = prediction[0]
        dkd_prob = prediction[1]
        
        # 显示预测结果
        st.header("DKD Prediction Results")

        # 使用进度条显示概率
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("No DKD Probability")
            st.progress(float(no_dkd_prob))
            st.write(f"{no_dkd_prob:.2%}")

        with col2:
            st.subheader("DKD Probability")
            st.progress(float(dkd_prob))
            st.write(f"{dkd_prob:.2%}")

        # 风险评估
        risk_level = "Low Risk" if dkd_prob < 0.3 else "Medium Risk" if dkd_prob < 0.6 else "High Risk"
        risk_color = "green" if dkd_prob < 0.3 else "orange" if dkd_prob < 0.6 else "red"

        st.markdown(f"### Risk Assessment: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
        
        # 临床建议
        st.header("Clinical Recommendations")
        st.write("Based on the DKD risk prediction, the following clinical recommendations are provided:")

        if dkd_prob > 0.5:  # 使用0.5作为阈值
            st.warning("""
            This patient has a high risk of developing DKD. Consider:
            - Enhanced diabetes management and glucose control
            - Regular monitoring of kidney function (eGFR, creatinine)
            - Blood pressure management (target <130/80 mmHg)
            - Lifestyle modifications (diet, exercise, smoking cessation)
            - Reducing environmental exposure to heavy metals
            - Regular albumin/protein monitoring
            - Consider ACE inhibitors or ARBs if appropriate
            """)
        else:
            st.success("""
            This patient has a relatively low risk of developing DKD. Standard care protocols are recommended:
            - Regular diabetes monitoring and management
            - Annual kidney function assessment
            - Maintain healthy lifestyle
            - Continue environmental exposure awareness
            """)
        
        # 添加模型解释
        st.write("---")
        st.subheader("Model Interpretation")

        try:
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            # 处理SHAP值格式 - 形状为(1, 10, 2)表示1个样本，10个特征，2个类别
            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                # 取第一个样本的正类（DKD类，索引1）的SHAP值
                shap_value = shap_values[0, :, 1]  # 形状变为(10,)
                expected_value = explainer.expected_value[1]  # 正类的期望值
            elif isinstance(shap_values, list):
                # 如果是列表格式，取正类的SHAP值
                shap_value = np.array(shap_values[1][0])
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_value = np.array(shap_values[0])
                expected_value = explainer.expected_value

            # 特征贡献分析表格
            st.subheader("Feature Contribution Analysis")

            # 创建贡献表格
            feature_values = []
            feature_impacts = []

            # 获取SHAP值
            for i, feature in enumerate(feature_names):
                feature_values.append(float(input_df[feature].iloc[0]))
                # SHAP值现在应该是一维数组
                impact_value = float(shap_value[i])
                feature_impacts.append(impact_value)

            shap_df = pd.DataFrame({
                'Feature': [feature_dict.get(f, f) for f in feature_names],
                'Value': feature_values,
                'Impact': feature_impacts
            })

            # 按绝对影响排序
            shap_df['Absolute Impact'] = shap_df['Impact'].abs()
            shap_df = shap_df.sort_values('Absolute Impact', ascending=False)

            # 显示表格
            st.table(shap_df[['Feature', 'Value', 'Impact']])
            
            # SHAP瀑布图
            st.subheader("SHAP Waterfall Plot")

            try:
                # 创建SHAP瀑布图
                fig_waterfall = plt.figure(figsize=(12, 8))

                # 使用新版本的waterfall plot
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_value,
                        base_values=expected_value,
                        data=input_df.iloc[0].values,
                        feature_names=[feature_dict.get(f, f) for f in feature_names]
                    ),
                    max_display=10,
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)
            except Exception as e:
                st.error(f"无法生成瀑布图: {str(e)}")
                # 使用条形图作为替代
                fig_bar = plt.figure(figsize=(10, 6))
                sorted_idx = np.argsort(np.abs(shap_value))[-10:]
                plt.barh(range(len(sorted_idx)), shap_value[sorted_idx])
                plt.yticks(range(len(sorted_idx)), [feature_dict.get(feature_names[i], feature_names[i]) for i in sorted_idx])
                plt.xlabel('SHAP Value')
                plt.title('Feature Impact on DKD Prediction')
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            # SHAP力图
            st.subheader("SHAP Force Plot")

            try:
                # 使用官方SHAP力图，HTML格式
                import streamlit.components.v1 as components

                force_plot = shap.force_plot(
                    expected_value,
                    shap_value,
                    input_df.iloc[0],
                    feature_names=[feature_dict.get(f, f) for f in feature_names]
                )

                # 获取SHAP的HTML内容，添加CSS来修复遮挡问题
                shap_html = f"""
                <head>
                    {shap.getjs()}
                    <style>
                        body {{
                            margin: 0;
                            padding: 20px 10px 40px 10px;
                            overflow: visible;
                        }}
                        .force-plot {{
                            margin: 20px 0 40px 0 !important;
                            padding: 20px 0 40px 0 !important;
                        }}
                        svg {{
                            margin: 20px 0 40px 0 !important;
                        }}
                        .tick text {{
                            margin-bottom: 20px !important;
                        }}
                        .force-plot-container {{
                            min-height: 200px !important;
                            padding-bottom: 50px !important;
                        }}
                    </style>
                </head>
                <body>
                    <div class="force-plot-container">
                        {force_plot.html()}
                    </div>
                </body>
                """

                # 增加更多高度空间
                components.html(shap_html, height=400, scrolling=False)

            except Exception as e:
                st.error(f"无法生成HTML力图: {str(e)}")
                st.info("请检查SHAP版本是否兼容")
            
        except Exception as e:
            st.error(f"无法生成SHAP解释: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.info("使用模型特征重要性作为替代")

            # 显示模型特征重要性
            st.write("---")
            st.subheader("Feature Importance")

            # 从随机森林模型获取特征重要性
            try:
                feature_importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': [feature_dict.get(f, f) for f in feature_names],
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False)

                fig, _ = plt.subplots(figsize=(10, 6))
                plt.barh(range(len(importance_df)), importance_df['Importance'], color='skyblue')
                plt.yticks(range(len(importance_df)), importance_df['Feature'])
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.title('Feature Importance')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e2:
                st.error(f"无法显示特征重要性: {str(e2)}")

if __name__ == "__main__":
    main()
