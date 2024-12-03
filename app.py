# Run the Streamlit app with `streamlit run app.py`
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from datetime import date
from sklearn.preprocessing import OneHotEncoder
from PyALE import ale
import random
import matplotlib.transforms as mtrans
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import t
def quantile_ied(x_vec, q):
    x_vec = x_vec.sort_values()
    n = len(x_vec) - 1
    m = 0
    j = (n * q + m).astype(int)  # location of the value
    g = n * q + m - j

    gamma = (g != 0).astype(int)
    quant_res = (1 - gamma) * x_vec.shift(1, fill_value=0).iloc[j] + gamma * x_vec.iloc[
        j
    ]
    quant_res.index = q
    # add min at quantile zero and max at quantile one (if needed)
    if 0 in q:
        quant_res.loc[0] = x_vec.min()
    if 1 in q:
        quant_res.loc[1] = x_vec.max()
    return quant_res

def CI_estimate(x_vec, C=0.95):
    alpha = 1 - C
    n = len(x_vec)
    stand_err = x_vec.std() / np.sqrt(n)
    critical_val = 1 - (alpha / 2)
    z_star = stand_err * t.ppf(critical_val, n - 1)
    return z_star
def aleplot_1D_continuous(X, model, feature, grid_size=20, include_CI=True, C=0.95):
    quantiles = np.linspace(0, 1, grid_size + 1, endpoint=True)
    bins = [X[feature].min()] + quantile_ied(X[feature].squeeze(), quantiles).to_list()
    bins = np.unique(bins)
    feat_cut = pd.cut(X[feature], bins, include_lowest=True)

    bin_codes = feat_cut.cat.codes
    bin_codes_unique = np.unique(bin_codes)

    X1 = X.copy()
    X2 = X.copy()
    X1[feature] = [bins[i] for i in bin_codes]
    X2[feature] = [bins[i + 1] for i in bin_codes]
    try:
        y_1 = model.predict(X1).ravel()
        y_2 = model.predict(X2).ravel()
    except Exception as ex:
        raise Exception(
            "Please check that your model is fitted, and accepts X as input."
        )

    delta_df = pd.DataFrame({feature: bins[bin_codes + 1], "Delta": y_2 - y_1})
    res_df = delta_df.groupby([feature], observed=False).Delta.agg(
        [("eff", "mean"), "size"]
    )
    res_df["eff"] = res_df["eff"].cumsum()
    res_df.loc[min(bins), :] = 0
    # subtract the total average of a moving average of size 2
    mean_mv_avg = (
        (res_df["eff"] + res_df["eff"].shift(1, fill_value=0)) / 2 * res_df["size"]
    ).sum() / res_df["size"].sum()
    res_df = res_df.sort_index().assign(eff=res_df["eff"] - mean_mv_avg)
    if include_CI:
        ci_est = delta_df.groupby(feature, observed=False).Delta.agg(
            [("CI_estimate", lambda x: CI_estimate(x, C=C))]
        )
        ci_est = ci_est.sort_index()
        lowerCI_name = "lowerCI_" + str(int(C * 100)) + "%"
        upperCI_name = "upperCI_" + str(int(C * 100)) + "%"
        res_df[lowerCI_name] = res_df[["eff"]].subtract(ci_est["CI_estimate"], axis=0)
        res_df[upperCI_name] = res_df[["eff"]].add(
            ci_est["CI_estimate"], axis=0
        )
    return res_df

def plot_ale_c(X, res_df):
    """Plot ALE for a given feature and return the figure."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('black')  # 设置整个图表背景为黑色
    ax.set_facecolor('black')  # 设置子图背景为黑色
    
    feature_name = res_df.index.name
    # position: jitter
    # to see the distribution of the data points clearer, each point x will be nudged a random value between
    # -minimum distance between two data points and +
    sorted_values = X[feature_name].sort_values()
    values_diff = abs(sorted_values.shift() - sorted_values)
    np.random.seed(123)
    rug = X.apply(
        lambda row: row[feature_name]
        + np.random.uniform(
            -values_diff[values_diff > 0].min() / 2,
            values_diff[values_diff > 0].min() / 2,
        ),
        axis=1,
    )
    
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res_df[["eff"]], color='gold')  # 设置折线为金色
    tr = mtrans.offset_copy(ax.transData, fig=fig, x=0.0, y=-5, units="points")
    ax.plot(
        rug,
        [res_df.drop("size", axis=1).min().min()] * len(rug),
        "|",
        color="k",
        alpha=0.2,
        transform=tr,
    )
    lowerCI_name = res_df.columns[res_df.columns.str.contains("lowerCI")]
    upperCI_name = res_df.columns[res_df.columns.str.contains("upperCI")]
    if (len(lowerCI_name) == 1) and (len(upperCI_name) == 1):
        label = lowerCI_name.str.split("_")[0][1] + " confidence interval"
        ax.fill_between(
            res_df.index,
            y1=res_df[lowerCI_name[0]],
            y2=res_df[upperCI_name[0]],
            alpha=0.5,
            color="#A8A8A8",  # 设置置信区间为灰色
            label=label,
        )
        legend = ax.legend()
        legend.get_frame().set_facecolor('black')  # 设置图例背景为黑色
        for text in legend.get_texts():
            text.set_color('white')  # 设置图例文字为白色
    ax.set_xlabel(res_df.index.name, color='white')  # 设置x轴标签为白色
    ax.set_ylabel("Effect on prediction (centered)", color='white')  # 设置y轴标签为白色
    ax.set_title("1D ALE Plot - Continuous", color='white')  # 设置标题为白色
    
    # 设置刻度和边框颜色为白色
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    
    return fig

def plot_ale_d(X, res_df):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('black')  # 设置整个图表背景为黑色
    ax.set_facecolor('black')  # 设置子图背景为黑色
    
    feature_name = res_df.index.name
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.set_xlabel(feature_name, color='white')  # 设置x轴标签为白色
    ax.set_ylabel("Effect on prediction (centered)", color='white')  # 设置y轴标签为白色
    yerr = 0
    lowerCI_name = res_df.columns[res_df.columns.str.contains("lowerCI")]
    upperCI_name = res_df.columns[res_df.columns.str.contains("upperCI")]
    if (len(lowerCI_name) == 1) and (len(upperCI_name) == 1):
        yerr = res_df[upperCI_name].subtract(res_df["eff"], axis=0).iloc[:, 0]
    
    ax.errorbar(
        res_df.index.astype(str),
        res_df["eff"],
        yerr=yerr,
        capsize=3,
        marker="o",
        linestyle="dashed",
        color="gold",  # 设置折线颜色为金色
        ecolor="darkgrey",  # 设置置信区间颜色为灰色
    )
    
    ax2 = ax.twinx()
    ax2.set_ylabel("Size", color="lightblue")
    ax2.bar(res_df.index.astype(str), res_df["size"], alpha=0.1, align="center", color='lightblue')
    ax2.tick_params(axis="y", labelcolor="lightblue")
    ax2.set_title("1D ALE Plot - Discrete/Categorical", color='white')  # 设置标题为白色
    
    # 设置刻度和边框颜色为白色
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    
    # 设置第二个y轴的边框颜色为白色
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    
    fig.tight_layout()    
    return fig
# Set Streamlit page configuration
st.set_page_config(page_title="Operation Centre for the Analysis of Obesity Data", layout="wide")

def display_model_performance(r2_adj):
    if r2_adj > 0.7:
        st.sidebar.write(f'调整后的 R^2: {r2_adj:.3f} - 模型拟合良好')
    elif 0.4 <= r2_adj <= 0.7:
        st.sidebar.write(f'调整后的 R^2: {r2_adj:.3f} - 模型拟合一般')
    else:
        st.sidebar.write(f'调整后的 R^2: {r2_adj:.3f} - 模型拟合差，请重新选择解释变量')

def diaconis_freeman_bins(data):
    # 计算 IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # 计算数据点的数量
    n = len(data)
    
    # 计算 bin 大小
    bin_width = (2 * IQR) / (n ** (1 / 3))
    
    # 计算 bin 数量
    data_range = np.max(data) - np.min(data)
    num_bins = int(np.ceil(data_range / bin_width))
    
    return bin_width, num_bins


# 缓存数据加载和预处理
@st.cache_data
def load_data():
    # 这里导入了数据库的数据集，里面应该有很多NAN  
    rawdata = pd.read_excel("D3_15.xlsx")
    colsel = ["Patient_ID", "BMI", "Father_BMI", "Mother_BMI", "Sex", "Age", "Waist_Ht", "SBP", "DBP", "TG", "Chol"]
    mydata = rawdata[colsel].copy()  # Ensure this is a separate copy
    mydata.columns = ["ID", "BMI", "FBMI", "MBMI", "GENDER", "AGE", "WAIST", "SBP", "DBP", "TG", "CHOL"]

    # Convert non-numeric values to NaN
    mydata = mydata.apply(pd.to_numeric, errors='coerce')
    mydata['ID'] = pd.to_numeric(mydata['ID'], errors='coerce').astype('Int64')

    mydata2 = mydata.copy()
    mydata2["GENDER"] = mydata2["GENDER"].replace({1: 'MALE', 2: 'FEMALE'})

    mydata = mydata.dropna().reset_index(drop=True)
    mydata2 = mydata2.dropna().reset_index(drop=True)
    for col in [ 'FBMI', 'MBMI', 'AGE', 'WAIST', 'SBP', 'DBP', 'TG', 'CHOL']:#'BMI',
        mydata2.loc[mydata2.sample(frac=0.2).index, col] = np.nan

    return mydata, mydata2

mydata, mydata2 = load_data()

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    /* 全局背景和文本颜色设置 */
    body {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* 主要容器的样式 */
    .main {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* 侧边栏的样式 */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* 侧边栏内所有元素的样式 */
    [data-testid="stSidebar"] .css-1d391kg, 
    [data-testid="stSidebar"] .css-18e3th9, 
    [data-testid="stSidebar"] .css-1lcbmhc, 
    [data-testid="stSidebar"] .css-1r6slb0,
    [data-testid="stSidebar"] .css-1v3fvcr, 
    [data-testid="stSidebar"] .css-1y4p8pa, 
    [data-testid="stSidebar"] .css-1aumxhk, 
    [data-testid="stSidebar"] .css-1cpxqw2 {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* 侧边栏内悬停效果 */
    [data-testid="stSidebar"] .css-1d391kg:hover, 
    [data-testid="stSidebar"] .css-18e3th9:hover, 
    [data-testid="stSidebar"] .css-1lcbmhc:hover, 
    [data-testid="stSidebar"] .css-1r6slb0:hover, 
    [data-testid="stSidebar"] .css-1v3fvcr:hover, 
    [data-testid="stSidebar"] .css-1y4p8pa:hover, 
    [data-testid="stSidebar"] .css-1aumxhk:hover, 
    [data-testid="stSidebar"] .css-1cpxqw2:hover {
        color: #dddddd !important;
    }

    /* 按钮样式 */
    .stButton button {
        background-color: #ffffff !important;
        color: #000000 !important; /* 按钮文本颜色 */
    }

    /* 按钮悬停样式 */
    .stButton button:hover {
        background-color: #dddddd !important;
        color: #000000 !important; /* 按钮文本悬停颜色 */
    }

    /* 表格的样式 */
    .stTable thead th, .stTable tbody td {
        color: #ffffff !important;
    }

    /* 链接样式 */
    a {
        color: #ffffff !important;
    }

    a:hover {
        color: #dddddd !important;
    }

    /* 标题样式 */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* 特定Streamlit标题样式 */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff !important;
    }

    /* 额外的Streamlit标题样式 */
    .css-10trblm, .css-1v3fvcr, .css-1aumxhk, .css-1d391kg {
        color: #ffffff !important;
    }

    /* 选择框和多选框的提示文本字体颜色 */
    .stSelectbox label,
    .stMultiselect label,
    .stTextInput label {
        color: #ffffff !important;
    }

    /* 多选框中的项目字体颜色 */
    .stMultiSelect div,
    .stMultiSelect span {
        color: #ffffff !important;
    }

    /* 多选框的提示文本字体颜色 */
    .stMultiSelect label {
        color: #ffffff !important;
    }

    /* 输入框中的文本颜色 */
    .stTextInput div,
    .stTextInput input {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Operation Centre")
page = st.sidebar.selectbox("Choose your page", ["Operation center", "Patient appointment"])

# Main app
if page == "Operation center":
    st.title("Operation Centre for the Analysis of Obesity Data")

    # Inputs
    with st.sidebar:
        st.header("Inputs")
        if 'response_var' not in st.session_state:
            st.session_state['response_var'] = mydata.columns[1]
        if 'explanatory_vars' not in st.session_state:
            st.session_state['explanatory_vars'] = list(mydata.columns[1:])
        
        response_var = st.selectbox("Select the response", mydata.columns[1:], index=mydata.columns.get_loc(st.session_state['response_var']) - 1)
        tt = mydata.drop(columns=["ID"])
        tt = tt.drop(columns=response_var)
        explanatory_vars = st.multiselect("Select the explanatory variable(s)", tt.columns)#, default=st.session_state['explanatory_vars']
        n = st.number_input('选择数据集数量', min_value=1, max_value=500, value=100)
        start_button = st.button("Start")
        
    if start_button or 'xgb_model' in st.session_state:# 
        if start_button:   
            # 将选择的变量和模型状态保存到session state中
            st.session_state['response_var'] = response_var
            st.session_state['explanatory_vars'] = explanatory_vars
            st.session_state['n'] = n
            
            filtered_data = mydata2[["ID"] + [response_var] + st.session_state['explanatory_vars']]
            # filtered_data = filtered_data.dropna().reset_index(drop=True)
            # 如果可用数据的数量大于等于 n，随机选择 n 个患者
            if len(filtered_data) >= n:
                data = filtered_data.sample(n=n, random_state=0).reset_index(drop=True)
            else:
                st.warning(f'数据集中有效记录少于 {n} 个患者，请选择更小的 n 值。')
            st.session_state['data'] = data
            data = mydata2
            X = data.drop(columns=["ID", response_var])
            y = data[response_var]
            if "GENDER" in X.columns:               
                one_hot_encoder = OneHotEncoder().fit(X[["GENDER"]])
                def onehot_encode(feat, ohe=one_hot_encoder):
                    col_names = ohe.categories_[0]
                    feat_coded = pd.DataFrame(ohe.transform(feat).toarray())
                    feat_coded.columns = col_names
                    return feat_coded

                coded_feature = onehot_encode(X[["GENDER"]], one_hot_encoder)
                X = pd.concat([X, coded_feature], axis=1)
                features = [x for x in st.session_state['explanatory_vars'] if x != "GENDER"]
                features = features + coded_feature.columns.to_list()
            else:
                features = [x for x in X.columns if x != "GENDER"]
            xgb_model = XGBRegressor(objective='reg:squarederror',n_estimators = 300, max_depth=4, learning_rate= 0.2, seed = 42, verbose= True)            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            eval_set = [(X_train[features], y_train), (X_val[features], y_val)]
            xgb_model.fit(X_train[features], y_train, eval_set=eval_set, verbose=False)
            y_pred = xgb_model.predict(X_val[features])
            r2 = r2_score(y_val, y_pred)
            n = len(y_val)
            p = len(features)
            r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            display_model_performance(r2_adj)
            y_pred2 = xgb_model.predict(X[features])
            st.session_state['xgb_model'] = xgb_model
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['features'] = features
            if "GENDER" in X.columns: 
                st.session_state['one_hot_encoder'] = one_hot_encoder
                st.session_state['coded_feature'] = coded_feature
        xgb_model = st.session_state['xgb_model']
        X = st.session_state['X']
        y = st.session_state['y']
        features = st.session_state['features']
        if "GENDER" in X.columns: 
            one_hot_encoder = st.session_state['one_hot_encoder']
            coded_feature = st.session_state['coded_feature']
            def onehot_encode(feat, ohe=one_hot_encoder):
                col_names = ohe.categories_[0]
                feat_coded = pd.DataFrame(ohe.transform(feat).toarray())
                feat_coded.columns = col_names
                return feat_coded
        
        col1, col2, col3 = st.columns(3)
        
        # with col1:
        #     # 数据描述按性别
        #     st.header("Data Description by Gender")
        #     available_vars = [var for var in st.session_state['explanatory_vars'] if var !="GENDER"]

        #     # 检查并设置默认值
        #     if 'sum_var' not in st.session_state or st.session_state['sum_var'] not in available_vars:
        #         st.session_state['sum_var'] = available_vars[0] if available_vars else None

        #     if available_vars:
        #         sum_var_default_index = available_vars.index(st.session_state['sum_var']) if st.session_state['sum_var'] in available_vars else 0
        #         sum_var = st.selectbox("Select the variable to be described", available_vars, index=sum_var_default_index)

        #         # Update session state after selection
        #         st.session_state['sum_var'] = sum_var

        #         if sum_var is not None:
        #             # melted_data = mydata2.melt(id_vars=['GENDER'], value_vars=sum_var)
        #             data = st.session_state['data']
        #             melted_data = data.melt(id_vars=['GENDER'], value_vars=sum_var)

        #             # 创建子图
        #             fig, axes = plt.subplots(1, 2, figsize=(5, 5), sharey=True)
        #             fig.patch.set_facecolor('black')  # 设置整个图表背景为黑色

        #             # 标签颜色设置函数
        #             def set_labels_color(ax):
        #                 ax.title.set_color('white')
        #                 ax.xaxis.label.set_color('white')
        #                 ax.yaxis.label.set_color('white')
        #                 ax.tick_params(axis='x', colors='white')
        #                 ax.tick_params(axis='y', colors='white')
        #                 ax.spines['top'].set_color('white')
        #                 ax.spines['right'].set_color('white')
        #                 ax.spines['bottom'].set_color('white')
        #                 ax.spines['left'].set_color('white')
        #                 ax.set_facecolor('black')  # 设置子图背景为黑色

        #             # 绘制男性数据直方图
        #             male_data = melted_data[melted_data['GENDER'] == 'MALE']
        #             sns.histplot(
        #                 data=male_data,
        #                 x='value',
        #                 hue='GENDER',
        #                 multiple='dodge',
        #                 ax=axes[0],
        #                 palette=['gray']
        #             )
        #             male_mean = male_data['value'].mean()
        #             axes[0].axvline(male_mean, color='gold', linestyle='--', linewidth=2, label=f'Mean: {male_mean:.2f}')
        #             axes[0].legend().get_texts()[0].set_color('white')
        #             set_labels_color(axes[0])
        #             axes[0].set_title('MALE')

        #             # 绘制女性数据直方图
        #             female_data = melted_data[melted_data['GENDER'] == 'FEMALE']
        #             sns.histplot(
        #                 data=female_data,
        #                 x='value',
        #                 hue='GENDER',
        #                 multiple='dodge',
        #                 ax=axes[1],
        #                 palette=['gray']
        #             )
        #             female_mean = female_data['value'].mean()
        #             axes[1].axvline(female_mean, color='gold', linestyle='--', linewidth=2, label=f'Mean: {female_mean:.2f}')
        #             axes[1].legend().get_texts()[0].set_color('white')
        #             set_labels_color(axes[1])
        #             axes[1].set_title('FEMALE')

        #             # 调整布局
        #             plt.tight_layout()
        #         else:
        #             st.write("No variable selected for description.")
        #         st.pyplot(fig)
        #     else:
        #         st.write("No options to select.")

        # with col2:
        #     # Communication with AI doctor
        #     st.header("Communication with AI doctor")
        #     # 加载本地图像
        #     image_path = "chatbot.png"  # 替换为你的图像路径
        #     image = Image.open(image_path)
        #     # 显示图像
        #     st.image(image, use_column_width=True)
        #     question = st.text_input("Enter your question:")
        #     if st.button("Ask"):
        #         st.write("The answer is: Hello, I am an AI doctor, you can ask me any questions about obesity.")

        # # XGBoost model
        # # Here because gender is a catogarical variable, when using one-hot, it will become FEMALE and MALE,
        # # The importance will only compute for FEMALE, so I change FEMALE to GENDER.
        # # Global effect: feature importance
        # with col3:
        #     st.header("Global Effect: Feature Importance")
        #     importance = xgb_model.get_booster().get_score(importance_type='gain')
        #     if "GENDER" in X.columns: 
        #         importance_df = pd.DataFrame({'Feature': list(importance.keys())[0:-1]+["GENDER"], 'Importance': list(importance.values())})
        #     else:
        #         importance_df = pd.DataFrame({'Feature': list(importance.keys()), 'Importance': list(importance.values())})
        #     importance_df = importance_df.sort_values(by='Importance', ascending=True)
        #     most_fea = importance_df.iloc[-1]["Feature"]
        #     most_imp = importance_df.iloc[-1]["Importance"]
        #     fig = px.bar(importance_df, x='Importance', y='Feature', title='Feature Importance')
        #     # 更新图表布局和样式
        #     fig.update_layout(
        #         plot_bgcolor='black',    # 图表背景颜色
        #         paper_bgcolor='black',   # 纸张背景颜色
        #         font=dict(color='white'), # 文本颜色
        #         title=dict(font=dict(color='white')), # 标题颜色
        #         xaxis=dict(title=dict(font=dict(color='white')), tickfont=dict(color='white')), # X轴颜色
        #         yaxis=dict(title=dict(font=dict(color='white')), tickfont=dict(color='white'))  # Y轴颜色
        #     )

        #     # 更新条形颜色
        #     fig.update_traces(marker_color='gold')  # 条形颜色为金色
        #     fig.update_layout(height=350)  # 设置图的高度和数据描述的图一样
        #     st.plotly_chart(fig)
        #     imp_summary_text = f"""
        #         <p>Feature importance tells us how much each explanatory variable considered in the model contributes to predicting {response_var}. Based on the feature importance shown in the graph, {most_fea} is the largest contributor in predicting the response with a feature importance score of {most_imp:.2f}.</p>
        #         <p>Important remark: high feature importance does not imply causal relationship. Further investigation would be required to assess potential causal relationship between a feature and the response.</p>
        #         """
        #     st.markdown(imp_summary_text, unsafe_allow_html=True)

        col4, col5, col6 = st.columns(3)
        with col4:
            # Global effect: ALE plot
            st.header("Global Effect: ALE Plot")
            ale_var = st.selectbox("Select one variable", st.session_state['explanatory_vars'], key="ale_var")
            # ALE plot implementation can be complex; here we use a simple plot for illustration
            #fig, ax = plt.subplots()
            if ale_var=="GENDER":
                X_feat_raw = X.drop(coded_feature.columns.to_list(), axis=1, inplace=False).copy()
                ale_eff = ale(
                X=X_feat_raw,
                model=xgb_model,
                feature=[ale_var],
                encode_fun=onehot_encode,
                predictors=features,plot=False)
                fig = plot_ale_d(X,ale_eff) 
                
            else:
                random.seed(123)
                X_sample = X[features].loc[random.sample(X.index.to_list(), n//3), :]
                X_sample2 = X_sample.dropna(subset=[ale_var]) 
                # print(X_sample2)
                # ale_eff = aleplot_1D_continuous(X=X_sample2, model=xgb_model, feature=ale_var, grid_size=20, include_CI=True, C=0.95)
                rate = X_sample2.size/X_sample.size  
                
                # 使用 Diaconis-Freeman 方法计算 bin 大小和数量
                bin_width, num_bins = diaconis_freeman_bins(X_sample2[ale_var])  
                print(f"Bin Width: {bin_width}")
                print(f"Number of Bins: {num_bins}")         
                ale_eff = ale(
                X=X_sample2, model=xgb_model, feature=[ale_var], grid_size=200, include_CI=True, C=0.95,plot=False)
                fig = plot_ale_c(X,ale_eff) 
                print(ale_eff['lowerCI_95%']) 
                print(ale_eff['upperCI_95%']) 
                print(ale_eff['eff'])
            # st.pyplot(fig)
            
            ale_summary_text = f"""
                <p>Accumulated Local Effects (ALE) plot is a visualization tool for interpreting machine learning models. It helps us visualise the shape of the potential relationship between {ale_var} and {response_var}. The vertical axis of the ALE plot represents the change in {response_var}, and the horizontal axis represents the value taken by {ale_var}.</p>
                <p>Note that due to missing data, we only used {rate} of the data for plotting.</p>
                """
            st.markdown(ale_summary_text, unsafe_allow_html=True)
            
        # with col5:
        #     # Local effect: SHAP plot
        #     st.header("Local Effect: SHAP Plot")
        #     shap_id = st.selectbox("Select a patient by patient ID", mydata2["ID"])

        #     explainer = shap.TreeExplainer(xgb_model)
        #     shap_values = explainer.shap_values(X[features])
        #     exp = explainer.expected_value
        #     # print(exp)
        #     preds = xgb_model.predict(X[features])
            
        #     indice = np.where(mydata2["ID"].values == shap_id)
        #     pred_pa = preds[indice[0]]

        #     if "MALE" in X[features].columns:
        #         variables = list(X[features].columns[:-2])+["GENDER"]
        #         shap_values2 = shap_values[indice[0],:-1]
        #         shap_values2 = shap_values2.flatten()
        #     else:
        #         variables = list(X[features].columns)
        #         shap_values2 = shap_values[indice[0],:]
        #         shap_values2 = shap_values2.flatten()
        #     # 创建图表
        #     fig, ax = plt.subplots(figsize=(5, 5))

        #     # 设置背景颜色为黑色
        #     fig.patch.set_facecolor('black')
        #     ax.set_facecolor('black')

        #     # 绘制每个变量的线和点
        #     for i, (var, shap_value) in enumerate(zip(variables, shap_values2)):
        #         ax.plot([0, shap_value], [i, i], color='white', solid_capstyle='butt')
        #         ax.plot(shap_value, i, 'o', color='gold')

        #     # 设置y轴
        #     ax.set_yticks(range(len(variables)))
        #     ax.set_yticklabels(variables, color='white')

        #     # 设置x轴
        #     ax.tick_params(axis='x', colors='white')

        #     # 设置其他文字为白色
        #     ax.xaxis.label.set_color('white')
        #     ax.yaxis.label.set_color('white')
        #     ax.title.set_color('white')

        #     # 移除顶部和右侧边框
        #     ax.spines['top'].set_visible(False)
        #     ax.spines['right'].set_visible(False)
        #     ax.spines['left'].set_color('white')
        #     ax.spines['bottom'].set_color('white')

        #     st.pyplot(fig)

        # with col6:
        #     # AI prediagnosis
        #     st.header("AI Prediagnosis")
        #     shap_contributions = np.round(shap_values2,3)#np.round(np.array(shap_values)[indice[0],:],3)
        #     shap_features = variables#X[features].columns
        #     largest_shap = shap_features[np.argmax(shap_contributions)]
            
        #     if shap_contributions.max() < 0:
        #         summary_text = f"""
        #         <p>SHapley Additive exPlanations (SHAP) plot is a visualization tool for interpreting how each explanatory variable contributes to the predicted outcome of the response variable for an individual. The length of the segment is proportional to the effect that an explanatory variable has on {response_var} for a given individual {shap_id}. The direction of the effect is given by the extension of the segment. Extension to the right indicates a positive effect and extension to the left indicates a negative effect. </p>
        #         <p>The SHAP plot shows that the prediction model uses the variables:
        #         {', '.join([f'[{var}]' for var in shap_features])} to predict: [{response_var}]. The base (average) value of [{response_var}] is {exp:.2f} and the model's predicted value is {pred_pa:.2f}.</p>
        #         <p>The shapley contribution for each variable are:</p>
        #         <p>{'<br>'.join([f'[{var}] : {value:.2f}' for var, value in zip(shap_features, shap_contributions)])}</p>
        #         <p>All variables investigated indicate a [{response_var}] lower than the data sample.</p>
        #         """
        #     else:
        #         #st.write("The variable with the largest positive contribution is {} with a contribution of {:.2f}.".format(largest_shap, shap_contributions.max()))
        #         summary_text = f"""
        #         SHapley Additive exPlanations (SHAP) plot is a visualization tool for interpreting how each explanatory variable contributes to the predicted outcome of the response variable for an individual. The length of the segment is proportional to the effect that an explanatory variable has on {response_var} for a given individual {shap_id}. The direction of the effect is given by the extension of the segment. Extension to the right indicates a positive effect and extension to the left indicates a negative effect. </p>
        #         <p>The SHAP plot shows that the prediction model uses the variables:
        #         {', '.join([f'[{var}]' for var in shap_features])} to predict: [{response_var}].</p>
        #         <p>The shapley contribution for each variable are:</p>
        #         <p>{'<br>'.join([f'[{var}] : {value:.2f}' for var, value in zip(shap_features, shap_contributions)])}</p>
        #         <p>The Variable with the largest positive contribution {shap_contributions.max():.2f} is: <span style='color: gold;'>[{largest_shap}]</span></p>
        #         """
                
        #     st.markdown(summary_text, unsafe_allow_html=True)

elif page == "Patient appointment":
    
    st.title("Patient Appointment")

    st.header("Calendar and Events")
    year = st.selectbox("Choose the year:", ["2024", "2025"])
    date_input = st.date_input("Select a date", value=date.today())
    event_input = st.text_input("Enter an event:")
    record_event = st.button("Record Event")

    # Placeholder for events (in-memory storage)
    if "events" not in st.session_state:
        st.session_state.events = pd.DataFrame(columns=["Date", "Event"])

    # 记录事件
    if record_event:
        new_event = pd.DataFrame({"Date": [date_input], "Event": [event_input]})
        st.session_state.events = pd.concat([st.session_state.events, new_event], ignore_index=True)

    # 显示当前事件
    st.write("Current Events:")
    st.write(st.session_state.events)

    # 删除事件部分
    st.header("Delete Event")
    if not st.session_state.events.empty:
        event_ids = st.session_state.events.index.tolist()
        selected_event_id = st.selectbox("Select Event ID to delete", event_ids, format_func=lambda x: f"{st.session_state.events.iloc[x]['Date']} - {st.session_state.events.iloc[x]['Event']}")
        delete_event = st.button("Delete Selected Event")

        if delete_event:
            st.session_state.events = st.session_state.events.drop(selected_event_id).reset_index(drop=True)
            st.success(f"Event ID {selected_event_id} has been deleted.")
            
            # 更新选择框和事件表格
            st.experimental_rerun()
    else:
        st.write("No events to delete.")
