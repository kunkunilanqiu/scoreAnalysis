# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:38:20 2024

@author: pengy
"""
# In[1]:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo, FactorAnalyzer
from factor_analyzer import FactorAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix#混淆矩阵（错判矩阵）
# In[2]:
df = pd.read_csv(r"E:\Desktop\pythonclass\Student_performance_data _.csv")
df = df.drop(columns=['StudentID'])
print(df.duplicated())
print(df.info())
df.describe().to_excel(r'E:\Desktop\GradeDescribe.xlsx')

# In[3]:箱线图 

sns.catplot(data=df, x="Ethnicity", y="GPA", kind="boxen")

sns.catplot(data=df, x="ParentalSupport", y="GPA", kind="boxen")

sns.catplot(data=df, x="ParentalEducation", y="GPA",  kind="boxen")

sns.catplot(data=df, x="Gender", y="GPA",  kind="boxen")

sns.catplot(data=df, x="Tutoring", y="GPA", kind="boxen")

sns.displot(
    data=df,
    x="GPA", hue="ParentalSupport",
    kind="kde", height=6,
    multiple="fill", clip=(0, None),
    palette="ch:rot=-.25,hue=1,light=.75",
)
percentages = df.groupby('ParentalEducation')['ParentalSupport'].value_counts(normalize=True).unstack()
plt.figure(figsize=(10, 6))
percentages.plot(kind='bar', stacked=True)
plt.title('Parental Education & Support')
plt.xlabel('ParentalEducation')
plt.ylabel('ParentalSupport')
plt.xticks(rotation=0)
plt.legend(title='ParentalSupport', loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
# In[4]:
sns.displot(df, x="GPA", hue="ParentalSupport", kind="kde")

sns.displot(df, x="StudyTimeWeekly", y="GPA", binwidth=(2, .5), cbar=True)\

plt.figure()
# 使用 seaborn 绘制散点图
sns.scatterplot(x='Absences', y='GPA', data=df, color='b', marker='o', s=100)


# 设置图形标题和坐标轴标签
plt.title('Scatter Plot of Absences vs GPA')
plt.xlabel('Absences')
plt.ylabel('GPA')

# 显示图形
plt.show()

# In[5]:绘制热力图
plt.figure(figsize=(16, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title('The correlation among features', y=1.05)
plt.tight_layout()
plt.show()



# In[6]:线性回归
from sklearn.linear_model import LinearRegression


'''根据热力图结果，去除部分相关性过低的标签'''
df_1 = df.drop(columns=['GradeClass','Age','Ethnicity','Gender'])
scaler = StandardScaler()
df_1['StudyTimeWeekly'] = scaler.fit_transform(df_1[['StudyTimeWeekly']])
df_1['Absences'] = scaler.fit_transform(df_1[['Absences']])

stat,p = calculate_bartlett_sphericity(df_1)
print(f"p-value:{round(p,2)}")

# kmo降维 值越大越好
kmo_all,kmo_model = calculate_kmo(df_1)
print(kmo_model)
'''根据结果不适合做降维分析'''

lm1 = ols("GPA~C(ParentalEducation)+StudyTimeWeekly+Absences+C(Tutoring)+C(ParentalSupport)+C(Extracurricular)+C(Sports)+C(Music)+C(Volunteering)", data=df_1).fit()
lm1.summary = lm1.summary()
print(lm1.summary)

tmp = df_1
tmp['pred1'] = lm1.predict(df_1)
tmp['resid1'] = lm1.resid
tmp.plot('pred1', 'resid1', kind='scatter')
plt.show()


fig1 = plt.figure('Figure1', figsize=(6, 4)).add_subplot(111)
sorted_ = np.sort(tmp['resid1'])
yvals = np.arange(len(sorted_))/float(len(sorted_))
fig1.plot(sorted_, yvals, color='red')
plt.show()


fig2 = plt.figure('figure2', figsize=(6, 4)).add_subplot(111)
x_label = stats.norm.ppf(yvals)
fig2.plot(x_label, sorted_)
stats.probplot(tmp['resid1'], dist="norm", plot=plt)
fig2.plot(color='red')
plt.show()


# In[6]:逻辑回归

'''进行变量处理'''


'''进行哑变量处理'''
df_1 = df_1.drop(columns = ['pred1','resid1','GPA'])

ls = []
for x in ['ParentalEducation','Tutoring']:
    _ = pd.get_dummies(df_1[x],prefix=x,drop_first=True)
    ls.append(_)
dummies = pd.concat(ls,axis=1)

tmp = df_1.drop(columns=['ParentalEducation','Tutoring'])
X = pd.concat([tmp, dummies],axis=1)

print(X.info())
X = X.astype(int)
'''成绩等级1，2划分为好学生'''


y = (df['GradeClass'] < 3).astype(int)



# 逻辑回归建模

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
logit = sm.Logit(y_train,X_train)

result = logit.fit()

result.summary()


# In[7]:  混淆矩阵和roc图
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
confusion = pd.DataFrame(confusion_matrix(y_test, y_pred),index=["real_0","real_1"],columns=['pred_0','pred_1'])
print(confusion)


from sklearn.metrics import roc_curve, auc
predictions = model.predict_proba(X_test)
fpr,tpr,thresholds = roc_curve(y_test, predictions[:, 1])  
roc_auc = auc(fpr,tpr)  
print(roc_auc)

plt.figure()
lw = 2
plt.plot(fpr,tpr,color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[]:聚类分析
from sklearn.cluster import KMeans
X = df_1

# In[]:
km = KMeans(n_clusters=5,random_state=1234).fit(X)
cluster_data = df[['ParentalEducation','StudyTimeWeekly','Absences','Tutoring','ParentalSupport','Extracurricular','Sports','Music','Volunteering']]
result_df = pd.concat([
    cluster_data,
    pd.DataFrame(km.labels_,columns=["label"]),
    ],axis=1)
label_count_df = result_df["label"].value_counts()/result_df.shape[0]
label_count_df = label_count_df.reset_index()
label_count_df.columns = ["label","percent"]
label_count_df = label_count_df.sort_values("label")
print("每个标签所占的百分比")
print(label_count_df)
result_df['GPA'] = df['GPA']
print("聚类标签下的Gpa")
print(result_df.groupby(['label'])['GPA'].describe())
print("每个标签下的各个列的数据")
tmp = result_df.groupby("label",as_index=False).agg({
    "ParentalEducation": [np.mean,np.std],
    "StudyTimeWeekly":[np.mean,np.std],
    "Absences":[np.mean,np.std],
    "Tutoring":[np.mean,np.std],
    "ParentalSupport":[np.mean,np.std],
    "Extracurricular":[np.mean,np.std],
    "Sports":[np.mean,np.std],
    "Music":[np.mean,np.std],
    "Volunteering":[np.mean,np.std],
    })
print(tmp)

tmp.to_excel(r'E:\Desktop\temp.xlsx', index=True)


# In[]
tmp = pd.concat([
    df[["GradeClass"]],
    result_df["label"]
    ],axis=1)
groupby = pd.pivot_table(tmp, index=['label'],columns=["GradeClass"],aggfunc={"GradeClass":len})["GradeClass"]
groupby['total']  =groupby.sum(axis=1)
groupby = groupby.fillna(0)
cols = groupby.columns
for col in cols:
    if col == "total":continue
    groupby[col]  =groupby[col]/groupby["total"]
del groupby["total"]
groupby = groupby.stack().reset_index() 
groupby.columns = ['label','GradeClass','ratio']
groupby = groupby.query("ratio>0")
print(groupby)


result_df['GradeClass'] = df[['GradeClass']]
percentages = result_df.groupby('label')['GradeClass'].value_counts(normalize=True).unstack()

# 绘制堆叠条形图
plt.figure(figsize=(10, 6))
percentages.plot(kind='bar', stacked=True)
plt.title('GradeClass Distribution by Label')
plt.xlabel('Label')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(title='GradeClass', loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
