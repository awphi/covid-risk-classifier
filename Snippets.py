imp = dt_clf.steps[0][1].feature_importances_
df_train_cols = df.drop(['outcome'], axis=1).columns
imp_X = []
imp_y = []

for i,v in enumerate(imp):
    if v < 0.00001:
        continue
    print('%s: %.5f' % (df_train_cols[i], v))
    imp_y.append(df_train_cols[i])
    imp_X.append(v)

#Define size of bar plot
plt.figure(figsize=(10,8))
#Plot Searborn bar chart
sns.barplot(x=imp_X, y=imp_y)
#Add chart labels
plt.title('FEATURE IMPORTANCE')
plt.xlabel('FEATURE IMPORTANCE')
plt.ylabel('FEATURE NAMES')