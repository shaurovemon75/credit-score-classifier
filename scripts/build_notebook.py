"""Builds notebooks/01_credit_score_analysis.ipynb (run, then execute with nbconvert)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))

md("""# Smart Credit - Credit Score Classification
**Goal:** classify a customer's credit score (*Poor / Standard / Good*) from banking and credit data.

Workflow: 1) load data, 2) EDA, 3) cleaning, 4) model comparison, 5) final model + evaluation, 6) predictions on `test.csv`.
All heavy logic lives in the `creditscore/` package so the notebook, the training script and the API use identical code.""")

code("""import sys, json; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from IPython.display import Image, display
from creditscore import config as C, cleaning as K, modeling as M
sns.set_theme(style='whitegrid')
%matplotlib inline""")

md("## 1. Load the data")
code("""train = pd.read_csv(C.TRAIN_PATH, low_memory=False)
test  = pd.read_csv(C.TEST_PATH,  low_memory=False)
print('train', train.shape, '| test', test.shape)
print('customers in train:', train.Customer_ID.nunique(), '| months per customer:', train.groupby('Customer_ID').size().unique())
print('test has target column?', C.TARGET in test.columns)
train.head(3).T""")

md("## 2. Exploratory data analysis")
code("""fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
order = C.LABELS
train[C.TARGET].value_counts()[order].plot.bar(ax=ax[0], color=['#bc4749','#f2cc8f','#386641'], rot=0)
ax[0].set_title('Target distribution (class imbalance)')
(train[C.TARGET].value_counts(normalize=True)[order]*100).round(1).plot.pie(ax=ax[1], autopct='%1.1f%%', colors=['#bc4749','#f2cc8f','#386641'])
ax[1].set_ylabel(''); plt.tight_layout(); plt.show()""")

code("""# how dirty is the raw data?  (placeholders, underscores, impossible numbers)
dirty = pd.DataFrame({
  'missing (NaN)': train.isna().sum(),
  'has "_" noise': train.astype(str).apply(lambda s: s.str.contains('_$|^_+$', regex=True).sum()),
})
dirty[(dirty > 0).any(axis=1)].sort_values('missing (NaN)', ascending=False)""")

code("""print('Age range in raw data:', pd.to_numeric(train.Age.str.replace('_',''), errors='coerce').agg(['min','max']).tolist())
print('Num_Bank_Accounts raw range:', train.Num_Bank_Accounts.agg(['min','max']).tolist())
print('Interest_Rate raw range:', train.Interest_Rate.agg(['min','max']).tolist())""")

md("## 3. Cleaning and feature engineering")
md("""Rules (see `creditscore/cleaning.py`):
- strip `_` noise and convert text to numbers; placeholders (`_`, `NM`, `!@9#%8`) become missing
- values outside plausible ranges (age -500, 5000 credit cards, ...) become missing
- missing values are filled from the **same customer's other months** (median / mode); anything left is imputed inside the model pipeline
- `Credit_History_Age` -> months; `Type_of_Loan` -> one flag per loan type; ratio features added""")
code("""X, y, groups = M.load_xy()
print('model matrix:', X.shape, '| remaining NaN:', int(X.isna().sum().sum()))
X.describe().T[['min','50%','max']].round(2).head(18)""")

code("""df = X.copy(); df['Credit_Score'] = pd.Series(y).map(C.ID_TO_LABEL).values
cols = ['Delay_from_due_date','Num_of_Delayed_Payment','Interest_Rate','Outstanding_Debt','Credit_Utilization_Ratio','Num_Credit_Inquiries']
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, c in zip(axes.ravel(), cols):
    sns.boxplot(data=df, x='Credit_Score', y=c, order=C.LABELS, ax=ax, palette=['#bc4749','#f2cc8f','#386641'])
    ax.set_title(c)
plt.tight_layout(); plt.show()""")

code("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
pd.crosstab(df.Credit_Mix, df.Credit_Score, normalize='index')[C.LABELS].plot.bar(stacked=True, ax=ax[0], rot=0, color=['#bc4749','#f2cc8f','#386641'])
ax[0].set_title('Credit_Mix vs score')
pd.crosstab(df.Payment_of_Min_Amount, df.Credit_Score, normalize='index')[C.LABELS].plot.bar(stacked=True, ax=ax[1], rot=0, color=['#bc4749','#f2cc8f','#386641'])
ax[1].set_title('Payment_of_Min_Amount vs score'); plt.tight_layout(); plt.show()""")

code("""num = X.select_dtypes('number').drop(columns=[c for c in X.columns if c.startswith('Loan_')])
corr = num.assign(target=y).corr()
plt.figure(figsize=(12, 9)); sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation heatmap (target = 0 Poor, 1 Standard, 2 Good)'); plt.show()
corr['target'].drop('target').sort_values().round(3).tail(6), corr['target'].drop('target').sort_values().round(3).head(6)""")

md("""## 4. Model comparison
Customer-grouped 80/20 split: all 8 months of a customer stay on the same side, so the score is not inflated by memorising customers. Class imbalance is handled with balanced class weights. Results come from the full run of `python -m creditscore.train` (RandomForest takes ~1 min on a laptop).""")
code("""res = json.load(open('../reports/metrics.json'))
comp = pd.DataFrame(res['model_comparison']).set_index('model')
display(comp[['accuracy','macro_f1','weighted_f1','seconds']])
display(Image('../reports/figures/model_comparison.png'))""")

code("""pd.DataFrame(res['tuning'])[['num_leaves','learning_rate','n_estimators','min_child_samples','reg_lambda','accuracy','macro_f1']]""")

md("## 5. Final model - hold-out evaluation (live re-fit on 80% of customers)")
code("""from sklearn.metrics import classification_report, confusion_matrix
tr, va = M.grouped_split(X, y, groups)
model = M.lightgbm_pipeline(**res['best_params']).fit(X.iloc[tr], y[tr])
pred = model.predict(X.iloc[va])
print(classification_report(y[va], pred, target_names=C.LABELS, digits=3))
cm = confusion_matrix(y[va], pred)
sns.heatmap(cm / cm.sum(1, keepdims=True), annot=cm, fmt='d', cmap='Blues', xticklabels=C.LABELS, yticklabels=C.LABELS, cbar=False)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion matrix'); plt.show()""")

code("""display(Image('../reports/figures/feature_importance.png'))""")

md("""## 6. Predictions on `test.csv`
`test.csv` has no `Credit_Score` column, so it cannot be scored here; predictions are written to `reports/test_predictions.csv` by the training script.""")
code("""from creditscore.predict import predict_frame
sub = pd.read_csv('../reports/test_predictions.csv')
print(sub.shape); display(sub.head())
print(sub.Credit_Score.value_counts(normalize=True).round(3).to_dict())""")

md("""## 7. Conclusions
- Best model: tuned **LightGBM** (about 70% accuracy / 0.69 macro-F1 on unseen customers).
- Most influential signals (LightGBM gain): outstanding debt, whether only the minimum amount is paid, credit mix and interest rate.
- Errors are mostly between neighbouring classes: *Standard* has the lowest recall (0.64, often mistaken for Poor or Good) and *Good* the lowest precision (0.56). Almost no Good customers are labelled Poor.\n- Adding per-customer averages over all months only gave about +0.5 pt accuracy in a quick test, so the simpler single-record model was kept (it also works for the single-record API).
- The model is served through a FastAPI REST API (`api/main.py`) for single and batch predictions.""")

nbf.write(nb, "notebooks/01_credit_score_analysis.ipynb")
