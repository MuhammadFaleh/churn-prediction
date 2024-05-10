---
jupyter:
  accelerator: GPU
  colab:
  gpuClass: standard
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.1
  nbformat: 4
  nbformat_minor: 1
---

::: {.cell .markdown id="b73EDIimtPp7"}
# **Chrun Prediction ** {#chrun-prediction-}

This file will cover data smoting, MLP and decision tree along with
thier hybrid and ensemble approaches:

-   [Loading & smote](#loading)
-   [Model fitting](#model)
    -   [decision tree](#dt)
    -   [MLP](#MLP)
    -   [DRNN](#rnn)
    -   [XGB](#xgb)
    -   [Random Forest](#rf)
-   [hybrid](#hy)
-   [validation](#val)
:::

::: {.cell .markdown id="_6Gu-Dg8wthu"}
### imports
:::

::: {.cell .code execution_count="1" id="OIKR3ys4puDT"}
``` python
#imports
import pandas as pd
# from torch.utils.data
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from skorch import NeuralNetBinaryClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostRegressor
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns # For creating plots
import matplotlib.ticker as mtick # For specifying the axes tick format
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.under_sampling import NearMiss
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns #Visualization
import plotly.express as px
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import collections


# Other Libraries
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report,roc_curve
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
```
:::

::: {.cell .markdown id="Eve4dPqUwxmP"}
`<a id='loading'>`{=html}`</a>`{=html}

### loading
:::

::: {.cell .code execution_count="2"}
``` python
df_train = pd.read_csv("train_set_final.csv")
df_test = pd.read_csv("test_set_final.csv")
y_train_clean = df_train['STATUS']
X_train_clean = df_train.drop(['STATUS'], axis=1)
# X_train1 = X_train1[col]

y_test_clean = df_test['STATUS']
X_test_clean = df_test.drop(['STATUS'], axis=1)

```
:::

::: {.cell .code execution_count="3" id="sjiol97x_oW1" outputId="6d812dd1-2f13-433b-e4ab-0d216a406ec3" scrolled="true"}
``` python
df_train = pd.read_csv("train_resampled.csv")
df_test = pd.read_csv("test_resampled.csv")

y_train_sm = df_train['STATUS']
X_train_sm = df_train.drop(['STATUS'], axis=1)

y_test_sm = df_test['STATUS']
X_test_sm = df_test.drop(['STATUS'], axis=1)
```
:::

::: {.cell .code execution_count="14" id="8od4ZJVv_oW1"}
``` python
X_test_sm_c = X_test_sm.copy()
X_train_sm_c = X_train_sm.copy()


scale = StandardScaler()
scale.fit(X_train_sm_c)

scale_t = StandardScaler()
scale_t.fit(X_test_sm_c)

sc_train =scale.transform(X_train_sm_c)
sc_test =scale_t.transform(X_test_sm_c)
```
:::

::: {.cell .code execution_count="15" id="dssGVLgS_oW1"}
``` python
X_test_c = X_test_clean.copy()
X_train_c = X_train_clean.copy()


scale = StandardScaler()
scale.fit(X_train_c)

scale_t = StandardScaler()
scale_t.fit(X_test_c)

sc_train_clean = scale.transform(X_train_c)
sc_test_clean = scale_t.transform(X_test_c)
```
:::

::: {.cell .markdown id="9faVIm3J_oW2"}
data copying and scaling for MLP
:::

::: {.cell .code execution_count="6" id="kamwOJDDyqeO"}
``` python
def model_val(labels_test,pred, pred_proba):
    cnf_matrix=confusion_matrix(labels_test,pred)
    print("AUC",roc_auc_score(labels_test, pred_proba))
    print("the recall for this model is :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    print("TP",cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud
    print("TN",cnf_matrix[0,0]) # no. of normal transaction which are predited normal
    print("FP",cnf_matrix[0,1]) # no of normal transaction which are predicted fraud
    print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal

    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test,pred, digits=5))
    print(classification_report_imbalanced(labels_test, pred, digits=5))
```
:::

::: {.cell .code execution_count="7" id="v2z4IZtl_oW2"}
``` python
def roc_png(proba,y, name):
    fpr, tpr, thresholds = roc_curve(y, proba)

    auc = roc_auc_score(y, proba)

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='ِAUC = %0.4f' % auc)

    ax.set_xlim([-0.1, 1.0])
    ax.set_ylim([-0.1, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)

    # Add a title and legend
    ax.set_title(name, fontsize=18, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12)

    # Customize the tick labels and grid
    ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=2)
    ax.grid(color='lightgray', linestyle='--')

    # Remove the top and right spines
    sns.despine()

    # Save the plot to a file with high resolution
    fig.savefig(name + '.png', dpi=1200, bbox_inches='tight')

    # Show the plot
    plt.show()
```
:::

::: {.cell .code execution_count="8"}
``` python
def  CrossValidation(model,X,y,K):
    
    # Define the scoring metrics
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='weighted'),
               'recall': make_scorer(recall_score, average='weighted'),
               'f1_score': make_scorer(f1_score, average='weighted')}

    cv_results = cross_validate(model, X, y, cv=K, scoring=scoring)

   

   # Print the results
    for metric, values in cv_results.items():
        print(f"{metric}: {values.mean():.5f} (+/- {values.std() * 2:.4f})")
```
:::

::: {.cell .code execution_count="9"}
``` python
def model(model,features_train,features_test,labels_train,labels_test):
    clf= model
    
    clf.fit(features_train,labels_train)
    
    pred=clf.predict(features_test)
    
    pred_proba=clf.predict_proba(features_test)
    
    print("AUC",roc_auc_score(labels_test, pred_proba[:,1]))
    
    
    cnf_matrix=confusion_matrix(labels_test,pred)
    
   
    print("TP",cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud
    print("TN",cnf_matrix[0,0]) # no. of normal transaction which are predited normal
    print("FP",cnf_matrix[0,1]) # no of normal transaction which are predicted fraud
    print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal from sklearn.metrics import f1_score
    
    
    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test,pred,digits=5))
    return pred_proba, pred
```
:::

::: {.cell .markdown id="l3h308UOwRkl"}
`<a id='model'>`{=html}`</a>`{=html}

# **Model fiting**
:::

::: {.cell .markdown id="i3nH1AB0_oW2"}
`<a id='dt'>`{=html}`</a>`{=html}

## 1.1 Decision Tree Classifier {#11-decision-tree-classifier}

The Decision Tree classifier is a popular machine learning algorithm
that uses a tree-like model of decisions and their possible
consequences. Each internal node represents a feature or attribute, each
branch represents a decision rule, and each leaf node represents an
outcome. The Decision Tree Classifier is used for this particular task.

### Model Hyperparameters

The Decision Tree classifier used for this task was configured with the
following hyperparameters:

  -----------------------------------------------------------------------------------
  Hyperparameter   Value                    Description
  ---------------- ------------------------ -----------------------------------------
  ccp_alpha        1.5151515151515151e-05   Complexity parameter used for pruning the
                                            decision tree.

  random_state     1                        Seed value for random number generation
                                            to ensure reproducibility.
  -----------------------------------------------------------------------------------

These hyperparameters were chosen based on experimentation and may
require further tuning for optimal performance on specific data.
:::

::: {.cell .markdown id="L-NG_dht_oW3"}
### balanced
:::

::: {.cell .code id="gRFWfG96_oW3" outputId="01682c5b-4c23-4e2f-9532-8c607af3e471"}
``` python
model_dt = DecisionTreeClassifier(ccp_alpha =  1.5151515151515151e-05 , random_state=1)
model_dt.fit(X_train_sm, y_train_sm)
```

::: {.output .execute_result execution_count="37"}
```{=html}
<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier(ccp_alpha=1.5151515151515151e-05, random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(ccp_alpha=1.5151515151515151e-05, random_state=1)</pre></div></div></div></div></div>
```
:::
:::

::: {.cell .code id="BIHO2_DB_oW3" outputId="da18de17-390d-4b7f-a768-2b2253ab5f8e" scrolled="false"}
``` python
y_pred_nor = model_dt.predict(X_test_sm)
probas_nor= model_dt.predict_proba(X_test_sm)
model_val(y_test_sm, y_pred_nor, probas_nor[:,1])
```

::: {.output .stream .stdout}
    AUC 0.9261147149174054
    the recall for this model is : 0.9341469616069158
    TP 51436
    TN 38584
    FP 5928
    FN 3626

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.91410   0.86682   0.88983     44512
               1    0.89666   0.93415   0.91502     55062

        accuracy                        0.90405     99574
       macro avg    0.90538   0.90048   0.90243     99574
    weighted avg    0.90445   0.90405   0.90376     99574

                       pre       rec       spe        f1       geo       iba       sup

              0    0.91410   0.86682   0.93415   0.88983   0.89986   0.80429     44512
              1    0.89666   0.93415   0.86682   0.91502   0.89986   0.81519     55062

    avg / total    0.90445   0.90405   0.89692   0.90376   0.89986   0.81032     99574
:::
:::

::: {.cell .code id="t9RzNe5p_oW3" outputId="a54053ae-5fca-4ce8-8207-147981b587cc" scrolled="false"}
``` python
y_pred_train = model_dt.predict(X_train_sm)
y_proba_train = model_dt.predict_proba(X_train_sm)
model_val(y_train_sm, y_pred_train,y_proba_train[:,1])
```

::: {.output .stream .stdout}
    AUC 0.9906271812669769
    the recall for this model is : 0.9811666848280121
    TP 108050
    TN 93054
    FP 3446
    FN 2074

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97820   0.96429   0.97119     96500
               1    0.96909   0.98117   0.97509    110124

        accuracy                        0.97328    206624
       macro avg    0.97365   0.97273   0.97314    206624
    weighted avg    0.97335   0.97328   0.97327    206624

                       pre       rec       spe        f1       geo       iba       sup

              0    0.97820   0.96429   0.98117   0.97119   0.97269   0.94453     96500
              1    0.96909   0.98117   0.96429   0.97509   0.97269   0.94773    110124

    avg / total    0.97335   0.97328   0.97217   0.97327   0.97269   0.94623    206624
:::
:::

::: {.cell .code id="b8p14OVz_oW3" outputId="94fde46d-f465-47ac-cf34-0aa21ae91abb"}
``` python
roc_png(proba =probas_nor[:,1],y=y_test_sm, name = 'ROC Curve for DT')
```

::: {.output .display_data}
![](vertopal_af71f06758214898b489638f50e60adb/18e102df371be188efd75ddfad1b4ba7e21b9286.png)
:::
:::

::: {.cell .markdown id="_CqvNJ2m_oW3"}
**Dt with prune results:**

-   Test Accuracy = 90.405%.
-   Test F1-score = 90.243%.
-   Test Precision = 90.538%.
:::

::: {.cell .markdown id="rM1HKjvT_oW4"}
### unbalanced
:::

::: {.cell .code execution_count="14" id="xNmLR6mH_oW4" outputId="a6ac454e-7a74-47c8-ac1b-9bd30305e502"}
``` python
model_dt_clean = DecisionTreeClassifier(ccp_alpha =  1.5151515151515151e-05 , random_state=1)
model_dt_clean.fit(X_train_clean, y_train_clean)


y_pred_nor_clean = model_dt_clean.predict(X_test_clean)
probas_nor_clean = model_dt_clean.predict_proba(X_test_clean)
model_val(y_test_clean, y_pred_nor_clean, probas_nor_clean[:,1])
```

::: {.output .stream .stdout}
    AUC 0.7539959190561712
    the recall for this model is : 0.9783879989829647
    TP 53872
    TN 1290
    FP 1672
    FN 1190

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.52016   0.43552   0.47409      2962
               1    0.96990   0.97839   0.97412     55062

        accuracy                        0.95068     58024
       macro avg    0.74503   0.70695   0.72411     58024
    weighted avg    0.94694   0.95068   0.94860     58024

                       pre       rec       spe        f1       geo       iba       sup

              0    0.52016   0.43552   0.97839   0.47409   0.65277   0.40297      2962
              1    0.96990   0.97839   0.43552   0.97412   0.65277   0.44924     55062

    avg / total    0.94694   0.95068   0.46323   0.94860   0.65277   0.44687     58024
:::
:::

::: {.cell .markdown id="soHlxQig_oW4"}
**unbalanced Dt with prune results:**

-   Test Accuracy = 95.068%.
-   Test F1-score = 72.411%.
-   Test Precision = 74.503%.
:::

::: {.cell .markdown id="uKtWR85j_oW4"}
## Optimal Dt 1.2 {#optimal-dt-12}

(with ada, bagging and prune)
:::

::: {.cell .markdown id="IPDlAzVR_oW4"}
### balanced {#balanced}
:::

::: {.cell .code id="kb7tgpLp_oW4" outputId="68f6cab0-d841-4607-f13d-62f33e0be663"}
``` python
model_ada = BalancedBaggingClassifier(
AdaBoostClassifier(DecisionTreeClassifier(ccp_alpha =3.49297098164335e-04 , random_state=1),
                           random_state=1,n_estimators=120), n_jobs=-1,random_state=1)
model_ada.fit(X_train_sm, y_train_sm)
```

::: {.output .execute_result execution_count="22"}
```{=html}
<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>BalancedBaggingClassifier(estimator=AdaBoostClassifier(estimator=DecisionTreeClassifier(ccp_alpha=0.000349297098164335,
                                                                                        random_state=1),
                                                       n_estimators=120,
                                                       random_state=1),
                          n_jobs=-1, random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">BalancedBaggingClassifier</label><div class="sk-toggleable__content"><pre>BalancedBaggingClassifier(estimator=AdaBoostClassifier(estimator=DecisionTreeClassifier(ccp_alpha=0.000349297098164335,
                                                                                        random_state=1),
                                                       n_estimators=120,
                                                       random_state=1),
                          n_jobs=-1, random_state=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: AdaBoostClassifier</label><div class="sk-toggleable__content"><pre>AdaBoostClassifier(estimator=DecisionTreeClassifier(ccp_alpha=0.000349297098164335,
                                                    random_state=1),
                   n_estimators=120, random_state=1)</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(ccp_alpha=0.000349297098164335, random_state=1)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(ccp_alpha=0.000349297098164335, random_state=1)</pre></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>
```
:::
:::

::: {.cell .code id="AniaxSAG_oW4" outputId="db731425-9788-430b-de37-9662fda9fd3b"}
``` python

y_pred_ada = model_ada.predict(X_test_sm)
probas_ada=model_ada.predict_proba(X_test_sm)
model_val(y_test_sm, y_pred_ada, probas_ada[:,1])
```

::: {.output .stream .stdout}
    AUC 0.9934335805815762
    the recall for this model is : 0.9870872834259562
    TP 54351
    TN 41321
    FP 3191
    FN 711

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98308   0.92831   0.95491     44512
               1    0.94454   0.98709   0.96535     55062

        accuracy                        0.96081     99574
       macro avg    0.96381   0.95770   0.96013     99574
    weighted avg    0.96177   0.96081   0.96068     99574

                       pre       rec       spe        f1       geo       iba       sup

              0    0.98308   0.92831   0.98709   0.95491   0.95725   0.91094     44512
              1    0.94454   0.98709   0.92831   0.96535   0.95725   0.92171     55062

    avg / total    0.96177   0.96081   0.95459   0.96068   0.95725   0.91690     99574
:::
:::

::: {.cell .code id="zzz0NQ0E_oW5" outputId="e109c993-caac-438b-d6c5-ef474aac46f4"}
``` python
y_pred_train_bb = model_ada.predict(X_train_sm)
probas_=model_ada.predict_proba(X_test_sm)
model_val(y_test_sm, y_pred_ada, probas_[:,1])
```

::: {.output .stream .stdout}
    AUC 0.9934335805815762
    the recall for this model is : 0.9870872834259562
    TP 54351
    TN 41321
    FP 3191
    FN 711

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98308   0.92831   0.95491     44512
               1    0.94454   0.98709   0.96535     55062

        accuracy                        0.96081     99574
       macro avg    0.96381   0.95770   0.96013     99574
    weighted avg    0.96177   0.96081   0.96068     99574

                       pre       rec       spe        f1       geo       iba       sup

              0    0.98308   0.92831   0.98709   0.95491   0.95725   0.91094     44512
              1    0.94454   0.98709   0.92831   0.96535   0.95725   0.92171     55062

    avg / total    0.96177   0.96081   0.95459   0.96068   0.95725   0.91690     99574
:::
:::

::: {.cell .code id="k7AAgwVa_oW5" outputId="df041eda-d203-4636-bb3b-02b4e3109f25"}
``` python
roc_png(proba =probas_ada,y=y_test_sm, name = 'ROC Curve for Decison Tree with ADA')
```

::: {.output .display_data}
![](vertopal_af71f06758214898b489638f50e60adb/2801f1d959fdc34fd8fa5deee9e95742286e00bf.png)
:::
:::

::: {.cell .markdown id="Tuf6no05_oW5"}
**optimal dt results:**

-   Test Accuracy = 96.081%.
-   Test F1-score = 96.013%.
-   Test Precision = 96.381%.
:::

::: {.cell .markdown id="DJKSBEEq_oW5"}
### unbalanced {#unbalanced}
:::

::: {.cell .code id="OiuUOs75_oW6" outputId="1d790293-8169-4140-a527-682d0656cc73"}
``` python
model_ada_clean = BalancedBaggingClassifier(
AdaBoostClassifier(DecisionTreeClassifier(ccp_alpha =3.49297098164335e-04 , random_state=1),
                           random_state=1,n_estimators=120), n_jobs=-1,random_state=1)

model_ada_clean.fit(X_train_clean, y_train_clean)


y_pred_ada_clean = model_ada_clean.predict(X_test_clean)
probas_ada_clean = model_ada_clean.predict_proba(X_test_clean)
model_val(y_test_clean, y_pred_ada_clean, probas_ada_clean[:,1])
```

::: {.output .stream .stdout}
    AUC 0.9042727869885598
    the recall for this model is : 0.9014202172096909
    TP 49634
    TN 2007
    FP 955
    FN 5428

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.26994   0.67758   0.38607      2962
               1    0.98112   0.90142   0.93958     55062

        accuracy                        0.88999     58024
       macro avg    0.62553   0.78950   0.66283     58024
    weighted avg    0.94482   0.88999   0.91133     58024

                       pre       rec       spe        f1       geo       iba       sup

              0    0.26994   0.67758   0.90142   0.38607   0.78153   0.59712      2962
              1    0.98112   0.90142   0.67758   0.93958   0.78153   0.62446     55062

    avg / total    0.94482   0.88999   0.68901   0.91133   0.78153   0.62306     58024
:::
:::

::: {.cell .markdown id="kx-Q-TLl_oW6"}
**unbalanced optimal dt results:**

-   Test Accuracy = 88.276%.
-   Test F1-score = 62.402%.
-   Test Precision = 66.139%.
:::

::: {.cell .markdown id="_y4PDzKf_oW6"}
`<a id='MLP'>`{=html}`</a>`{=html}

## 1.3 Multi-Layer Perceptron (MLP) Classifier {#13-multi-layer-perceptron-mlp-classifier}

The Multi-Layer Perceptron (MLP) classifier is a type of artificial
neural network that is widely used for classification tasks. It consists
of multiple layers of interconnected neurons and uses backpropagation to
optimize the weights. The MLPClassifier is used for this particular
task.

### Model Hyperparameters {#model-hyperparameters}

The MLP classifier used for this task was configured with the following
hyperparameters:

  ----------------------------------------------------------------------------
  Hyperparameter       Value    Description
  -------------------- -------- ----------------------------------------------
  solver               adam     The solver algorithm for weight optimization.

  activation           relu     The activation function for hidden layers.

  hidden_layer_sizes   120      The number of neurons in the hidden layers.

  random_state         1        Seed value for random number generation.
  ----------------------------------------------------------------------------

These hyperparameters were chosen based on experimentation and domain
knowledge to achieve the best performance for the given task.
:::

::: {.cell .markdown id="2kqkPLkh_oW6"}
### balanced {#balanced}
:::

::: {.cell .code id="B0uq24HU_oW7" outputId="6e0c64e1-ea4c-40f0-efc9-075b4aa020c0"}
``` python
model_MLP =  MLPClassifier(solver='adam', activation= 'relu', hidden_layer_sizes=120, random_state=1 )
#BalancedBaggingClassifier(MLPClassifier(solver='adam', activation= 'relu', hidden_layer_sizes=120, random_state=1 ), n_jobs=-1,random_state=1)
model_MLP.fit(sc_train, y_train_sm)
```

::: {.output .execute_result execution_count="32"}
```{=html}
<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MLPClassifier(hidden_layer_sizes=120, random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" checked><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">MLPClassifier</label><div class="sk-toggleable__content"><pre>MLPClassifier(hidden_layer_sizes=120, random_state=1)</pre></div></div></div></div></div>
```
:::
:::

::: {.cell .code id="7emAC3KU_oW7" outputId="66846a96-515a-4d06-d56e-fc48a69dcba5"}
``` python
#0.95314   0.95167   0.94394   0.95146   0.94710   0.89769     99550
y_pred = model_MLP.predict(sc_test)
probas_MLP=model_MLP.predict_proba(sc_test)
model_val(y_test_sm, y_pred, probas_MLP[:,1])
```

::: {.output .stream .stdout}
    AUC 0.9801583062362406
    the recall for this model is : 0.9462242562929062
    TP 52101
    TN 41216
    FP 3296
    FN 2961

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.93297   0.92595   0.92945     44512
               1    0.94050   0.94622   0.94335     55062

        accuracy                        0.93716     99574
       macro avg    0.93674   0.93609   0.93640     99574
    weighted avg    0.93714   0.93716   0.93714     99574

                       pre       rec       spe        f1       geo       iba       sup

              0    0.93297   0.92595   0.94622   0.92945   0.93603   0.87438     44512
              1    0.94050   0.94622   0.92595   0.94335   0.93603   0.87793     55062

    avg / total    0.93714   0.93716   0.93501   0.93714   0.93603   0.87635     99574
:::
:::

::: {.cell .code id="Xv5qVBgv_oW7" outputId="b7abe39e-42e3-438e-f55f-320f7133bce8"}
``` python
#0.96789   0.96744   0.96495   0.96739   0.96601   0.93341
y_pred_train_cc = model_MLP.predict(sc_train)
probas_tr=model_MLP.predict_proba(sc_train)
model_val(y_train_sm, y_pred_train_cc,probas_tr[:,1])
```

::: {.output .stream .stdout}
    AUC 0.989310675784603
    the recall for this model is : 0.9489121354109913
    TP 104498
    TN 91492
    FP 5008
    FN 5626

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.94207   0.94810   0.94508     96500
               1    0.95427   0.94891   0.95158    110124

        accuracy                        0.94853    206624
       macro avg    0.94817   0.94851   0.94833    206624
    weighted avg    0.94857   0.94853   0.94854    206624

                       pre       rec       spe        f1       geo       iba       sup

              0    0.94207   0.94810   0.94891   0.94508   0.94851   0.89959     96500
              1    0.95427   0.94891   0.94810   0.95158   0.94851   0.89974    110124

    avg / total    0.94857   0.94853   0.94848   0.94854   0.94851   0.89967    206624
:::
:::

::: {.cell .code id="PmCaGM5G_oW7" outputId="0446762a-b93d-4b87-bb7e-d7aabdfff0f2"}
``` python
roc_png(proba =probas_MLP,y=y_test_sm, name = 'ROC Curve for MLP')
```

::: {.output .display_data}
![](vertopal_af71f06758214898b489638f50e60adb/308450548fd516dbd290dbff1de330f2371d8694.png)
:::
:::

::: {.cell .markdown id="Dw82Ou1S_oW8"}
**MLP results:**

-   Test Accuracy = 95.295%.
-   Test F1-score = 95.215%.
-   Test Precision = 95.546%.
:::

::: {.cell .markdown id="5XIN3XEn_oW8"}
### unbalanced {#unbalanced}
:::

::: {.cell .code id="6GhK022m_oW8" outputId="f687177f-28bd-4632-8bba-15a9ff892397"}
``` python
model_MLP_clean =  MLPClassifier(solver='adam', activation= 'relu', hidden_layer_sizes=120, random_state=1 )
#BalancedBaggingClassifier(MLPClassifier(solver='adam', activation= 'relu', hidden_layer_sizes=120, random_state=1 ), n_jobs=-1,random_state=1)
model_MLP_clean.fit(sc_train_clean, y_train_clean)


y_pred_mlp_clean = model_MLP_clean.predict(sc_test_clean)
probas_mlp_clean = model_MLP_clean.predict_proba(sc_test_clean)
model_val(y_test_clean, y_pred_mlp_clean, probas_mlp_clean[:,1])
```

::: {.output .stream .stdout}
    AUC 0.8341952246771799
    the recall for this model is : 0.9921361374450619
    TP 54629
    TN 463
    FP 2499
    FN 433

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.51674   0.15631   0.24002      2962
               1    0.95626   0.99214   0.97387     55062

        accuracy                        0.94947     58024
       macro avg    0.73650   0.57422   0.60694     58024
    weighted avg    0.93382   0.94947   0.93640     58024

                       pre       rec       spe        f1       geo       iba       sup

              0    0.51674   0.15631   0.99214   0.24002   0.39381   0.14212      2962
              1    0.95626   0.99214   0.15631   0.97387   0.39381   0.16805     55062

    avg / total    0.93382   0.94947   0.19898   0.93640   0.39381   0.16672     58024
:::
:::

::: {.cell .markdown id="qlt8XIQx_oW8"}
**unbalanced MLP results:**

-   Test Accuracy = 91.576%.
-   Test F1-score = 68.244%.
-   Test Precision = 64.762%.
:::

::: {.cell .markdown id="nZyiZ0qP_oW8"}
`<a id='rnn'>`{=html}`</a>`{=html}

## DRNN 1.4 {#drnn-14}
:::

::: {.cell .code id="uCLQgPEu_oW8"}
``` python
class Rnn_model(nn.Module): #best perofmer is rnn + 2 fc prelu + classifier layer 100 hiddenshape
    def __init__(self, input_shape:int, hidden_shape:int, output_shape:int):
        super(Rnn_model,self).__init__()
        self.next_shape = hidden_shape*2
        self.rnn =nn.RNN(input_shape, hidden_shape, batch_first=True)


        self.con2 = nn.Sequential(

            nn.Linear(in_features=hidden_shape,out_features=hidden_shape),
            nn.PReLU(),)

        self.classifer = nn.Sequential(
            nn.Linear(in_features=self.next_shape*2,out_features=self.next_shape*2),
            nn.PReLU(),
            nn.Linear(in_features=self.next_shape*2,out_features=self.next_shape),
            nn.PReLU(),
            nn.Linear(in_features=self.next_shape,out_features=hidden_shape),
            nn.PReLU(),
)
        self.classifer2 = nn.Linear(in_features=hidden_shape ,out_features=output_shape)

    def forward(self,x):
        z, a = self.rnn(x)
        z = z.contiguous().view(-1, self.next_shape//2)
        z = self.con2(z)
        z = self.classifer2(z)
        return z
```
:::

::: {.cell .code id="qRlplfdo_oW9"}
``` python
import skorch
class NeuralNetBinaryClassifier(skorch.NeuralNetBinaryClassifier):
    def fit(self, X, y, **fit_params):
        return super().fit(X, np.asarray(y, dtype=np.float32), **fit_params)
```
:::

::: {.cell .markdown id="OKpuYDEI_oW9"}
### balanced {#balanced}
:::

::: {.cell .code id="iFJWqPfG_oW9" outputId="10a452f4-7949-4655-e914-7e78494754ba" scrolled="true"}
``` python
torch.cuda.manual_seed(42)
torch.manual_seed(42)
model_rnn= NeuralNetBinaryClassifier(Rnn_model(53,100,1),
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.001,
    max_epochs=4,
    batch_size=700,
    verbose=False

)
X_train_ = torch.tensor(sc_train,dtype=torch.float32)
y_train_ = torch.tensor(y_train_sm,dtype=torch.float32)
model_rnn.fit(X_train_, y_train_)

```

::: {.output .execute_result execution_count="17"}
    <class '__main__.NeuralNetBinaryClassifier'>[initialized](
      module_=Rnn_model(
        (rnn): RNN(53, 100, batch_first=True)
        (con2): Sequential(
          (0): Linear(in_features=100, out_features=100, bias=True)
          (1): PReLU(num_parameters=1)
        )
        (classifer): Sequential(
          (0): Linear(in_features=400, out_features=400, bias=True)
          (1): PReLU(num_parameters=1)
          (2): Linear(in_features=400, out_features=200, bias=True)
          (3): PReLU(num_parameters=1)
          (4): Linear(in_features=200, out_features=100, bias=True)
          (5): PReLU(num_parameters=1)
        )
        (classifer2): Linear(in_features=100, out_features=1, bias=True)
      ),
    )
:::
:::

::: {.cell .code id="TlRmEm-5_oW9" outputId="ffd39c85-0967-4640-a626-d4210edabd7d"}
``` python
X_test_ = torch.tensor(sc_test,dtype=torch.float32)
y_test_ = torch.tensor(y_test_sm,dtype=torch.float32)

y_pred_rnn = model_rnn.predict(X_test_)
probas_rnn = model_rnn.predict_proba(X_test_)
model_val(y_test_, y_pred_rnn, probas_rnn[:,1])
```

::: {.output .stream .stdout}
    AUC 0.9737716403577187
    the recall for this model is : 0.9990192873488069
    TP 55008
    TN 41527
    FP 2985
    FN 54

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

             0.0    0.99870   0.93294   0.96470     44512
             1.0    0.94853   0.99902   0.97312     55062

        accuracy                        0.96948     99574
       macro avg    0.97361   0.96598   0.96891     99574
    weighted avg    0.97096   0.96948   0.96936     99574

                       pre       rec       spe        f1       geo       iba       sup

            0.0    0.99870   0.93294   0.99902   0.96470   0.96541   0.92587     44512
            1.0    0.94853   0.99902   0.93294   0.97312   0.96541   0.93818     55062

    avg / total    0.97096   0.96948   0.96248   0.96936   0.96541   0.93268     99574
:::
:::

::: {.cell .code id="Q8tiRq1J_oW9" outputId="fa22daf6-a105-4c29-ac6e-cc2706c47d38"}
``` python
roc_png(proba =probas_rnn,y=y_test_sm, name = 'ROC Curve for DRNN')
```

::: {.output .display_data}
![](vertopal_af71f06758214898b489638f50e60adb/d613ccd66e4504dc2c9dbc75a72d1e68e92944ee.png)
:::
:::

::: {.cell .markdown id="Yr1kzklp_oW-"}
**DRNN results:**

-   Test Accuracy = 96.948%.
-   Test F1-score = 96.891%.
-   Test Precision = 97.361%.
:::

::: {.cell .markdown id="aSrrAw-U_oW-"}
### unbalanced {#unbalanced}
:::

::: {.cell .code id="l_my0e01_oW-" outputId="fff0fc70-b1e1-4046-8d4d-605632ef7d18"}
``` python
torch.cuda.manual_seed(42)
torch.manual_seed(42)
model_rnn_clean= NeuralNetBinaryClassifier(Rnn_model(53,100,1),
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.001,
    max_epochs=4,
    batch_size=700,
    verbose=False

)
X_train_clean = torch.tensor(sc_train_clean,dtype=torch.float32)
y_train_clean = torch.tensor(y_train_clean,dtype=torch.float32)
model_rnn_clean.fit(X_train_clean, y_train_clean)
```

::: {.output .execute_result execution_count="24"}
    <class 'skorch.classifier.NeuralNetBinaryClassifier'>[initialized](
      module_=Rnn_model(
        (rnn): RNN(53, 100, batch_first=True)
        (con2): Sequential(
          (0): Linear(in_features=100, out_features=100, bias=True)
          (1): PReLU(num_parameters=1)
        )
        (classifer): Sequential(
          (0): Linear(in_features=400, out_features=400, bias=True)
          (1): PReLU(num_parameters=1)
          (2): Linear(in_features=400, out_features=200, bias=True)
          (3): PReLU(num_parameters=1)
          (4): Linear(in_features=200, out_features=100, bias=True)
          (5): PReLU(num_parameters=1)
        )
        (classifer2): Linear(in_features=100, out_features=1, bias=True)
      ),
    )
:::
:::

::: {.cell .code id="cAErUKEl_oW-" outputId="c026f365-7a8e-4ae0-f721-709209a0a279"}
``` python
X_test_clean = torch.tensor(sc_test_clean,dtype=torch.float32)
y_test_clean = torch.tensor(y_test_clean,dtype=torch.float32)

y_pred_rnn_clean = model_rnn_clean.predict(X_test_clean)
probas_rnn_clean = model_rnn_clean.predict_proba(X_test_clean)
model_val(y_test_clean, y_pred_rnn_clean, probas_rnn_clean[:,1])
```

::: {.output .stream .stdout}
    AUC 0.7255950084725558
    the recall for this model is : 0.9998910319276452
    TP 55056
    TN 0
    FP 2962
    FN 6

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

             0.0    0.00000   0.00000   0.00000      2962
             1.0    0.94895   0.99989   0.97375     55062

        accuracy                        0.94885     58024
       macro avg    0.47447   0.49995   0.48688     58024
    weighted avg    0.90051   0.94885   0.92405     58024

                       pre       rec       spe        f1       geo       iba       sup

            0.0    0.00000   0.00000   0.99989   0.00000   0.00000   0.00000      2962
            1.0    0.94895   0.99989   0.00000   0.97375   0.00000   0.00000     55062

    avg / total    0.90051   0.94885   0.05104   0.92405   0.00000   0.00000     58024
:::
:::

::: {.cell .markdown id="1BQNTugr_oW-"}
**unbalanced DRNN results:**

-   Test Accuracy = 81.230%.
-   Test F1-score = 49.075%.
-   Test Precision = 50.369%.
:::

::: {.cell .markdown id="RzQPD32D_oW_"}
`<a id='xgb'>`{=html}`</a>`{=html}

## 1.5 XGBoost Classifier {#15-xgboost-classifier}

XGBoost (Extreme Gradient Boosting) is a popular gradient boosting
framework that is known for its efficiency and effectiveness in solving
machine learning problems. It is based on the gradient boosting
algorithm and provides high predictive performance. The XGBoost
Classifier is used for this particular task.

### Model Hyperparameters {#model-hyperparameters}

The XGBoost classifier used for this task was configured with the
following hyperparameters:

  ----------------------------------------------------------------------------------
  Hyperparameter      Value             Description
  ------------------- ----------------- --------------------------------------------
  n_estimators        110               Number of boosting rounds (decision trees)
                                        to build.

  max_depth           13                Maximum depth of each decision tree.

  learning_rate       0.2               Step size shrinkage used in update to
                                        prevent overfitting.

  objective           binary:logitraw   The loss function to be minimized during
                                        training.

  max_features        None              Maximum number of features to consider for
                                        splitting.

  verbosity           0                 Verbosity mode for printing messages during
                                        training.

  use_label_encoder   False             Whether to use label encoding for target
                                        variables.

  n_jobs              -1                Number of parallel threads to use for
                                        fitting and predicting.

  random_state        0                 Seed value for random number generation.
  ----------------------------------------------------------------------------------

These hyperparameters were chosen based on experimentation and domain
knowledge to achieve the best performance for the given task.
:::

::: {.cell .markdown id="BPxgO6SV_oW_"}
### balanced {#balanced}
:::

::: {.cell .code execution_count="69" id="6-VYDyaX_oW_" outputId="d728adba-510f-4c4b-eada-fc8bf23e7b9c"}
``` python
xg= xgb.XGBClassifier(n_estimators=110, max_depth=13,
                    learning_rate=0.2,
                    objective='binary:logitraw',
                    max_features=None,
                    verbosity  = 0,

                    use_label_encoder=False,
                    n_jobs=-1,
                   random_state  = 0 )

xg.fit(X_train_sm,y_train_sm)

y_pred_xg= xg.predict(X_test_sm)
probas_xg=xg.predict_proba(X_test_sm)
model_val(y_test_sm, y_pred_xg, probas_xg[:,1])
```

::: {.output .stream .stdout}
    AUC 0.9963331891947907
    the recall for this model is : 0.9851258581235698
    TP 54243
    TN 42421
    FP 2091
    FN 819

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98106   0.95302   0.96684     44512
               1    0.96288   0.98513   0.97388     55062

        accuracy                        0.97078     99574
       macro avg    0.97197   0.96907   0.97036     99574
    weighted avg    0.97101   0.97078   0.97073     99574

                       pre       rec       spe        f1       geo       iba       sup

              0    0.98106   0.95302   0.98513   0.96684   0.96894   0.93583     44512
              1    0.96288   0.98513   0.95302   0.97388   0.96894   0.94186     55062

    avg / total    0.97101   0.97078   0.96737   0.97073   0.96894   0.93917     99574
:::
:::

::: {.cell .code execution_count="71"}
``` python
roc_png(proba =probas_xg[:,1] ,y=y_test_sm, name = 'ROC Curve for XGB')
```

::: {.output .display_data}
![](vertopal_af71f06758214898b489638f50e60adb/e91a0bc66dedcc485e2c34a757819b47b26ca7e1.png)
:::
:::

::: {.cell .markdown}
**XGBoost results:**

-   Test Accuracy = 97.078%.
-   Test F1-score = 97.036%.
-   Test Precision = 97.197%.
:::

::: {.cell .markdown id="3tHXDhiZ_oW_"}
### unbalanced {#unbalanced}
:::

::: {.cell .code execution_count="70" id="DdCVQ0YT_oW_" outputId="bcfbb32a-9b11-4810-baa6-bb52f8004c4d"}
``` python
xg_clean = xgb.XGBClassifier(n_estimators=110, max_depth=13,
                    learning_rate=0.2,
                    objective='binary:logitraw',
                    max_features=None,
                    verbosity  = 0,

                    # tree_method = 'gpu_hist',
                    use_label_encoder=False,
                    n_jobs=-1,
                   random_state  = 0 )

xg_clean.fit(X_train_clean,y_train_clean)

y_pred_xg_clean = xg.predict(X_test_clean)
probas_xg_clean = xg.predict_proba(X_test_clean)
model_val(y_test_clean, y_pred_xg_clean, probas_xg_clean[:,1])
```

::: {.output .stream .stdout}
    AUC 0.9757214879569434
    the recall for this model is : 0.9851258581235698
    TP 54243
    TN 2033
    FP 929
    FN 819

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.71283   0.68636   0.69935      2962
               1    0.98316   0.98513   0.98414     55062

        accuracy                        0.96987     58024
       macro avg    0.84800   0.83574   0.84174     58024
    weighted avg    0.96936   0.96987   0.96960     58024

                       pre       rec       spe        f1       geo       iba       sup

              0    0.71283   0.68636   0.98513   0.69935   0.82228   0.65595      2962
              1    0.98316   0.98513   0.68636   0.98414   0.82228   0.69635     55062

    avg / total    0.96936   0.96987   0.70161   0.96960   0.82228   0.69429     58024
:::
:::

::: {.cell .markdown id="Up_AXoaZAibh"}
`<a id='rf'>`{=html}`</a>`{=html}

## 1.6 Random Forest {#16-random-forest}

Random Forest is an ensemble learning method that combines multiple
decision trees to make predictions. Each decision tree in the ensemble
is built on a randomly sampled subset of the training data, and the
final prediction is determined by aggregating the predictions of
individual trees.

### Model Hyperparameters {#model-hyperparameters}

The Optimal Random Forest classifier used for this task was configured
with the following hyperparameters:

  --------------------------------------------------------------------------
  Hyperparameter      Value     Description
  ------------------- --------- --------------------------------------------
  n_estimators        100       Number of decision trees in the random
                                forest.

  max_depth           32        Maximum depth of each decision tree.

  min_samples_leaf    1         Minimum number of samples required in a leaf
                                node.

  max_features        None      Maximum number of features to consider for
                                splitting.

  random_state        0         Seed value for random number generation.

  criterion           entropy   The function to measure the quality of a
                                split.

  min_samples_split   2         Minimum number of samples required to split
                                an internal node.

  bootstrap           True      Whether to use bootstrap samples when
                                building trees.

  n_jobs              -1        Number of jobs to run in parallel for
                                fitting and predicting.
  --------------------------------------------------------------------------
:::

::: {.cell .markdown id="K6L_Fp7SAwf_"}
### balanced {#balanced}
:::

::: {.cell .code execution_count="12" id="wO2jxQIdAYVT"}
``` python
rf = RandomForestClassifier(n_estimators=17, max_depth=19, min_samples_leaf=1,
                               max_features=None, random_state=0,
                               criterion='entropy', min_samples_split=2,
                               bootstrap=True,n_jobs=-1)
```
:::

::: {.cell .code execution_count="13" id="e5f745ce" outputId="3ff9d2b0-3dad-498f-ab44-cb5e02efee01"}
``` python
pred_proba, pred = model(rf,X_train_sm,X_test_sm,y_train_sm,y_test_sm)
```

::: {.output .stream .stdout}
    AUC 0.9842557341608325
    TP 53024
    TN 40169
    FP 4343
    FN 2038

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.95171   0.90243   0.92642     44512
               1    0.92429   0.96299   0.94324     55062

        accuracy                        0.93592     99574
       macro avg    0.93800   0.93271   0.93483     99574
    weighted avg    0.93655   0.93592   0.93572     99574
:::
:::

::: {.cell .code execution_count="68" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":165}" id="8874837b" outputId="9db0ad3a-dad4-4b2d-fea0-3dd84b71d7cb"}
``` python
roc_png(proba =pred_proba[:,1] ,y=y_test_sm, name = 'ROC Curve for Random forest')
```

::: {.output .display_data}
![](vertopal_af71f06758214898b489638f50e60adb/39e83f14763c45a5e6007a63ed6f41ab0c9c41e1.png)
:::
:::

::: {.cell .markdown}
**Random Forest results:**

-   Test Accuracy = 93.592%.
-   Test F1-score = 93.483%.
-   Test Precision = 93.800%.
:::

::: {.cell .markdown id="sNwzzFkMA0tD"}
### unbalanced {#unbalanced}
:::

::: {.cell .code execution_count="66"}
``` python
rf_clean = RandomForestClassifier(n_estimators=17, max_depth=19, min_samples_leaf=1,
                               max_features=None, random_state=0,
                               criterion='entropy', min_samples_split=2,
                               bootstrap=True,n_jobs=-1)
```
:::

::: {.cell .code execution_count="67"}
``` python
rf_pred_proba_clean, rf_pred_clean = model(rf_clean,X_train_clean,X_test_clean,y_train_clean,y_test_clean)
```

::: {.output .stream .stdout}
    AUC 0.9529085909687567
    TP 55053
    TN 432
    FP 2530
    FN 9

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97959   0.14585   0.25389      2962
               1    0.95606   0.99984   0.97746     55062

        accuracy                        0.95624     58024
       macro avg    0.96783   0.57284   0.61568     58024
    weighted avg    0.95726   0.95624   0.94052     58024
:::
:::

::: {.cell .markdown}
## 1.7 Optimal Random Forest {#17-optimal-random-forest}
:::

::: {.cell .markdown}
### balanced {#balanced}
:::

::: {.cell .code execution_count="29" id="9fbea124" scrolled="true"}
``` python
rf_op  = BalancedBaggingClassifier(AdaBoostClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100, max_depth=32, min_samples_leaf=1,
                               max_features=None, random_state=0,
                               criterion='entropy', min_samples_split=2,
                               bootstrap=True),),n_jobs=-1)
```
:::

::: {.cell .code execution_count="30" id="f4a28ded" outputId="d52cef1f-8f77-4c45-c3fd-349caaa1469f"}
``` python
rf_pred_proba2, rf_pred2 = model(rf_op,X_train_sm,X_test_sm,y_train_sm,y_test_sm)
```

::: {.output .stream .stdout}
    AUC 0.9908501793847395
    TP 53783
    TN 40723
    FP 3789
    FN 1279

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.96955   0.91488   0.94142     44512
               1    0.93419   0.97677   0.95500     55062

        accuracy                        0.94910     99574
       macro avg    0.95187   0.94582   0.94821     99574
    weighted avg    0.94999   0.94910   0.94893     99574

    [[1.43405127e-01 8.56594873e-01]
     [1.20727119e-01 8.79272881e-01]
     [1.21167334e-01 8.78832666e-01]
     ...
     [6.96693542e-01 3.03306458e-01]
     [9.99999998e-01 1.90380206e-09]
     [6.85601461e-01 3.14398539e-01]] [1 1 1 ... 0 0 0]
:::
:::

::: {.cell .code execution_count="32" id="24e730af" outputId="d95b6ba7-5f29-444f-8e1e-c6076e3d9ee1"}
``` python
roc_png(proba =rf_pred_proba2[:,1] ,y=y_test_sm, name = 'ROC Curve for Random forest & ada')
```

::: {.output .display_data}
![](vertopal_af71f06758214898b489638f50e60adb/60195ff292a90638482fc2abd93cbe9fec13eb4d.png)
:::
:::

::: {.cell .markdown}
**Optimal Random Forest results:**

-   Test Accuracy = 94.910%.
-   Test F1-score = 95.737%.
-   Test Precision = 95.880%.
:::

::: {.cell .markdown}
### unbalanced {#unbalanced}
:::

::: {.cell .code execution_count="37"}
``` python
rf_op_clean  = BalancedBaggingClassifier(AdaBoostClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100, max_depth=32, min_samples_leaf=1,
                               max_features=None, random_state=0,
                               criterion='entropy', min_samples_split=2,
                               bootstrap=True),),n_jobs=-1)
```
:::

::: {.cell .code execution_count="38"}
``` python
rf_pred_proba2_clean, rf_pred2_clean = model(rf_op_clean,X_train_clean,X_test_clean,y_train_clean,y_test_clean)
```

::: {.output .stream .stdout}
    AUC 0.8985513776367643
    TP 43992
    TN 2413
    FP 549
    FN 11070

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.17897   0.81465   0.29346      2962
               1    0.98767   0.79895   0.88335     55062

        accuracy                        0.79976     58024
       macro avg    0.58332   0.80680   0.58840     58024
    weighted avg    0.94639   0.79976   0.85323     58024
:::
:::

::: {.cell .markdown id="hPCChSPj_oW_"}
`<a id='hy'>`{=html}`</a>`{=html}

## Hybrid models
:::

::: {.cell .markdown id="DGQ-AZTq_oXA"}
## hybrid MLP with Optimal DT 2.1 {#hybrid-mlp-with-optimal-dt-21}

split = 80% for DT and 20% for MLP
:::

::: {.cell .code id="jwq-d_8j_oXA"}
``` python
y_mlp = probas_MLP[:,1]
y_dt = probas_ada[:,1]

y_preds_a = (y_mlp * .2) + (y_dt* .8)
```
:::

::: {.cell .code id="_6vN2UZD_oXA" outputId="a5aed6aa-c3ee-43c9-e84b-bde2036d83a8"}
``` python
y_preds = (y_preds_a > .5 ).astype(int)
model_val(y_test_sm, y_preds,y_preds_a)
```

::: {.output .stream .stdout}
    AUC 0.9878937994307496
    the recall for this model is : 0.9556681558969888
    TP 52621
    TN 41458
    FP 3054
    FN 2441

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.94440   0.93139   0.93785     44512
               1    0.94515   0.95567   0.95038     55062

        accuracy                        0.94481     99574
       macro avg    0.94477   0.94353   0.94411     99574
    weighted avg    0.94481   0.94481   0.94478     99574

                       pre       rec       spe        f1       geo       iba       sup

              0    0.94440   0.93139   0.95567   0.93785   0.94345   0.88794     44512
              1    0.94515   0.95567   0.93139   0.95038   0.94345   0.89226     55062

    avg / total    0.94481   0.94481   0.94224   0.94478   0.94345   0.89033     99574
:::
:::

::: {.cell .code id="_oOCZakN_oXA" outputId="8b75124c-9311-4fa9-9f57-502a76a7452a"}
``` python
y_mlp_clean = probas_mlp_clean[:,1]
y_dt_clean = probas_ada_clean[:,1]

y_preds_a_clean = (y_mlp_clean * .2) + (y_dt_clean* .8)
y_preds_clean = (y_preds_a_clean > .5 ).astype(int)
model_val(y_test_clean, y_preds_clean,y_preds_a_clean)
```

::: {.output .stream .stdout}
    AUC 0.9116698870251497
    the recall for this model is : 0.966528640441684
    TP 53219
    TN 1507
    FP 1455
    FN 1843

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.44985   0.50878   0.47750      2962
               1    0.97339   0.96653   0.96995     55062

        accuracy                        0.94316     58024
       macro avg    0.71162   0.73765   0.72372     58024
    weighted avg    0.94666   0.94316   0.94481     58024

                       pre       rec       spe        f1       geo       iba       sup

              0    0.44985   0.50878   0.96653   0.47750   0.70125   0.46924      2962
              1    0.97339   0.96653   0.50878   0.96995   0.70125   0.51426     55062

    avg / total    0.94666   0.94316   0.53215   0.94481   0.70125   0.51196     58024
:::
:::

::: {.cell .code id="a6zdL6Nf_oXB" outputId="829e5854-882d-48af-fa7c-a2212cb72f46"}
``` python
roc_png(proba =y_preds_a,y=y_test_sm, name = 'ROC Curve for DT with MLP')
```

::: {.output .display_data}
![](vertopal_af71f06758214898b489638f50e60adb/5d40531a5ea77b18c251d8cdfc66b8aeb4e53cc5.png)
:::
:::

::: {.cell .markdown id="ViECvrRw_oXB"}
**hybrid MLP & Optimal DT results:**

-   Test Accuracy = 95.807%.
-   Test F1-score = 95.737%.
-   Test Precision = 95.880%.
:::

::: {.cell .markdown id="4kCtfLpN_oXB"}
## hybrid DRNN with Optimal DT 2.2 {#hybrid-drnn-with-optimal-dt-22}

split = 40% for DT and 60% for DRNN
:::

::: {.cell .code id="J8aP8El-_oXB"}
``` python
y_rnn = probas_rnn[:,1]
y_dt = probas_ada[:,1]

y_preds_Dt_rnn = (y_rnn * .6) + (y_dt* .4)
```
:::

::: {.cell .code id="dLUbsRGM_oXB" outputId="2b891565-862e-49e1-cdf4-ee69124d1fbb"}
``` python
y_preds_rnn_dt = (y_preds_Dt_rnn > .5 ).astype(int)
model_val(y_test_sm, y_preds_rnn_dt,y_preds_Dt_rnn)
```

::: {.output .stream .stdout}
    AUC 0.9956017054306288
    the recall for this model is : 0.9990192873488069
    TP 55008
    TN 41528
    FP 2984
    FN 54

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99870   0.93296   0.96471     44512
               1    0.94854   0.99902   0.97313     55062

        accuracy                        0.96949     99574
       macro avg    0.97362   0.96599   0.96892     99574
    weighted avg    0.97097   0.96949   0.96937     99574

                       pre       rec       spe        f1       geo       iba       sup

              0    0.99870   0.93296   0.99902   0.96471   0.96543   0.92589     44512
              1    0.94854   0.99902   0.93296   0.97313   0.96543   0.93820     55062

    avg / total    0.97097   0.96949   0.96249   0.96937   0.96543   0.93270     99574
:::
:::

::: {.cell .markdown id="yX7GPTSL_oXB"}
**hybrid DRNN & Optimal DT results:**

-   Test Accuracy = 96.949%.
-   Test F1-score = 96.892%.
-   Test Precision = 97.370%.
:::

::: {.cell .markdown}
## hybrid Random Forest with XGB 2.3 {#hybrid-random-forest-with-xgb-23}
:::

::: {.cell .markdown}
split = 60% xgb and 30% for DRNN

### balanced {#balanced}
:::

::: {.cell .code execution_count="73"}
``` python
y_xgb = probas_xg[:,1]
y_rf = rf_pred_proba2[:,1]

y_pred_rf_xgb = (y_xgb * .6) + (y_rf* .3)
y_preds_rf_xgb = (y_pred_rf_xgb > .5 ).astype(int)
model_val(y_test_sm, y_preds_rf_xgb,y_pred_rf_xgb)
```

::: {.output .stream .stdout}
    AUC 0.9963571928367476
    the recall for this model is : 0.9843449202716937
    TP 54200
    TN 42494
    FP 2018
    FN 862

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98012   0.95466   0.96722     44512
               1    0.96410   0.98434   0.97412     55062

        accuracy                        0.97108     99574
       macro avg    0.97211   0.96950   0.97067     99574
    weighted avg    0.97126   0.97108   0.97104     99574

                       pre       rec       spe        f1       geo       iba       sup

              0    0.98012   0.95466   0.98434   0.96722   0.96939   0.93693     44512
              1    0.96410   0.98434   0.95466   0.97412   0.96939   0.94251     55062

    avg / total    0.97126   0.97108   0.96793   0.97104   0.96939   0.94001     99574
:::
:::

::: {.cell .markdown}
### unbalanced {#unbalanced}
:::

::: {.cell .code execution_count="74"}
``` python
y_xgb_clean = probas_xg_clean[:,1]
y_rf_clean = rf_pred_proba2_clean[:,1]

y_pred_rf_xgb_clean = (y_xgb_clean * .6) + (y_rf_clean* .3)
y_preds_rf_xgb_clean = (y_pred_rf_xgb_clean > .5 ).astype(int)
model_val(y_test_clean, y_preds_rf_xgb_clean,y_pred_rf_xgb_clean)
```

::: {.output .stream .stdout}
    AUC 0.9758848296994334
    the recall for this model is : 0.9832370782027533
    TP 54139
    TN 2088
    FP 874
    FN 923

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.69346   0.70493   0.69915      2962
               1    0.98411   0.98324   0.98367     55062

        accuracy                        0.96903     58024
       macro avg    0.83879   0.84408   0.84141     58024
    weighted avg    0.96928   0.96903   0.96915     58024

                       pre       rec       spe        f1       geo       iba       sup

              0    0.69346   0.70493   0.98324   0.69915   0.83253   0.67382      2962
              1    0.98411   0.98324   0.70493   0.98367   0.83253   0.71240     55062

    avg / total    0.96928   0.96903   0.71914   0.96915   0.83253   0.71043     58024
:::
:::

::: {.cell .markdown id="1GsbB48e_oXC"}
## hybrid DRNN with XGB 2.4 {#hybrid-drnn-with-xgb-24}
:::

::: {.cell .markdown id="O-U4SlLt_oXC"}
split = 35% xgb and 65% for DRNN
:::

::: {.cell .markdown id="B6jsjzpX_oXC"}
### balanced {#balanced}
:::

::: {.cell .code id="C3DO9NH6_oXC"}
``` python
y_xg = probas_xg[:,1]
y_rnn = probas_rnn[:,1]

y_preds_a = (y_xg  * .35) + (y_rnn* .65)
```
:::

::: {.cell .code id="p47f2chV_oXC" outputId="80a7a73c-e095-4a92-b191-6a5d4e0ce8ca"}
``` python
y_preds_rnn_xg = (y_preds_a > .5 ).astype(int)
print(y_preds_a)
model_val(y_test_sm,y_preds_rnn_xg, y_preds_a)
```

::: {.output .stream .stdout}
    [ 1.7993357   1.1106367   2.191417   ... -1.2282809  -1.8141944
     -0.24791579]
    AUC 0.9979924397720303
    the recall for this model is : 0.9947513711815771
    TP 54773
    TN 42518
    FP 1994
    FN 289

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99325   0.95520   0.97385     44512
               1    0.96487   0.99475   0.97958     55062

        accuracy                        0.97707     99574
       macro avg    0.97906   0.97498   0.97672     99574
    weighted avg    0.97756   0.97707   0.97702     99574

                       pre       rec       spe        f1       geo       iba       sup

              0    0.99325   0.95520   0.99475   0.97385   0.97478   0.94643     44512
              1    0.96487   0.99475   0.95520   0.97958   0.97478   0.95395     55062

    avg / total    0.97756   0.97707   0.97288   0.97702   0.97478   0.95059     99574
:::
:::

::: {.cell .code id="_E_IOaDT_oXC" outputId="1d056de2-cf53-4ee0-fbe6-e4ae58831fec"}
``` python
roc_png(proba =y_preds_a ,y=y_test_sm, name = 'ROC Curve for DRNN with XGB' )
```

::: {.output .display_data}
![](vertopal_af71f06758214898b489638f50e60adb/d2aad12fc2c3762434e1e298c3402d9fef729ed5.png)
:::
:::

::: {.cell .markdown id="5C0XkuzC_oXD"}
### unbalanced {#unbalanced}
:::

::: {.cell .code id="DYiYNgeT_oXD"}
``` python
y_xg_clean = probas_xg_clean[:,1]
y_rnn_clean = probas_rnn_clean[:,1]

y_preds_a_clean = (y_xg_clean  * .35) + (y_rnn_clean* .65)
```
:::

::: {.cell .code id="xoakhvk8_oXD" outputId="d8ac8f01-7d25-4039-9cd1-f0a91e6cfab6" scrolled="false"}
``` python
y_preds_rnn_xg_clean = (y_preds_a_clean > .5 ).astype(int)
model_val(y_test_clean,y_preds_rnn_xg_clean, y_preds_a_clean)
```

::: {.output .stream .stdout}
    AUC 0.9679684390398439
    the recall for this model is : 0.9922814282082016
    TP 54637
    TN 1561
    FP 1401
    FN 425

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

             0.0    0.78600   0.52701   0.63096      2962
             1.0    0.97500   0.99228   0.98356     55062

        accuracy                        0.96853     58024
       macro avg    0.88050   0.75965   0.80726     58024
    weighted avg    0.96535   0.96853   0.96556     58024

                       pre       rec       spe        f1       geo       iba       sup

            0.0    0.78600   0.52701   0.99228   0.63096   0.72315   0.49861      2962
            1.0    0.97500   0.99228   0.52701   0.98356   0.72315   0.54727     55062

    avg / total    0.96535   0.96853   0.55076   0.96556   0.72315   0.54479     58024
:::
:::

::: {.cell .markdown id="5KIpGj31_oXD"}
`<a id='val'>`{=html}`</a>`{=html}

## Validation
:::

::: {.cell .markdown id="O8h6ebSb_oXD"}
## DT & MLP {#dt--mlp}
:::

::: {.cell .code id="6S3XV9Q7_oXD" outputId="d06ed2c0-36eb-439c-ae3c-6e61003840ec" scrolled="true"}
``` python
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, precision_score

# some hyper parameters
SEED = 1970
test_train_split_SEED = 1970
# FOLDS = 10
show_fold_stats = True
VERBOSE = 0
FOLDS = 10
# Lets put aside a small test set, so we can check performance of different classifiers against it
disease_train, disease_test, disease_y_train, disease_y_test =X_train_sm,X_test_sm,y_train_sm,y_test_sm
dieases_train_sc =sc_train
disease_test_sc = sc_test

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
lw = 2

mean_tpr_hybrid = 0.0
mean_fpr_hybrid = np.linspace(0, 1, 100)
lw_hybrid = 2

score_array_hybrid=[]
accuracy_array_hybrid=[]

score_array=[]
accuracy_array=[]
i = 0

skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = SEED)

for fold, (idxT,idxV) in enumerate(skf.split(disease_train, disease_y_train)):

    X_train = disease_train.iloc[idxT]
    X_val = disease_train.iloc[idxV]
    X_val_sc = dieases_train_sc[idxV]
    y_train = disease_y_train[idxT]
    y_val = disease_y_train[idxV]

    X_train_sc = dieases_train_sc[idxT]


    clf  = BalancedBaggingClassifier(
    AdaBoostClassifier(DecisionTreeClassifier(ccp_alpha =3.49297098164335e-04 , random_state=1),
                           random_state=1,n_estimators=120), n_jobs=-1)
    clf.fit(X_train, y_train)

    RF_pred_class = clf.predict(X_val)
    RF_preds = clf.predict_proba(X_val)

    RF_AUC_test_score = roc_auc_score(y_val, RF_preds[:,1])
    RF_f1_test = f1_score(y_val, RF_pred_class)
    RF_recall_test = recall_score(y_val, RF_pred_class)
    RF_precision_test = precision_score(y_val, RF_pred_class)
    RF_accuracy_score_test  = accuracy_score(y_val,RF_pred_class)
    if show_fold_stats:
        print('-' * 80)
        print('Fold : %s'%(fold+1))
        print('TRAIN ROC AUC score for DT model, validation set: %.4f'%RF_AUC_test_score)
        print('F1 : %.4f, Recall : %.4f , Precision : %.4f , Accuracy : %.4f'%(RF_f1_test, RF_recall_test, RF_precision_test,RF_accuracy_score_test))
        print(confusion_matrix(y_val, RF_pred_class))
        print("\n----------Classification Report------------------------------------")
        print(classification_report(y_val,RF_pred_class, digits=5))
        print('-'*90)

    XGB_model = BalancedBaggingClassifier(
    MLPClassifier(solver='adam', activation= 'relu', hidden_layer_sizes=120, learning_rate_init=0.0015 ), n_jobs=-1)

    XGB_model.fit(X_train_sc, y_train)

    XGB_preds = XGB_model.predict_proba(X_val_sc)
    XGB_class = XGB_model.predict(X_val_sc)

    XGB_score = roc_auc_score(y_val, XGB_preds[:,1])
    XGB_f1 = f1_score(y_val, XGB_class)
    XGB_recall = recall_score(y_val, XGB_class)
    XGB_precision = precision_score(y_val, XGB_class)
    XGB_accuracy_score_test  = accuracy_score(y_val,XGB_class)


    if show_fold_stats:
        print('TRAIN ROC AUC score for MLP model, validation set: %.4f'%XGB_score)
        print('F1 : %.4f, Recall : %.4f , Precision : %.4f ,  Accuracy : %.4f  '%(XGB_f1, XGB_recall, XGB_precision,XGB_accuracy_score_test))
        print(confusion_matrix(y_val, XGB_class))
        print("\n----------Classification Report------------------------------------")
        print(classification_report(y_val,XGB_class, digits=5))
        print('-'*90)

    RF_preds_test = clf.predict_proba(disease_test)
    XGB_preds_test = XGB_model.predict_proba(disease_test_sc)
    XGB_class_test = XGB_model.predict(disease_test_sc)
    RF_class_test = clf.predict(disease_test)
    y_mlp = XGB_preds_test[:,1]
    y_dt = RF_preds_test[:,1]

    print("-"*45,"mlp test ","-"*45)
    print('F1 : %.4f, Recall : %.4f , Precision : %.4f ,  Accuracy : %.4f '%(f1_score(disease_y_test, XGB_class_test), recall_score(disease_y_test, XGB_class_test), precision_score(disease_y_test, XGB_class_test),accuracy_score(disease_y_test,XGB_class_test)))
    print(confusion_matrix(disease_y_test,  XGB_class_test))
    print("\n----------Classification Report------------------------------------")
    print(classification_report(disease_y_test, XGB_class_test, digits=5))
    print('-'*90)


    print("-"*45,"dt test ","-"*45)
    print('F1 : %.4f, Recall : %.4f , Precision : %.4f ,  Accuracy : %.4f '%(f1_score(disease_y_test, RF_class_test), recall_score(disease_y_test, RF_class_test), precision_score(disease_y_test, RF_class_test),accuracy_score(disease_y_test,RF_class_test)))
    print(confusion_matrix(disease_y_test, RF_class_test))
    print("\n----------Classification Report------------------------------------")
    print(classification_report(disease_y_test,RF_class_test, digits=5))
    print('-'*90)


    avg_preds_test = (y_mlp * .2) + (y_dt* .8)
    print("-"*45,"hybrid","-"*45)
    RF_test_AUC = roc_auc_score(disease_y_test, RF_preds_test[:,1])
    print('ROC AUC score for DT for test set: %.4f'%RF_test_AUC)
    XGB_test_AUC = roc_auc_score(disease_y_test, XGB_preds_test[:,1])
    print('ROC AUC score for MLP model test set: %.4f'%XGB_test_AUC)
    average_AUC = roc_auc_score(disease_y_test, avg_preds_test )
    print('ROC AUC score hybrid model test set: %.4f'%average_AUC)
    y_mlp_class = XGB_class_test
    y_dt_class = RF_class_test
    avg_class = (y_mlp_class * .2) + (y_dt_class* .8)
    avg_class = (avg_class > .5 ).astype(int)
    print('F1 : %.4f, Recall : %.4f , Precision : %.4f ,  Accuracy : %.4f '%(f1_score(disease_y_test, avg_class), recall_score(disease_y_test, avg_class), precision_score(disease_y_test, avg_class),accuracy_score(disease_y_test,avg_class)))
    print(confusion_matrix(disease_y_test, avg_class))
    print("\n----------Classification Report------------------------------------")
    print(classification_report(disease_y_test,avg_class, digits=5))
    print('-'*90)
```

::: {.output .stream .stdout}
    --------------------------------------------------------------------------------
    Fold : 1
    TRAIN ROC AUC score for DT model, validation set: 0.9974
    F1 : 0.9823, Recall : 0.9908 , Precision : 0.9740 , Accuracy : 0.9810
    [[ 9359   291]
     [  101 10907]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98932   0.96984   0.97949      9650
               1    0.97401   0.99082   0.98235     11008

        accuracy                        0.98102     20658
       macro avg    0.98167   0.98033   0.98092     20658
    weighted avg    0.98117   0.98102   0.98101     20658

    ------------------------------------------------------------------------------------------
    TRAIN ROC AUC score for MLP model, validation set: 0.9882
    F1 : 0.9565, Recall : 0.9714 , Precision : 0.9420 ,  Accuracy : 0.9529  
    [[ 8992   658]
     [  315 10693]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.96615   0.93181   0.94867      9650
               1    0.94203   0.97138   0.95648     11008

        accuracy                        0.95290     20658
       macro avg    0.95409   0.95160   0.95258     20658
    weighted avg    0.95330   0.95290   0.95283     20658

    ------------------------------------------------------------------------------------------
    --------------------------------------------- mlp test  ---------------------------------------------
    F1 : 0.9563, Recall : 0.9799 , Precision : 0.9338 ,  Accuracy : 0.9505 
    [[40689  3823]
     [ 1109 53929]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97347   0.91411   0.94286     44512
               1    0.93380   0.97985   0.95627     55038

        accuracy                        0.95046     99550
       macro avg    0.95364   0.94698   0.94956     99550
    weighted avg    0.95154   0.95046   0.95027     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- dt test  ---------------------------------------------
    F1 : 0.9586, Recall : 0.9956 , Precision : 0.9242 ,  Accuracy : 0.9524 
    [[40018  4494]
     [  244 54794]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99394   0.89904   0.94411     44512
               1    0.92420   0.99557   0.95856     55038

        accuracy                        0.95241     99550
       macro avg    0.95907   0.94730   0.95133     99550
    weighted avg    0.95538   0.95241   0.95210     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- hybrid ---------------------------------------------
    ROC AUC score for DT for test set: 0.9943
    ROC AUC score for MLP model test set: 0.9857
    ROC AUC score hybrid model test set: 0.9930
    F1 : 0.9586, Recall : 0.9956 , Precision : 0.9242 ,  Accuracy : 0.9524 
    [[40018  4494]
     [  244 54794]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99394   0.89904   0.94411     44512
               1    0.92420   0.99557   0.95856     55038

        accuracy                        0.95241     99550
       macro avg    0.95907   0.94730   0.95133     99550
    weighted avg    0.95538   0.95241   0.95210     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Fold : 2
    TRAIN ROC AUC score for DT model, validation set: 0.9970
    F1 : 0.9812, Recall : 0.9883 , Precision : 0.9741 , Accuracy : 0.9798
    [[ 9361   289]
     [  129 10879]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98641   0.97005   0.97816      9650
               1    0.97412   0.98828   0.98115     11008

        accuracy                        0.97977     20658
       macro avg    0.98026   0.97917   0.97966     20658
    weighted avg    0.97986   0.97977   0.97975     20658

    ------------------------------------------------------------------------------------------
    TRAIN ROC AUC score for MLP model, validation set: 0.9879
    F1 : 0.9541, Recall : 0.9666 , Precision : 0.9419 ,  Accuracy : 0.9504  
    [[ 8994   656]
     [  368 10640]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.96069   0.93202   0.94614      9650
               1    0.94193   0.96657   0.95409     11008

        accuracy                        0.95043     20658
       macro avg    0.95131   0.94930   0.95011     20658
    weighted avg    0.95069   0.95043   0.95038     20658

    ------------------------------------------------------------------------------------------
    --------------------------------------------- mlp test  ---------------------------------------------
    F1 : 0.9547, Recall : 0.9791 , Precision : 0.9314 ,  Accuracy : 0.9486 
    [[40545  3967]
     [ 1151 53887]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97240   0.91088   0.94063     44512
               1    0.93143   0.97909   0.95466     55038

        accuracy                        0.94859     99550
       macro avg    0.95191   0.94498   0.94765     99550
    weighted avg    0.94975   0.94859   0.94839     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- dt test  ---------------------------------------------
    F1 : 0.9578, Recall : 0.9951 , Precision : 0.9232 ,  Accuracy : 0.9515 
    [[39954  4558]
     [  270 54768]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99329   0.89760   0.94302     44512
               1    0.92317   0.99509   0.95778     55038

        accuracy                        0.95150     99550
       macro avg    0.95823   0.94635   0.95040     99550
    weighted avg    0.95452   0.95150   0.95118     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- hybrid ---------------------------------------------
    ROC AUC score for DT for test set: 0.9944
    ROC AUC score for MLP model test set: 0.9849
    ROC AUC score hybrid model test set: 0.9925
    F1 : 0.9578, Recall : 0.9951 , Precision : 0.9232 ,  Accuracy : 0.9515 
    [[39954  4558]
     [  270 54768]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99329   0.89760   0.94302     44512
               1    0.92317   0.99509   0.95778     55038

        accuracy                        0.95150     99550
       macro avg    0.95823   0.94635   0.95040     99550
    weighted avg    0.95452   0.95150   0.95118     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Fold : 3
    TRAIN ROC AUC score for DT model, validation set: 0.9973
    F1 : 0.9829, Recall : 0.9908 , Precision : 0.9751 , Accuracy : 0.9817
    [[ 9372   278]
     [  101 10907]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98934   0.97119   0.98018      9650
               1    0.97515   0.99082   0.98292     11008

        accuracy                        0.98165     20658
       macro avg    0.98224   0.98101   0.98155     20658
    weighted avg    0.98178   0.98165   0.98164     20658

    ------------------------------------------------------------------------------------------
    TRAIN ROC AUC score for MLP model, validation set: 0.9887
    F1 : 0.9561, Recall : 0.9693 , Precision : 0.9432 ,  Accuracy : 0.9526  
    [[ 9008   642]
     [  338 10670]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.96383   0.93347   0.94841      9650
               1    0.94325   0.96930   0.95609     11008

        accuracy                        0.95256     20658
       macro avg    0.95354   0.95138   0.95225     20658
    weighted avg    0.95286   0.95256   0.95250     20658

    ------------------------------------------------------------------------------------------
    --------------------------------------------- mlp test  ---------------------------------------------
    F1 : 0.9552, Recall : 0.9785 , Precision : 0.9330 ,  Accuracy : 0.9493 
    [[40646  3866]
     [ 1182 53856]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97174   0.91315   0.94153     44512
               1    0.93302   0.97852   0.95523     55038

        accuracy                        0.94929     99550
       macro avg    0.95238   0.94584   0.94838     99550
    weighted avg    0.95034   0.94929   0.94911     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- dt test  ---------------------------------------------
    F1 : 0.9583, Recall : 0.9953 , Precision : 0.9239 ,  Accuracy : 0.9521 
    [[40003  4509]
     [  258 54780]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99359   0.89870   0.94377     44512
               1    0.92395   0.99531   0.95830     55038

        accuracy                        0.95211     99550
       macro avg    0.95877   0.94701   0.95104     99550
    weighted avg    0.95509   0.95211   0.95180     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- hybrid ---------------------------------------------
    ROC AUC score for DT for test set: 0.9944
    ROC AUC score for MLP model test set: 0.9850
    ROC AUC score hybrid model test set: 0.9925
    F1 : 0.9583, Recall : 0.9953 , Precision : 0.9239 ,  Accuracy : 0.9521 
    [[40003  4509]
     [  258 54780]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99359   0.89870   0.94377     44512
               1    0.92395   0.99531   0.95830     55038

        accuracy                        0.95211     99550
       macro avg    0.95877   0.94701   0.95104     99550
    weighted avg    0.95509   0.95211   0.95180     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Fold : 4
    TRAIN ROC AUC score for DT model, validation set: 0.9970
    F1 : 0.9808, Recall : 0.9893 , Precision : 0.9724 , Accuracy : 0.9793
    [[ 9341   309]
     [  118 10890]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98753   0.96798   0.97765      9650
               1    0.97241   0.98928   0.98077     11008

        accuracy                        0.97933     20658
       macro avg    0.97997   0.97863   0.97921     20658
    weighted avg    0.97947   0.97933   0.97932     20658

    ------------------------------------------------------------------------------------------
    TRAIN ROC AUC score for MLP model, validation set: 0.9881
    F1 : 0.9550, Recall : 0.9760 , Precision : 0.9348 ,  Accuracy : 0.9510  
    [[ 8901   749]
     [  264 10744]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97119   0.92238   0.94616      9650
               1    0.93483   0.97602   0.95498     11008

        accuracy                        0.95096     20658
       macro avg    0.95301   0.94920   0.95057     20658
    weighted avg    0.95182   0.95096   0.95086     20658

    ------------------------------------------------------------------------------------------
    --------------------------------------------- mlp test  ---------------------------------------------
    F1 : 0.9556, Recall : 0.9839 , Precision : 0.9289 ,  Accuracy : 0.9495 
    [[40369  4143]
     [  884 54154]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97857   0.90692   0.94139     44512
               1    0.92893   0.98394   0.95564     55038

        accuracy                        0.94950     99550
       macro avg    0.95375   0.94543   0.94852     99550
    weighted avg    0.95113   0.94950   0.94927     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- dt test  ---------------------------------------------
    F1 : 0.9582, Recall : 0.9951 , Precision : 0.9238 ,  Accuracy : 0.9519 
    [[39996  4516]
     [  268 54770]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99334   0.89854   0.94357     44512
               1    0.92383   0.99513   0.95815     55038

        accuracy                        0.95194     99550
       macro avg    0.95859   0.94684   0.95086     99550
    weighted avg    0.95491   0.95194   0.95163     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- hybrid ---------------------------------------------
    ROC AUC score for DT for test set: 0.9945
    ROC AUC score for MLP model test set: 0.9855
    ROC AUC score hybrid model test set: 0.9931
    F1 : 0.9582, Recall : 0.9951 , Precision : 0.9238 ,  Accuracy : 0.9519 
    [[39996  4516]
     [  268 54770]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99334   0.89854   0.94357     44512
               1    0.92383   0.99513   0.95815     55038

        accuracy                        0.95194     99550
       macro avg    0.95859   0.94684   0.95086     99550
    weighted avg    0.95491   0.95194   0.95163     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Fold : 5
    TRAIN ROC AUC score for DT model, validation set: 0.9971
    F1 : 0.9822, Recall : 0.9902 , Precision : 0.9743 , Accuracy : 0.9809
    [[ 9363   287]
     [  108 10900]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98860   0.97026   0.97934      9650
               1    0.97435   0.99019   0.98220     11008

        accuracy                        0.98088     20658
       macro avg    0.98147   0.98022   0.98077     20658
    weighted avg    0.98100   0.98088   0.98087     20658

    ------------------------------------------------------------------------------------------
    TRAIN ROC AUC score for MLP model, validation set: 0.9878
    F1 : 0.9543, Recall : 0.9688 , Precision : 0.9402 ,  Accuracy : 0.9505  
    [[ 8972   678]
     [  344 10664]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.96307   0.92974   0.94611      9650
               1    0.94022   0.96875   0.95427     11008

        accuracy                        0.95053     20658
       macro avg    0.95165   0.94925   0.95019     20658
    weighted avg    0.95090   0.95053   0.95046     20658

    ------------------------------------------------------------------------------------------
    --------------------------------------------- mlp test  ---------------------------------------------
    F1 : 0.9538, Recall : 0.9783 , Precision : 0.9305 ,  Accuracy : 0.9476 
    [[40488  4024]
     [ 1193 53845]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97138   0.90960   0.93947     44512
               1    0.93046   0.97832   0.95379     55038

        accuracy                        0.94759     99550
       macro avg    0.95092   0.94396   0.94663     99550
    weighted avg    0.94876   0.94759   0.94739     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- dt test  ---------------------------------------------
    F1 : 0.9589, Recall : 0.9954 , Precision : 0.9250 ,  Accuracy : 0.9528 
    [[40068  4444]
     [  251 54787]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99377   0.90016   0.94465     44512
               1    0.92497   0.99544   0.95891     55038

        accuracy                        0.95284     99550
       macro avg    0.95937   0.94780   0.95178     99550
    weighted avg    0.95574   0.95284   0.95254     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- hybrid ---------------------------------------------
    ROC AUC score for DT for test set: 0.9945
    ROC AUC score for MLP model test set: 0.9847
    ROC AUC score hybrid model test set: 0.9924
    F1 : 0.9589, Recall : 0.9954 , Precision : 0.9250 ,  Accuracy : 0.9528 
    [[40068  4444]
     [  251 54787]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99377   0.90016   0.94465     44512
               1    0.92497   0.99544   0.95891     55038

        accuracy                        0.95284     99550
       macro avg    0.95937   0.94780   0.95178     99550
    weighted avg    0.95574   0.95284   0.95254     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Fold : 6
    TRAIN ROC AUC score for DT model, validation set: 0.9976
    F1 : 0.9844, Recall : 0.9908 , Precision : 0.9780 , Accuracy : 0.9833
    [[ 9405   245]
     [  101 10907]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98938   0.97461   0.98194      9650
               1    0.97803   0.99082   0.98439     11008

        accuracy                        0.98325     20658
       macro avg    0.98370   0.98272   0.98316     20658
    weighted avg    0.98333   0.98325   0.98324     20658

    ------------------------------------------------------------------------------------------
    TRAIN ROC AUC score for MLP model, validation set: 0.9894
    F1 : 0.9597, Recall : 0.9705 , Precision : 0.9492 ,  Accuracy : 0.9566  
    [[ 9078   572]
     [  325 10683]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.96544   0.94073   0.95292      9650
               1    0.94918   0.97048   0.95971     11008

        accuracy                        0.95658     20658
       macro avg    0.95731   0.95560   0.95631     20658
    weighted avg    0.95677   0.95658   0.95654     20658

    ------------------------------------------------------------------------------------------
    --------------------------------------------- mlp test  ---------------------------------------------
    F1 : 0.9548, Recall : 0.9781 , Precision : 0.9326 ,  Accuracy : 0.9488 
    [[40623  3889]
     [ 1203 53835]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97124   0.91263   0.94102     44512
               1    0.93263   0.97814   0.95484     55038

        accuracy                        0.94885     99550
       macro avg    0.95193   0.94539   0.94793     99550
    weighted avg    0.94989   0.94885   0.94866     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- dt test  ---------------------------------------------
    F1 : 0.9581, Recall : 0.9955 , Precision : 0.9235 ,  Accuracy : 0.9519 
    [[39970  4542]
     [  246 54792]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99388   0.89796   0.94349     44512
               1    0.92345   0.99553   0.95814     55038

        accuracy                        0.95190     99550
       macro avg    0.95867   0.94675   0.95081     99550
    weighted avg    0.95494   0.95190   0.95159     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- hybrid ---------------------------------------------
    ROC AUC score for DT for test set: 0.9945
    ROC AUC score for MLP model test set: 0.9848
    ROC AUC score hybrid model test set: 0.9924
    F1 : 0.9581, Recall : 0.9955 , Precision : 0.9235 ,  Accuracy : 0.9519 
    [[39970  4542]
     [  246 54792]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99388   0.89796   0.94349     44512
               1    0.92345   0.99553   0.95814     55038

        accuracy                        0.95190     99550
       macro avg    0.95867   0.94675   0.95081     99550
    weighted avg    0.95494   0.95190   0.95159     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Fold : 7
    TRAIN ROC AUC score for DT model, validation set: 0.9972
    F1 : 0.9823, Recall : 0.9888 , Precision : 0.9759 , Accuracy : 0.9810
    [[ 9381   269]
     [  123 10884]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98706   0.97212   0.97953      9650
               1    0.97588   0.98883   0.98231     11007

        accuracy                        0.98102     20657
       macro avg    0.98147   0.98047   0.98092     20657
    weighted avg    0.98110   0.98102   0.98101     20657

    ------------------------------------------------------------------------------------------
    TRAIN ROC AUC score for MLP model, validation set: 0.9884
    F1 : 0.9555, Recall : 0.9699 , Precision : 0.9415 ,  Accuracy : 0.9519  
    [[ 8987   663]
     [  331 10676]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.96448   0.93130   0.94760      9650
               1    0.94153   0.96993   0.95552     11007

        accuracy                        0.95188     20657
       macro avg    0.95300   0.95061   0.95156     20657
    weighted avg    0.95225   0.95188   0.95182     20657

    ------------------------------------------------------------------------------------------
    --------------------------------------------- mlp test  ---------------------------------------------
    F1 : 0.9550, Recall : 0.9797 , Precision : 0.9316 ,  Accuracy : 0.9490 
    [[40553  3959]
     [ 1119 53919]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97315   0.91106   0.94108     44512
               1    0.93160   0.97967   0.95503     55038

        accuracy                        0.94899     99550
       macro avg    0.95237   0.94536   0.94805     99550
    weighted avg    0.95018   0.94899   0.94879     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- dt test  ---------------------------------------------
    F1 : 0.9595, Recall : 0.9956 , Precision : 0.9258 ,  Accuracy : 0.9535 
    [[40123  4389]
     [  240 54798]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99405   0.90140   0.94546     44512
               1    0.92585   0.99564   0.95947     55038

        accuracy                        0.95350     99550
       macro avg    0.95995   0.94852   0.95247     99550
    weighted avg    0.95634   0.95350   0.95321     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- hybrid ---------------------------------------------
    ROC AUC score for DT for test set: 0.9946
    ROC AUC score for MLP model test set: 0.9851
    ROC AUC score hybrid model test set: 0.9926
    F1 : 0.9595, Recall : 0.9956 , Precision : 0.9258 ,  Accuracy : 0.9535 
    [[40123  4389]
     [  240 54798]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99405   0.90140   0.94546     44512
               1    0.92585   0.99564   0.95947     55038

        accuracy                        0.95350     99550
       macro avg    0.95995   0.94852   0.95247     99550
    weighted avg    0.95634   0.95350   0.95321     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Fold : 8
    TRAIN ROC AUC score for DT model, validation set: 0.9974
    F1 : 0.9814, Recall : 0.9905 , Precision : 0.9725 , Accuracy : 0.9800
    [[ 9342   308]
     [  105 10902]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98889   0.96808   0.97837      9650
               1    0.97252   0.99046   0.98141     11007

        accuracy                        0.98001     20657
       macro avg    0.98070   0.97927   0.97989     20657
    weighted avg    0.98017   0.98001   0.97999     20657

    ------------------------------------------------------------------------------------------
    TRAIN ROC AUC score for MLP model, validation set: 0.9884
    F1 : 0.9553, Recall : 0.9721 , Precision : 0.9390 ,  Accuracy : 0.9515  
    [[ 8955   695]
     [  307 10700]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.96685   0.92798   0.94702      9650
               1    0.93901   0.97211   0.95527     11007

        accuracy                        0.95149     20657
       macro avg    0.95293   0.95004   0.95114     20657
    weighted avg    0.95202   0.95149   0.95142     20657

    ------------------------------------------------------------------------------------------
    --------------------------------------------- mlp test  ---------------------------------------------
    F1 : 0.9552, Recall : 0.9801 , Precision : 0.9315 ,  Accuracy : 0.9492 
    [[40545  3967]
     [ 1094 53944]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97373   0.91088   0.94125     44512
               1    0.93150   0.98012   0.95519     55038

        accuracy                        0.94916     99550
       macro avg    0.95261   0.94550   0.94822     99550
    weighted avg    0.95038   0.94916   0.94896     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- dt test  ---------------------------------------------
    F1 : 0.9584, Recall : 0.9953 , Precision : 0.9242 ,  Accuracy : 0.9523 
    [[40016  4496]
     [  256 54782]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99364   0.89899   0.94395     44512
               1    0.92415   0.99535   0.95843     55038

        accuracy                        0.95227     99550
       macro avg    0.95890   0.94717   0.95119     99550
    weighted avg    0.95522   0.95227   0.95196     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- hybrid ---------------------------------------------
    ROC AUC score for DT for test set: 0.9946
    ROC AUC score for MLP model test set: 0.9850
    ROC AUC score hybrid model test set: 0.9928
    F1 : 0.9584, Recall : 0.9953 , Precision : 0.9242 ,  Accuracy : 0.9523 
    [[40016  4496]
     [  256 54782]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99364   0.89899   0.94395     44512
               1    0.92415   0.99535   0.95843     55038

        accuracy                        0.95227     99550
       macro avg    0.95890   0.94717   0.95119     99550
    weighted avg    0.95522   0.95227   0.95196     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Fold : 9
    TRAIN ROC AUC score for DT model, validation set: 0.9975
    F1 : 0.9831, Recall : 0.9908 , Precision : 0.9754 , Accuracy : 0.9818
    [[ 9375   275]
     [  101 10906]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98934   0.97150   0.98034      9650
               1    0.97540   0.99082   0.98305     11007

        accuracy                        0.98180     20657
       macro avg    0.98237   0.98116   0.98170     20657
    weighted avg    0.98192   0.98180   0.98179     20657

    ------------------------------------------------------------------------------------------
    TRAIN ROC AUC score for MLP model, validation set: 0.9886
    F1 : 0.9540, Recall : 0.9692 , Precision : 0.9392 ,  Accuracy : 0.9502  
    [[ 8960   690]
     [  339 10668]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.96354   0.92850   0.94570      9650
               1    0.93925   0.96920   0.95399     11007

        accuracy                        0.95019     20657
       macro avg    0.95140   0.94885   0.94984     20657
    weighted avg    0.95060   0.95019   0.95012     20657

    ------------------------------------------------------------------------------------------
    --------------------------------------------- mlp test  ---------------------------------------------
    F1 : 0.9552, Recall : 0.9782 , Precision : 0.9333 ,  Accuracy : 0.9493 
    [[40662  3850]
     [ 1200 53838]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97133   0.91351   0.94153     44512
               1    0.93326   0.97820   0.95520     55038

        accuracy                        0.94927     99550
       macro avg    0.95230   0.94585   0.94837     99550
    weighted avg    0.95029   0.94927   0.94909     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- dt test  ---------------------------------------------
    F1 : 0.9581, Recall : 0.9954 , Precision : 0.9234 ,  Accuracy : 0.9518 
    [[39968  4544]
     [  251 54787]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99376   0.89792   0.94341     44512
               1    0.92341   0.99544   0.95807     55038

        accuracy                        0.95183     99550
       macro avg    0.95859   0.94668   0.95074     99550
    weighted avg    0.95487   0.95183   0.95152     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- hybrid ---------------------------------------------
    ROC AUC score for DT for test set: 0.9944
    ROC AUC score for MLP model test set: 0.9851
    ROC AUC score hybrid model test set: 0.9925
    F1 : 0.9581, Recall : 0.9954 , Precision : 0.9234 ,  Accuracy : 0.9518 
    [[39968  4544]
     [  251 54787]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99376   0.89792   0.94341     44512
               1    0.92341   0.99544   0.95807     55038

        accuracy                        0.95183     99550
       macro avg    0.95859   0.94668   0.95074     99550
    weighted avg    0.95487   0.95183   0.95152     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Fold : 10
    TRAIN ROC AUC score for DT model, validation set: 0.9970
    F1 : 0.9812, Recall : 0.9885 , Precision : 0.9741 , Accuracy : 0.9799
    [[ 9361   289]
     [  127 10880]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.98661   0.97005   0.97826      9650
               1    0.97412   0.98846   0.98124     11007

        accuracy                        0.97986     20657
       macro avg    0.98037   0.97926   0.97975     20657
    weighted avg    0.97996   0.97986   0.97985     20657

    ------------------------------------------------------------------------------------------
    TRAIN ROC AUC score for MLP model, validation set: 0.9884
    F1 : 0.9561, Recall : 0.9705 , Precision : 0.9422 ,  Accuracy : 0.9526  
    [[ 8995   655]
     [  325 10682]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.96513   0.93212   0.94834      9650
               1    0.94222   0.97047   0.95614     11007

        accuracy                        0.95256     20657
       macro avg    0.95368   0.95130   0.95224     20657
    weighted avg    0.95292   0.95256   0.95250     20657

    ------------------------------------------------------------------------------------------
    --------------------------------------------- mlp test  ---------------------------------------------
    F1 : 0.9560, Recall : 0.9793 , Precision : 0.9339 ,  Accuracy : 0.9502 
    [[40696  3816]
     [ 1140 53898]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.97275   0.91427   0.94260     44512
               1    0.93388   0.97929   0.95605     55038

        accuracy                        0.95022     99550
       macro avg    0.95332   0.94678   0.94932     99550
    weighted avg    0.95126   0.95022   0.95004     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- dt test  ---------------------------------------------
    F1 : 0.9589, Recall : 0.9951 , Precision : 0.9253 ,  Accuracy : 0.9529 
    [[40091  4421]
     [  270 54768]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99331   0.90068   0.94473     44512
               1    0.92531   0.99509   0.95893     55038

        accuracy                        0.95288     99550
       macro avg    0.95931   0.94789   0.95183     99550
    weighted avg    0.95571   0.95288   0.95258     99550

    ------------------------------------------------------------------------------------------
    --------------------------------------------- hybrid ---------------------------------------------
    ROC AUC score for DT for test set: 0.9942
    ROC AUC score for MLP model test set: 0.9854
    ROC AUC score hybrid model test set: 0.9924
    F1 : 0.9589, Recall : 0.9951 , Precision : 0.9253 ,  Accuracy : 0.9529 
    [[40091  4421]
     [  270 54768]]

    ----------Classification Report------------------------------------
                  precision    recall  f1-score   support

               0    0.99331   0.90068   0.94473     44512
               1    0.92531   0.99509   0.95893     55038

        accuracy                        0.95288     99550
       macro avg    0.95931   0.94789   0.95183     99550
    weighted avg    0.95571   0.95288   0.95258     99550

    ------------------------------------------------------------------------------------------
:::
:::

::: {.cell .markdown id="gFdEKZl3_oXE"}
## DRNN
:::

::: {.cell .code id="pe2KO0Ml_oXE" outputId="1af0a02b-efab-4a48-94be-fe86e9845c9c"}
``` python
CrossValidation(model_rnn,X_train_,y_train_,10)
```

::: {.output .stream .stdout}
    fit_time: 644.19995 (+/- 43.4538)
    score_time: 5.90258 (+/- 0.6561)
    test_accuracy: 0.97041 (+/- 0.1704)
    test_precision: 0.98027 (+/- 0.1112)
    test_recall: 0.97041 (+/- 0.1704)
    test_f1_score: 0.96717 (+/- 0.1899)
:::
:::

::: {.cell .code id="mCmj4U8i_oXE" outputId="aba3d7f8-a656-4157-d1ea-10f42a95aa94"}
``` python
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, precision_score

n_splits = 10

# Initialize the K object
kf = KFold(n_splits=n_splits, shuffle=True)

# Initialize lists to store the evaluation metrics for each fold
acc_scores = []
prec_scores = []
rec_scores = []
f1_scores = []

disease_train, disease_test, disease_y_train, disease_y_test =X_train_sm,X_test_sm,y_train_sm,y_test_sm
dieases_train_sc =sc_train
disease_test_sc = sc_test
SEED = 1970

skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = SEED)
for fold, (idxT,idxV) in enumerate(skf.split(disease_train, disease_y_train)):

    X_train = disease_train.iloc[idxT]
    X_val = disease_train.iloc[idxV]
    X_train_sc = dieases_train_sc[idxT]
    X_val_sc = dieases_train_sc[idxV]
    y_train = disease_y_train[idxT]
    y_val = disease_y_train[idxV]


    # Train the model on the training set for this fold
    xg= xgb.XGBClassifier(n_estimators=110, max_depth=13,
                    learning_rate=0.2,
                    objective='binary:logitraw',
                    max_features=None,
                    verbosity  = 0,

                    # tree_method = 'gpu_hist',
                    use_label_encoder=False,
                    n_jobs=-1,
                   random_state  = 0 )

    xg.fit(X_train,y_train)
    probas_xg=xg.predict_proba(X_val)

    Drnn = BalancedBaggingClassifier(
    MLPClassifier(solver='adam', activation= 'relu', hidden_layer_sizes=120, learning_rate_init=0.0015 ), n_jobs=-1)

    Drnn.fit(X_train_sc, y_train)
    probas_rnn = Drnn.predict_proba(X_val_sc)
    y_xgb = probas_xg[:,1]
    y_drnn = probas_rnn[:,1]

    y_preds_hy = (y_xgb  * .35) + (y_drnn* .65)
    y_preds_drnn_xgb = (y_preds_hy > .5 ).astype(int)

    y_pred = y_preds_drnn_xgb
    y_true = y_val

    # Calculate the evaluation metrics for this fold
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average='macro')
    rec = recall_score(y_val, y_pred, average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')

    # Append the evaluation metrics to the lists
    acc_scores.append(acc)
    prec_scores.append(prec)
    rec_scores.append(rec)
    f1_scores.append(f1)

# Print the average evaluation metrics over all folds
print('Accuracy:', np.mean(acc_scores))
print('Precision:', np.mean(prec_scores))
print('Recall:', np.mean(rec_scores))
print('F1 score:', np.mean(f1_scores))
```

::: {.output .stream .stdout}
    Accuracy: 0.9804330559183096
    Precision: 0.9804524313748377
    Recall: 0.9802408827208108
    F1 score: 0.9803433864500224
:::
:::

::: {.cell .code id="888a1181" outputId="32e3346e-5ea3-4925-fcc2-5d4c3551916d" scrolled="false"}
``` python
CrossValidation(rf,X_train,y_train,10)
```

::: {.output .stream .stdout}
    fit_time: 121.98865 (+/- 28.9948)
    score_time: 0.24519 (+/- 0.0370)
    test_accuracy: 0.95383 (+/- 0.0624)
    test_precision: 0.95526 (+/- 0.0544)
    test_recall: 0.95383 (+/- 0.0624)
    test_f1_score: 0.95361 (+/- 0.0638)
:::
:::

::: {.cell .code id="ab529b42" outputId="7d364a13-2a58-4bea-98db-8b90f994a616" scrolled="false"}
``` python
CrossValidation(rf1,X_train,y_train,10)
```

::: {.output .stream .stdout}
    fit_time: 6316.53706 (+/- 3208.1614)
    score_time: 12.64109 (+/- 7.7462)
    test_accuracy: 0.99350 (+/- 0.0133)
    test_precision: 0.99359 (+/- 0.0128)
    test_recall: 0.99350 (+/- 0.0133)
    test_f1_score: 0.99350 (+/- 0.0134)
:::
:::

::: {.cell .code id="\"00977926\"" outputId="6166aad3-d856-4903-8c68-fd43913c2fe1" scrolled="true"}
``` python
CrossValidation(rf2,X_train,y_train,10)
```

::: {.output .stream .stdout}
    fit_time: 797.57185 (+/- 38.3452)
    score_time: 7.54442 (+/- 4.5054)
    test_accuracy: 0.96351 (+/- 0.0603)
    test_precision: 0.96497 (+/- 0.0526)
    test_recall: 0.96351 (+/- 0.0603)
    test_f1_score: 0.96332 (+/- 0.0615)
:::
:::
