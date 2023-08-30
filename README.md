---
jupyter:
  colab:
    authorship_tag: ABX9TyNJCZgv/DXbV7Vm1Rj1zx7y
    include_colab_link: true
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell markdown" colab_type="text" id="view-in-github">

<a href="https://colab.research.google.com/github/jinrumiao/titanic-analysis/blob/main/%E5%B0%88%E9%A1%8C%E5%AF%A6%E4%BD%9C_01%EF%BC%9A%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E5%AD%98%E6%B4%BB%E9%A0%90%E6%B8%AC.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

</div>

<div class="cell code" execution_count="1" id="1H4G77tIBcha">

``` python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.style.use("ggplot")
sns.set_style("white")
matplotlib.rcParams["figure.figsize"] = 8, 6
```

</div>

<div class="cell code" execution_count="2" id="QqBKGnxZBdKe">

``` python
def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get("row", None)
    col = kwargs.get("col", None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, fill=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()


def plot_catecories(df, cat, target, **kwargs):
    row = kwargs.get("row", None)
    col = kwargs.get("col", None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target)
    facet.add_legend()


def clean_ticket(ticket):
    ticket = ticket.replace(".", "").replace("/", "")
    ticket = ticket.split()
    ticket = map(lambda t: t.strip(), ticket)
    ticket = list(filter(lambda t: not t.isdigit(), ticket))

    if len(ticket) > 0:
        return ticket[0]
    else:
        return "XXX"
```

</div>

<div class="cell code" execution_count="3"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="s70XpfI-BgmB" outputId="2d32c775-96b4-4a96-9fa9-6bf2866eea84">

``` python
# 載入csv
df = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv')

print(df.info())
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    None

</div>

</div>

<div class="cell markdown" id="iELkIgoyBnS3">

**處理離散型資料**

</div>

<div class="cell code" execution_count="4" id="JBaRuNZqBjTF">

``` python
# 1、Name中間包含頭銜，又代表社會地位，可能會與是否生存有關
title_dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Dr": "Officer",
    "Rev": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dona": "Royalty",
    "Lady": "Royalty",
    "the Countess": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
}
df["Title"] = df["Name"].map(lambda name: name.split(", ")[1].split(".")[0].strip())
df["Title"] = df["Title"].map(title_dictionary)

title_one_hot_ec = pd.get_dummies(df["Title"], prefix="Title", dtype=int)
df = pd.concat([df, title_one_hot_ec], axis=1)
```

</div>

<div class="cell code" execution_count="5" id="o53YY07XBr6H">

``` python
# 2、Sex
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
```

</div>

<div class="cell code" execution_count="6" id="a0uJxIvUBuyC">

``` python
# 3、Ticket
df["Ticket_cat"] = df["Ticket"].map(clean_ticket)

ticket_one_hot_ec = pd.get_dummies(df["Ticket_cat"], prefix="Ticket", dtype=int)
df = pd.concat([df, ticket_one_hot_ec], axis=1)
```

</div>

<div class="cell code" execution_count="7" id="yE7IS-r5BxWM">

``` python
# 4、Cabin 204/891
df["Cabin"] = df["Cabin"].fillna("U")
df["Cabin"] = df["Cabin"].astype(str).map(lambda c: c[0])

cabin_one_hot_ec = pd.get_dummies(df["Cabin"], prefix="Cabin", dtype=int)
df = pd.concat([df, cabin_one_hot_ec], axis=1)
```

</div>

<div class="cell code" execution_count="8" id="QaGIvuksB2cI">

``` python
# 5、Embarked 889/891
df["Embarked"] = df["Embarked"].fillna("U")

embarked_one_hot_ec = pd.get_dummies(df["Embarked"], prefix="Embarked", dtype=int)
df = pd.concat([df, embarked_one_hot_ec], axis=1)
```

</div>

<div class="cell code" execution_count="9"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="8kiodrxVB8er" outputId="c4381b72-64f3-49ac-da8c-738adf7324c4">

``` python
df.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 64 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   PassengerId     891 non-null    int64  
     1   Survived        891 non-null    int64  
     2   Pclass          891 non-null    int64  
     3   Name            891 non-null    object 
     4   Sex             891 non-null    int64  
     5   Age             714 non-null    float64
     6   SibSp           891 non-null    int64  
     7   Parch           891 non-null    int64  
     8   Ticket          891 non-null    object 
     9   Fare            891 non-null    float64
     10  Cabin           891 non-null    object 
     11  Embarked        891 non-null    object 
     12  Title           891 non-null    object 
     13  Title_Master    891 non-null    int64  
     14  Title_Miss      891 non-null    int64  
     15  Title_Mr        891 non-null    int64  
     16  Title_Mrs       891 non-null    int64  
     17  Title_Officer   891 non-null    int64  
     18  Title_Royalty   891 non-null    int64  
     19  Ticket_cat      891 non-null    object 
     20  Ticket_A4       891 non-null    int64  
     21  Ticket_A5       891 non-null    int64  
     22  Ticket_AS       891 non-null    int64  
     23  Ticket_C        891 non-null    int64  
     24  Ticket_CA       891 non-null    int64  
     25  Ticket_CASOTON  891 non-null    int64  
     26  Ticket_FC       891 non-null    int64  
     27  Ticket_FCC      891 non-null    int64  
     28  Ticket_Fa       891 non-null    int64  
     29  Ticket_LINE     891 non-null    int64  
     30  Ticket_PC       891 non-null    int64  
     31  Ticket_PP       891 non-null    int64  
     32  Ticket_PPP      891 non-null    int64  
     33  Ticket_SC       891 non-null    int64  
     34  Ticket_SCA4     891 non-null    int64  
     35  Ticket_SCAH     891 non-null    int64  
     36  Ticket_SCOW     891 non-null    int64  
     37  Ticket_SCPARIS  891 non-null    int64  
     38  Ticket_SCParis  891 non-null    int64  
     39  Ticket_SOC      891 non-null    int64  
     40  Ticket_SOP      891 non-null    int64  
     41  Ticket_SOPP     891 non-null    int64  
     42  Ticket_SOTONO2  891 non-null    int64  
     43  Ticket_SOTONOQ  891 non-null    int64  
     44  Ticket_SP       891 non-null    int64  
     45  Ticket_STONO    891 non-null    int64  
     46  Ticket_STONO2   891 non-null    int64  
     47  Ticket_SWPP     891 non-null    int64  
     48  Ticket_WC       891 non-null    int64  
     49  Ticket_WEP      891 non-null    int64  
     50  Ticket_XXX      891 non-null    int64  
     51  Cabin_A         891 non-null    int64  
     52  Cabin_B         891 non-null    int64  
     53  Cabin_C         891 non-null    int64  
     54  Cabin_D         891 non-null    int64  
     55  Cabin_E         891 non-null    int64  
     56  Cabin_F         891 non-null    int64  
     57  Cabin_G         891 non-null    int64  
     58  Cabin_T         891 non-null    int64  
     59  Cabin_U         891 non-null    int64  
     60  Embarked_C      891 non-null    int64  
     61  Embarked_Q      891 non-null    int64  
     62  Embarked_S      891 non-null    int64  
     63  Embarked_U      891 non-null    int64  
    dtypes: float64(2), int64(56), object(6)
    memory usage: 445.6+ KB

</div>

</div>

<div class="cell code" execution_count="10"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:1000}"
id="8XT_TPJfCBqe" outputId="5bff1896-c04a-411d-95d6-08d45a7aa77f">

``` python
plot_catecories(df, cat="Sex", target="Survived")  # 繪出Sex與Survived的關係，女性的生存率遠高於男性
plot_catecories(df, cat="Title", target="Survived")  # 繪出Title與Survived的關係，Mrs與Miss為生存率最高的兩個分類
plot_catecories(df, cat="Ticket_cat", target="Survived")  # 繪出Ticket與Survived的關係，Ticket與生存率並無明顯關係
plot_catecories(df, cat="Cabin", target="Survived")  # 繪出Cabin與Survived的關係，Cabin與生存率並無明顯關係
plot_catecories(df, cat="Embarked", target="Survived")  # 繪出Embarked與Survived的關係，扣除無資料的乘客，C點登船的乘客有較高的生存率

plt.show()
```

<div class="output stream stderr">

    /usr/local/lib/python3.10/dist-packages/seaborn/axisgrid.py:712: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)
    /usr/local/lib/python3.10/dist-packages/seaborn/axisgrid.py:712: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)
    /usr/local/lib/python3.10/dist-packages/seaborn/axisgrid.py:712: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)
    /usr/local/lib/python3.10/dist-packages/seaborn/axisgrid.py:712: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)
    /usr/local/lib/python3.10/dist-packages/seaborn/axisgrid.py:712: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)

</div>

<div class="output display_data">

![](433d1c6b887b84a4bdac671d5bb0aee51450e656.png)

</div>

<div class="output display_data">

![](1211db06dd2de675c53e312aa9e02422799f1d29.png)

</div>

<div class="output display_data">

![](dbe7b8b7995213f46e892293f022e16aec9c1fb1.png)

</div>

<div class="output display_data">

![](6ff53106126b1f4facdae2e242419b2275d8a3f2.png)

</div>

<div class="output display_data">

![](a1758b432d95856b23eda9538d605abebfe152a6.png)

</div>

</div>

<div class="cell markdown" id="YcUkDDJoCgz6">

**處理連續型資料**

-   其中Age的空值有使用mean以及median比較，兩者的最終成果無太大差異
-   Pclass也有測試改用one_hot_encoder的形式，結果也與原本的label_encoder無差異

</div>

<div class="cell code" execution_count="11" id="xHF2_FWiCjTQ">

``` python
# 1、Pclass、Age、Fare
df["Age"].fillna(df["Age"].mean(), inplace=True)

normalize_columns = ["Pclass", "Age", "Fare"]
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df[normalize_columns])
normalized_df = pd.DataFrame(normalized_data, columns=normalize_columns)

df[normalize_columns] = normalized_df
```

</div>

<div class="cell code" execution_count="12" id="b1JIzm1-Cn4p">

``` python
# 2、SibSp、Parch
df["Family_size"] = df["SibSp"] + df["Parch"] + 1

df["Family_Single"] = df["Family_size"].map(lambda s: 1 if s == 1 else 0)
df["Family_Small"] = df["Family_size"].map(lambda s: 1 if 2 <= s <= 4 else 0)
df["Family_Large"] = df["Family_size"].map(lambda s: 1 if 5 <= s else 0)
```

</div>

<div class="cell code" execution_count="13"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="vmulEFrGDI5P" outputId="fab6faf3-08f9-4a67-a318-359373fc6a10">

``` python
df.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 68 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   PassengerId     891 non-null    int64  
     1   Survived        891 non-null    int64  
     2   Pclass          891 non-null    float64
     3   Name            891 non-null    object 
     4   Sex             891 non-null    int64  
     5   Age             891 non-null    float64
     6   SibSp           891 non-null    int64  
     7   Parch           891 non-null    int64  
     8   Ticket          891 non-null    object 
     9   Fare            891 non-null    float64
     10  Cabin           891 non-null    object 
     11  Embarked        891 non-null    object 
     12  Title           891 non-null    object 
     13  Title_Master    891 non-null    int64  
     14  Title_Miss      891 non-null    int64  
     15  Title_Mr        891 non-null    int64  
     16  Title_Mrs       891 non-null    int64  
     17  Title_Officer   891 non-null    int64  
     18  Title_Royalty   891 non-null    int64  
     19  Ticket_cat      891 non-null    object 
     20  Ticket_A4       891 non-null    int64  
     21  Ticket_A5       891 non-null    int64  
     22  Ticket_AS       891 non-null    int64  
     23  Ticket_C        891 non-null    int64  
     24  Ticket_CA       891 non-null    int64  
     25  Ticket_CASOTON  891 non-null    int64  
     26  Ticket_FC       891 non-null    int64  
     27  Ticket_FCC      891 non-null    int64  
     28  Ticket_Fa       891 non-null    int64  
     29  Ticket_LINE     891 non-null    int64  
     30  Ticket_PC       891 non-null    int64  
     31  Ticket_PP       891 non-null    int64  
     32  Ticket_PPP      891 non-null    int64  
     33  Ticket_SC       891 non-null    int64  
     34  Ticket_SCA4     891 non-null    int64  
     35  Ticket_SCAH     891 non-null    int64  
     36  Ticket_SCOW     891 non-null    int64  
     37  Ticket_SCPARIS  891 non-null    int64  
     38  Ticket_SCParis  891 non-null    int64  
     39  Ticket_SOC      891 non-null    int64  
     40  Ticket_SOP      891 non-null    int64  
     41  Ticket_SOPP     891 non-null    int64  
     42  Ticket_SOTONO2  891 non-null    int64  
     43  Ticket_SOTONOQ  891 non-null    int64  
     44  Ticket_SP       891 non-null    int64  
     45  Ticket_STONO    891 non-null    int64  
     46  Ticket_STONO2   891 non-null    int64  
     47  Ticket_SWPP     891 non-null    int64  
     48  Ticket_WC       891 non-null    int64  
     49  Ticket_WEP      891 non-null    int64  
     50  Ticket_XXX      891 non-null    int64  
     51  Cabin_A         891 non-null    int64  
     52  Cabin_B         891 non-null    int64  
     53  Cabin_C         891 non-null    int64  
     54  Cabin_D         891 non-null    int64  
     55  Cabin_E         891 non-null    int64  
     56  Cabin_F         891 non-null    int64  
     57  Cabin_G         891 non-null    int64  
     58  Cabin_T         891 non-null    int64  
     59  Cabin_U         891 non-null    int64  
     60  Embarked_C      891 non-null    int64  
     61  Embarked_Q      891 non-null    int64  
     62  Embarked_S      891 non-null    int64  
     63  Embarked_U      891 non-null    int64  
     64  Family_size     891 non-null    int64  
     65  Family_Single   891 non-null    int64  
     66  Family_Small    891 non-null    int64  
     67  Family_Large    891 non-null    int64  
    dtypes: float64(3), int64(59), object(6)
    memory usage: 473.5+ KB

</div>

</div>

<div class="cell code" execution_count="14"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:1000}"
id="rr4jsOauCtAu" outputId="be5fba4f-4bf3-4a8a-97ae-cdb0927adadb">

``` python
plot_distribution(df, var="Pclass", target="Survived", row="Sex")  # 繪出Pclass與Survived的關係，Pclass 3的乘客的生存率較低
plot_distribution(df, var="Age", target="Survived", row="Sex")  # 繪出Age與Survived的關係，Age與是否生存的關係較不明顯
plot_distribution(df, var="SibSp", target="Survived", row="Sex")  # 繪出SibSp與Survived的關係，沒有與手足一起登船的乘客占大多數
plot_distribution(df, var="Parch", target="Survived", row="Sex")  # 繪出SibSp與Survived的關係，沒有與父母或子女一起登船的乘客占大多數
plot_distribution(df, var="Family_size", target="Survived", row="Sex")  # 繪出Family_size與Survived的關係，沒有與父母或子女一起登船的乘客占大多數
plot_distribution(df, var="Fare", target="Survived", row="Sex")  # 繪出Fare與Survived的關係，票價與是否生存較無關係

plt.show()
```

<div class="output display_data">

![](a8245aab52e900d1e596cf2585faf3f587e7bea7.png)

</div>

<div class="output display_data">

![](0c302897e7bc118330b0b003c76bc44ecba22d18.png)

</div>

<div class="output display_data">

![](2638799b481ea06b28d02b0fb5600c8de115f816.png)

</div>

<div class="output display_data">

![](3d6ca5d36d2786cc89042d877d2c14d75aa91f7a.png)

</div>

<div class="output display_data">

![](bf697600be9a2ee0befbad697c19d9c342d15566.png)

</div>

<div class="output display_data">

![](fc5ccfb3ad65a90955e85df14c0da4649d2a8d3c.png)

</div>

</div>

<div class="cell code" execution_count="18" id="fvQRkV8eEBU2">

``` python
# 整理Dataset內容，將沒有特別意義的或已處理過的資料去除
df.drop(columns=["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked", "Title", "Ticket_cat", "Family_size"],
    axis=1, inplace=True, errors="ignore")
```

</div>

<div class="cell code" execution_count="19" id="UcTNGzi-EEn5">

``` python
df_train = df.copy()

columns_X = set(df_train.columns) - {'Survived'}
columns_y = ['Survived']

train_X = df_train[list(columns_X)]
train_y = df_train[columns_y]
```

</div>

<div class="cell code" execution_count="20"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Q7UiY7fuEITS" outputId="a5b6afac-b4de-4482-e4a8-e20ca2f9b1c3">

``` python
log = LogisticRegression(random_state=0, max_iter=3000)
scores = cross_val_score(log, train_X, train_y.values.ravel(), cv=5, scoring='accuracy')
print(scores)
print(scores.mean())
```

<div class="output stream stdout">

    [0.81564246 0.81460674 0.81460674 0.8258427  0.85955056]
    0.8260498399347185

</div>

</div>
