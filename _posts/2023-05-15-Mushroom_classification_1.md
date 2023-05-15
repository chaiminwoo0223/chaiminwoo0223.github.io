---
layout: single
title:  "jupyter notebook 변환하기!"
categories: coding
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# Configure the Environment



```python
%matplotlib inline
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings; warnings.filterwarnings(action='once')
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import keras
```

<pre>
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
</pre>
# Read / Explore the Data



```python
data_path = '/content/drive/MyDrive/3학년1학기/명예학회/Mushrooms/mushrooms.csv'
df = pd.read_csv(data_path)
df.head()
```

<pre>
/usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
  and should_run_async(code)
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
</pre>
<pre>
  class cap-shape cap-surface cap-color bruises odor gill-attachment  \
0     p         x           s         n       t    p               f   
1     e         x           s         y       t    a               f   
2     e         b           s         w       t    l               f   
3     p         x           y         w       t    p               f   
4     e         x           s         g       f    n               f   

  gill-spacing gill-size gill-color  ... stalk-surface-below-ring  \
0            c         n          k  ...                        s   
1            c         b          k  ...                        s   
2            c         b          n  ...                        s   
3            c         n          n  ...                        s   
4            w         b          k  ...                        s   

  stalk-color-above-ring stalk-color-below-ring veil-type veil-color  \
0                      w                      w         p          w   
1                      w                      w         p          w   
2                      w                      w         p          w   
3                      w                      w         p          w   
4                      w                      w         p          w   

  ring-number ring-type spore-print-color population habitat  
0           o         p                 k          s       u  
1           o         p                 n          n       g  
2           o         p                 n          n       m  
3           o         p                 k          s       u  
4           o         e                 n          a       g  

[5 rows x 23 columns]
</pre>

```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8124 entries, 0 to 8123
Data columns (total 23 columns):
 #   Column                    Non-Null Count  Dtype 
---  ------                    --------------  ----- 
 0   class                     8124 non-null   object
 1   cap-shape                 8124 non-null   object
 2   cap-surface               8124 non-null   object
 3   cap-color                 8124 non-null   object
 4   bruises                   8124 non-null   object
 5   odor                      8124 non-null   object
 6   gill-attachment           8124 non-null   object
 7   gill-spacing              8124 non-null   object
 8   gill-size                 8124 non-null   object
 9   gill-color                8124 non-null   object
 10  stalk-shape               8124 non-null   object
 11  stalk-root                8124 non-null   object
 12  stalk-surface-above-ring  8124 non-null   object
 13  stalk-surface-below-ring  8124 non-null   object
 14  stalk-color-above-ring    8124 non-null   object
 15  stalk-color-below-ring    8124 non-null   object
 16  veil-type                 8124 non-null   object
 17  veil-color                8124 non-null   object
 18  ring-number               8124 non-null   object
 19  ring-type                 8124 non-null   object
 20  spore-print-color         8124 non-null   object
 21  population                8124 non-null   object
 22  habitat                   8124 non-null   object
dtypes: object(23)
memory usage: 1.4+ MB
</pre>
<pre>
/usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
  and should_run_async(code)
</pre>

```python
df.describe()
```

<pre>
/usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
  and should_run_async(code)
</pre>
<pre>
       class cap-shape cap-surface cap-color bruises  odor gill-attachment  \
count   8124      8124        8124      8124    8124  8124            8124   
unique     2         6           4        10       2     9               2   
top        e         x           y         n       f     n               f   
freq    4208      3656        3244      2284    4748  3528            7914   

       gill-spacing gill-size gill-color  ... stalk-surface-below-ring  \
count          8124      8124       8124  ...                     8124   
unique            2         2         12  ...                        4   
top               c         b          b  ...                        s   
freq           6812      5612       1728  ...                     4936   

       stalk-color-above-ring stalk-color-below-ring veil-type veil-color  \
count                    8124                   8124      8124       8124   
unique                      9                      9         1          4   
top                         w                      w         p          w   
freq                     4464                   4384      8124       7924   

       ring-number ring-type spore-print-color population habitat  
count         8124      8124              8124       8124    8124  
unique           3         5                 9          6       7  
top              o         p                 w          v       d  
freq          7488      3968              2388       4040    3148  

[4 rows x 23 columns]
</pre>

```python
columns = df.columns
for col in columns:
    print('{feat_name}:{feat_values}'.format(feat_name=col, feat_values=df[col].unique()))
```

<pre>
class:['p' 'e']
cap-shape:['x' 'b' 's' 'f' 'k' 'c']
cap-surface:['s' 'y' 'f' 'g']
cap-color:['n' 'y' 'w' 'g' 'e' 'p' 'b' 'u' 'c' 'r']
bruises:['t' 'f']
odor:['p' 'a' 'l' 'n' 'f' 'c' 'y' 's' 'm']
gill-attachment:['f' 'a']
gill-spacing:['c' 'w']
gill-size:['n' 'b']
gill-color:['k' 'n' 'g' 'p' 'w' 'h' 'u' 'e' 'b' 'r' 'y' 'o']
stalk-shape:['e' 't']
stalk-root:['e' 'c' 'b' 'r' '?']
stalk-surface-above-ring:['s' 'f' 'k' 'y']
stalk-surface-below-ring:['s' 'f' 'y' 'k']
stalk-color-above-ring:['w' 'g' 'p' 'n' 'b' 'e' 'o' 'c' 'y']
stalk-color-below-ring:['w' 'p' 'g' 'b' 'n' 'e' 'y' 'o' 'c']
veil-type:['p']
veil-color:['w' 'n' 'o' 'y']
ring-number:['o' 't' 'n']
ring-type:['p' 'e' 'l' 'f' 'n']
spore-print-color:['k' 'n' 'u' 'h' 'w' 'r' 'o' 'y' 'b']
population:['s' 'n' 'a' 'v' 'y' 'c']
habitat:['u' 'g' 'm' 'd' 'p' 'w' 'l']
</pre>
<pre>
/usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
  and should_run_async(code)
</pre>

```python
df.shape
```

<pre>
(8124, 23)
</pre>

```python
df["class"].value_counts()
```

<pre>
e    4208
p    3916
Name: class, dtype: int64
</pre>

```python
df["class"].unique()
```

<pre>
/usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
  and should_run_async(code)
</pre>
<pre>
array(['p', 'e'], dtype=object)
</pre>

```python
sns.countplot(x='class', data=df)
```

<pre>
<Axes: xlabel='class', ylabel='count'>
</pre>
<pre>
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
</pre>
<pre>
<Figure size 640x480 with 1 Axes>
</pre>

```python
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['class'])
print(y)
```

<pre>
[1 0 0 ... 0 1 0]
</pre>
<pre>
/usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
  and should_run_async(code)
</pre>

```python
X = df.drop('class', axis=1)
columns = X.columns

for i in range(len(X.columns)):
    le = preprocessing.LabelEncoder()
    X[columns[i]] = le.fit_transform(X[columns[i]])

X.head()
```

<pre>
/usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
  and should_run_async(code)
</pre>
<pre>
   cap-shape  cap-surface  cap-color  bruises  odor  gill-attachment  \
0          5            2          4        1     6                1   
1          5            2          9        1     0                1   
2          0            2          8        1     3                1   
3          5            3          8        1     6                1   
4          5            2          3        0     5                1   

   gill-spacing  gill-size  gill-color  stalk-shape  ...  \
0             0          1           4            0  ...   
1             0          0           4            0  ...   
2             0          0           5            0  ...   
3             0          1           5            0  ...   
4             1          0           4            1  ...   

   stalk-surface-below-ring  stalk-color-above-ring  stalk-color-below-ring  \
0                         2                       7                       7   
1                         2                       7                       7   
2                         2                       7                       7   
3                         2                       7                       7   
4                         2                       7                       7   

   veil-type  veil-color  ring-number  ring-type  spore-print-color  \
0          0           2            1          4                  2   
1          0           2            1          4                  3   
2          0           2            1          4                  3   
3          0           2            1          4                  2   
4          0           2            1          0                  3   

   population  habitat  
0           3        5  
1           2        1  
2           2        3  
3           3        5  
4           0        1  

[5 rows x 22 columns]
</pre>

```python
for col in columns:
    print('{}:{}'.format(col, X[col].nunique()))
```

<pre>
cap-shape:6
cap-surface:4
cap-color:10
bruises:2
odor:9
gill-attachment:2
gill-spacing:2
gill-size:2
gill-color:12
stalk-shape:2
stalk-root:5
stalk-surface-above-ring:4
stalk-surface-below-ring:4
stalk-color-above-ring:9
stalk-color-below-ring:9
veil-type:1
veil-color:4
ring-number:3
ring-type:5
spore-print-color:9
population:6
habitat:7
</pre>
<pre>
/usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
  and should_run_async(code)
</pre>

```python
sequences = []
columns = X.columns

for idx, row in X.iterrows():
    sequence = []
    for i in range(len(columns)):
        sequence.append(row[columns[i]])
    sequences.append(sequence)

print('{sequence} : {label}'.format(sequence=sequences[0], label=y[0]))
print('len of sequences :', len(sequences[0]))
```

<pre>
[5, 2, 4, 1, 6, 1, 0, 1, 4, 0, 3, 2, 2, 7, 7, 0, 2, 1, 4, 2, 3, 5] : 1
len of sequences : 22
</pre>

```python
RANDOM_SEED = 42
x_train, x_test, y_train, y_test = train_test_split(sequences, y, test_size=0.1, random_state=RANDOM_SEED)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_train = np.array(y_train)

print(x_train, y_train)
```

<pre>
[[2 0 9 ... 1 4 0]
 [5 0 5 ... 2 3 0]
 [2 0 2 ... 2 4 0]
 ...
 [2 3 4 ... 3 5 4]
 [3 2 2 ... 7 4 4]
 [3 0 3 ... 7 2 1]] [1 1 0 ... 0 1 0]
</pre>
# Build and Train the Model



```python
def build_model():
    embeddings_dims = 300
    max_seq_length = len(sequences[0])
    max_features = 12
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    using_pretrained_emb = False 

    model = Sequential()

    if using_pretrained_emb:
        model.add(layers.Embedding(max_features, embeddings_dims, input_length=max_seq_length, trainable=False))
    else:
        model.add(layers.Embedding(max_features, embeddings_dims, input_length=max_seq_length))

    model.add(layers.Dropout(0.5))
    model.add(layers.Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(hidden_dims))
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return model
```


```python
model = build_model()
model.summary()
```

<pre>
<frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
<frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()
</pre>
<pre>
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 22, 300)           3600      
                                                                 
 dropout (Dropout)           (None, 22, 300)           0         
                                                                 
 conv1d (Conv1D)             (None, 20, 250)           225250    
                                                                 
 max_pooling1d (MaxPooling1D  (None, 10, 250)          0         
 )                                                               
                                                                 
 conv1d_1 (Conv1D)           (None, 8, 250)            187750    
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 4, 250)           0         
 1D)                                                             
                                                                 
 conv1d_2 (Conv1D)           (None, 2, 250)            187750    
                                                                 
 global_max_pooling1d (Globa  (None, 250)              0         
 lMaxPooling1D)                                                  
                                                                 
 dense (Dense)               (None, 250)               62750     
                                                                 
 dropout_1 (Dropout)         (None, 250)               0         
                                                                 
 activation (Activation)     (None, 250)               0         
                                                                 
 dense_1 (Dense)             (None, 1)                 251       
                                                                 
 activation_1 (Activation)   (None, 1)                 0         
                                                                 
=================================================================
Total params: 667,351
Trainable params: 667,351
Non-trainable params: 0
_________________________________________________________________
</pre>

```python
# Train
history = model.fit(x_train, y_train, epochs=10, verbose=True, validation_data=(x_test, y_test), batch_size=16)
```

<pre>
Epoch 1/10
457/457 [==============================] - 22s 10ms/step - loss: 0.6921 - accuracy: 0.5345 - val_loss: 0.6907 - val_accuracy: 0.6224
Epoch 2/10
457/457 [==============================] - 3s 6ms/step - loss: 0.6908 - accuracy: 0.5484 - val_loss: 0.6892 - val_accuracy: 0.6691
Epoch 3/10
457/457 [==============================] - 3s 6ms/step - loss: 0.6897 - accuracy: 0.5685 - val_loss: 0.6877 - val_accuracy: 0.6962
Epoch 4/10
457/457 [==============================] - 3s 7ms/step - loss: 0.6886 - accuracy: 0.5799 - val_loss: 0.6862 - val_accuracy: 0.7515
Epoch 5/10
457/457 [==============================] - 4s 8ms/step - loss: 0.6876 - accuracy: 0.5961 - val_loss: 0.6846 - val_accuracy: 0.8290
Epoch 6/10
457/457 [==============================] - 3s 6ms/step - loss: 0.6865 - accuracy: 0.6146 - val_loss: 0.6829 - val_accuracy: 0.8733
Epoch 7/10
457/457 [==============================] - 3s 6ms/step - loss: 0.6844 - accuracy: 0.6423 - val_loss: 0.6811 - val_accuracy: 0.8782
Epoch 8/10
457/457 [==============================] - 3s 6ms/step - loss: 0.6832 - accuracy: 0.6570 - val_loss: 0.6791 - val_accuracy: 0.8844
Epoch 9/10
457/457 [==============================] - 3s 8ms/step - loss: 0.6815 - accuracy: 0.6768 - val_loss: 0.6769 - val_accuracy: 0.8868
Epoch 10/10
457/457 [==============================] - 3s 7ms/step - loss: 0.6803 - accuracy: 0.6846 - val_loss: 0.6745 - val_accuracy: 0.8893
</pre>
# Metrics



```python
# 예측하기
predictions = model.predict(x_test)
# 예측된 클래스 인덱스 가져오기
predicted_classes = np.argmax(predictions, axis=1)

cnn_metrics = {'acc': metrics.accuracy_score(y_test, predicted_classes)}
cnn_metrics['prec'] = metrics.precision_score(y_test, predicted_classes)
cnn_metrics['rec'] = metrics.recall_score(y_test, predicted_classes)
cnn_metrics['f1'] = metrics.f1_score(y_test, predicted_classes)
cnn_metrics['f1_macro'] = metrics.f1_score(y_test, predicted_classes, average='macro')
cnn_metrics['auc'] = metrics.roc_auc_score(y_test, predicted_classes)

for metric in cnn_metrics:
  print('{metric_name}:{metric_value}'.format(metric_name=metric, metric_value=cnn_metrics[metric]))

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Get training and test accuracy history.
training_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

# Visualize acc history
plt.plot(epoch_count, training_acc, 'r--')
plt.plot(epoch_count, test_acc, 'b-')
plt.legend(['Training Acc', 'Test Acc'])
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.show();
```

# Visualize Activations



```python
from keras.models import Model

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(x_train[1].reshape(1, 22))

for layer_num, act in enumerate(activations):
    if len(act.shape) > 2:
        plt.rcParams["axes.grid"] = False
        plt.matshow(act[0, :, :], cmap='viridis')
    else:
        plt.figure(figsize = (16,1))
        sns.heatmap(act, cbar=False, cmap='viridis')
```


```python
```
