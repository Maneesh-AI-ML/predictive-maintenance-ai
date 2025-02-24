{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "7c782aaa-9c33-4a59-9638-948b16659e52",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pandas is working!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "print(\"Pandas is working!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "b928325a-2d0e-4752-bbac-222abe081456",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "df = pd.read_csv(\"C:\\\\Users\\\\manee\\\\Desktop\\\\NewChapter-Python, AI & ML\\\\F GitHub Project\\\\ai4i+2020+predictive+maintenance+dataset\\\\maintenance_data_ai4i2020.csv\")  # Importing file\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "3210cfd2-3219-4dd1-af61-c248090a4c75",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset Head:\n",
      "    UDI Product ID Type  Air temperature [K]  Process temperature [K]  \\\n",
      "0    1     M14860    M                298.1                    308.6   \n",
      "1    2     L47181    L                298.2                    308.7   \n",
      "2    3     L47182    L                298.1                    308.5   \n",
      "3    4     L47183    L                298.2                    308.6   \n",
      "4    5     L47184    L                298.2                    308.7   \n",
      "\n",
      "   Rotational speed [rpm]  Torque [Nm]  Tool wear [min]  Machine failure  TWF  \\\n",
      "0                    1551         42.8                0                0    0   \n",
      "1                    1408         46.3                3                0    0   \n",
      "2                    1498         49.4                5                0    0   \n",
      "3                    1433         39.5                7                0    0   \n",
      "4                    1408         40.0                9                0    0   \n",
      "\n",
      "   HDF  PWF  OSF  RNF  \n",
      "0    0    0    0    0  \n",
      "1    0    0    0    0  \n",
      "2    0    0    0    0  \n",
      "3    0    0    0    0  \n",
      "4    0    0    0    0  \n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 10000 entries, 0 to 9999\n",
      "Data columns (total 14 columns):\n",
      " #   Column                   Non-Null Count  Dtype  \n",
      "---  ------                   --------------  -----  \n",
      " 0   UDI                      10000 non-null  int64  \n",
      " 1   Product ID               10000 non-null  object \n",
      " 2   Type                     10000 non-null  object \n",
      " 3   Air temperature [K]      10000 non-null  float64\n",
      " 4   Process temperature [K]  10000 non-null  float64\n",
      " 5   Rotational speed [rpm]   10000 non-null  int64  \n",
      " 6   Torque [Nm]              10000 non-null  float64\n",
      " 7   Tool wear [min]          10000 non-null  int64  \n",
      " 8   Machine failure          10000 non-null  int64  \n",
      " 9   TWF                      10000 non-null  int64  \n",
      " 10  HDF                      10000 non-null  int64  \n",
      " 11  PWF                      10000 non-null  int64  \n",
      " 12  OSF                      10000 non-null  int64  \n",
      " 13  RNF                      10000 non-null  int64  \n",
      "dtypes: float64(3), int64(9), object(2)\n",
      "memory usage: 1.1+ MB\n",
      "\n",
      "Dataset Summary:\n",
      " None\n",
      "\n",
      "Missing Values:\n",
      " UDI                        0\n",
      "Product ID                 0\n",
      "Type                       0\n",
      "Air temperature [K]        0\n",
      "Process temperature [K]    0\n",
      "Rotational speed [rpm]     0\n",
      "Torque [Nm]                0\n",
      "Tool wear [min]            0\n",
      "Machine failure            0\n",
      "TWF                        0\n",
      "HDF                        0\n",
      "PWF                        0\n",
      "OSF                        0\n",
      "RNF                        0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Printing some basic info\n",
    "print(\"Dataset Head:\\n\", df.head())\n",
    "print(\"\\nDataset Summary:\\n\", df.info())\n",
    "print(\"\\nMissing Values:\\n\", df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "dfdad255-59f0-4f2c-a531-ddb9a8d281f3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Column names in the dataset: Index(['UDI', 'Product ID', 'Type', 'Air temperature [K]',\n",
      "       'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',\n",
      "       'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF',\n",
      "       'RNF'],\n",
      "      dtype='object')\n",
      "\n",
      "Unique Failure Types: [0 1]\n"
     ]
    }
   ],
   "source": [
    "print(\"Column names in the dataset:\", df.columns)\n",
    "# Getting Machine failure types\n",
    "print(\"\\nUnique Failure Types:\", df['Machine failure'].unique())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "4cc12b6e-b2da-4df2-9230-43a7978ffe7a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Value counts for 'Machine Failure':\n",
      " Machine failure\n",
      "0    9661\n",
      "1     339\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(\"\\nValue counts for 'Machine Failure':\\n\", df['Machine failure'].value_counts())\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "7681441b-637d-42e9-a91e-218fe1f86a6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#9,661 non-failures (0s) and only 339 failures (1s)- only ~3.4% failures\n",
    "#Data set is highly imbalanced\n",
    "#Accuracy will look high, but the model won’t actually be useful\n",
    "# Using data balancing techniques: Oversampling to duplicate failure examples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "ac3dd0ba-b138-4985-819d-9cc6341428c7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "New class distribution: Counter({0: 9661, 1: 9661})\n"
     ]
    }
   ],
   "source": [
    "from imblearn.over_sampling import RandomOverSampler\n",
    "from collections import Counter\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\", category=FutureWarning)\n",
    "\n",
    "# Select features (X) and target (y)\n",
    "X = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]\n",
    "y = df['Machine failure']  \n",
    "\n",
    "# Applying Oversampling\n",
    "oversample = RandomOverSampler(sampling_strategy=1.0, random_state=42)\n",
    "X_resampled, y_resampled = oversample.fit_resample(X, y)\n",
    "\n",
    "# Checking new class distribution\n",
    "print(\"New class distribution:\", Counter(y_resampled))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "fe15cdd9-7421-4675-a8b6-387a78e62242",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Accuracy: 1.00\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.99      1.00      1934\n",
      "           1       0.99      1.00      1.00      1931\n",
      "\n",
      "    accuracy                           1.00      3865\n",
      "   macro avg       1.00      1.00      1.00      3865\n",
      "weighted avg       1.00      1.00      1.00      3865\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Splitting data into train/test\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)\n",
    "\n",
    "# Training a RandomForest model\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Prediction and evaluation\n",
    "y_pred = model.predict(X_test)\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "print(f\"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}\")\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "3211256d-fb02-4333-a854-4a3b9eb4b495",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Model is too perfect, not a real world scenario\n",
    "#Reducing overfitting and preventing memorization\n",
    "model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "d314eaf1-37cc-429a-8a18-5d615cad332c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of duplicates between train & test sets: 44000\n"
     ]
    }
   ],
   "source": [
    "# Checking for duplicate rows in test set\n",
    "duplicates = X_test.merge(X_train, how='inner')\n",
    "print(f\"Number of duplicates between train & test sets: {len(duplicates)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "0def66ee-c22f-4359-b802-e36aa48e6e20",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correcting Order: First Split Data, Then Apply Oversampling on Train Set Only\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Apply oversampling ONLY to the training data (not the test set)\n",
    "oversample = RandomOverSampler(sampling_strategy=1.0, random_state=42)\n",
    "X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "3b8cd49f-6a8e-481e-b031-5d89157a429f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of duplicates between train & test sets: 0\n"
     ]
    }
   ],
   "source": [
    "duplicates = X_test.merge(X_train_resampled, how='inner')\n",
    "print(f\"Number of duplicates between train & test sets: {len(duplicates)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "7d496831-42df-4ca2-84ed-4af5c6329faa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Accuracy: 0.91\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.91      0.95      1939\n",
      "           1       0.24      0.90      0.38        61\n",
      "\n",
      "    accuracy                           0.91      2000\n",
      "   macro avg       0.62      0.91      0.67      2000\n",
      "weighted avg       0.97      0.91      0.93      2000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)\n",
    "model.fit(X_train_resampled, y_train_resampled)\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "print(f\"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}\")\n",
    "print(classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "828506d3-bdd0-4540-a2d1-15dbd34b618a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f49a1e1-655f-4515-a66b-dad9368b1110",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
