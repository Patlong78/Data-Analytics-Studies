# df.info()

> Actual results from 03 - Read Dataset.ipynb

---

## Call 1: After Loading Excel (Line 27)

```python
df.info()
```

**Output:**
```
<class 'pandas.DataFrame'>
RangeIndex: 30655 entries, 0 to 30654
Data columns (total 10 columns):
 #   Column                               Non-Null Count  Dtype 
---  ------                               --------------  ----- 
 0   No.                                  30655 non-null  object
 1   Country (or Region)                  30655 non-null  str   
 2   Plant Name                           30655 non-null  str   
 3   Number of Units                      30655 non-null  object
 4   Total Plant Installed Capacity (MW)  30655 non-null  object
 5   Fuel Types                           30655 non-null  str   
 6   CO2 Emissions (Mg)                   30655 non-null  object
 7   SO2 Emissions (Mg)                   30655 non-null  object
 8   NOx Emissions (Mg)                   30655 non-null  object
 9   PM2.5 Emissions (Mg)                 30655 non-null  object
dtypes: object(7), str(3)
memory usage: 2.3+ MB
```

---

## Call 2: After Adding Fuel Types Index (Line 68-69)

```python
df.info()
```

**Output:**
```
<class 'pandas.DataFrame'>
RangeIndex: 30655 entries, 0 to 30654
Data columns (total 11 columns):
 #   Column                               Non-Null Count  Dtype 
---  ------                               --------------  ----- 
 0   No.                                  30655 non-null  object
 1   Country (or Region)                  30655 non-null  str   
 2   Plant Name                           30655 non-null  str   
 3   Number of Units                      30655 non-null  object
 4   Total Plant Installed Capacity (MW)  30655 non-null  object
 5   Fuel Types                           30655 non-null  str   
 6   CO2 Emissions (Mg)                   30655 non-null  object
 7   SO2 Emissions (Mg)                   30655 non-null  object
 8   NOx Emissions (Mg)                   30655 non-null  object
 9   PM2.5 Emissions (Mg)                 30655 non-null  object
 10  Fuel Types Index                     30655 non-null  int64 
dtypes: int64(1), object(7), str(3)
memory usage: 2.6+ MB
```

---

## Call 3: After Adding Country Index (Line 77-78)

```python
df.info()
```

**Output:**
```
<class 'pandas.DataFrame'>
RangeIndex: 30655 entries, 0 to 30654
Data columns (total 12 columns):
 #   Column                               Non-Null Count  Dtype 
---  ------                               --------------  ----- 
 0   No.                                  30655 non-null  object
 1   Country (or Region)                  30655 non-null  str   
 2   Plant Name                           30655 non-null  str   
 3   Number of Units                      30655 non-null  object
 4   Total Plant Installed Capacity (MW)  30655 non-null  object
 5   Fuel Types                           30655 non-null  str   
 6   CO2 Emissions (Mg)                   30655 non-null  object
 7   SO2 Emissions (Mg)                   30655 non-null  object
 8   NOx Emissions (Mg)                   30655 non-null  object
 9   PM2.5 Emissions (Mg)                 30655 non-null  object
 10  Fuel Types Index                     30655 non-null  int64 
 11  Country (or Region) Index            30655 non-null  int64 
dtypes: int64(2), object(7), str(3)
memory usage: 2.8+ MB
```

---

## Summary of Changes

| Stage | Columns | Rows | int64 columns | memory |
|-------|---------|------|---------------|--------|
| After load | 10 | 30,655 | 0 | 2.3+ MB |
| + Fuel Types Index | 11 | 30,655 | 1 | 2.6+ MB |
| + Country Index | 12 | 30,655 | 2 | 2.8+ MB |

---

*Source: 03 - Read Dataset.ipynb, cells #VSC-d96e0bf5, #VSC-46872c75, #VSC-26c32ba6*
*Last updated: 2026-04-29*

---

## Usage in 03 - Read Dataset.ipynb

### First Call (Line 27)

```python
df.info()
```

**Expected output after loading Excel file:**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   passengerId  891 non-null    int64
 1   survived     891 non-null    int64
 2   pclass       891 non-null    int64
 3   name         891 non-null    object
 4   sex           891 non-null    object
 5   age          714 non-null    float64
 6   sibSp        891 non-null    int64
 7   parch        891 non-null    int64
 8   ticket       891 non-null    object
 9   fare         891 non-null    float64
 10  cabin        204 non-null    object
 11  embarked     891 non-null    object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
```

### Second Call (Line 68-69) — After adding Fuel Types Index

```python
df.info()
```

### Third Call (Line 77-78) — After adding Country (or Region) Index

```python
df.info()
```

---

## Key Insights from the Notebook

| Observation | Meaning |
|-------------|---------|
| **Non-Null Count** | Rows with valid data — gaps indicate missing values |
| **Dtype** | Data type — `int64`, `float64`, `object` (string), `bool`, `datetime64` |
| **Memory Usage** | Helps identify if dataset fits in RAM |
| **Index Type** | `RangeIndex` = default integer index (0 to n-1) |

---

## Common Use Cases in This Notebook

1. **After loading data** — Verify column types and missing values
2. **After feature engineering** — Confirm new columns were added
3. **Before modeling** — Check data types are correct for sklearn

---

## Related Methods

| Method | Purpose |
|--------|---------|
| `df.describe()` | Statistical summary of numeric columns |
| `df.dtypes` | Series of column data types |
| `df.isnull().sum()` | Count missing values per column |
| `df.shape` | (rows, columns) tuple |
| `df.head()` | First 5 rows |
| `df.columns` | Column names list |

---

*Source: 03 - Read Dataset.ipynb, cells #VSC-d96e0bf5, #VSC-46872c75, #VSC-26c32ba6*
*Last updated: 2026-04-29*