# Hướng dẫn hoàn thành Kaggle House Prices (như Data Scientist)

Checklist từng bước để sửa notebook hiện tại, xây pipeline sạch, và nộp bài Kaggle.

**Mục tiêu cuối:** có `submission.csv` (cột `Id`, `SalePrice`) từ model đã validate bằng **RMSLE**, tái lập được kết quả.

**Dữ liệu sẵn có trong repo**

- `train.csv` — 1460 mẫu, có `SalePrice`
- `test.csv` — 1459 mẫu, **không** có `SalePrice`
- `data_description.txt` — dictionary cột
- `Final_Project_HousePricing.ipynb` — notebook cần chỉnh / viết lại phần modeling

**Metric Kaggle:** Root Mean Squared Logarithmic Error (RMSLE) — càng thấp càng tốt.

---

## Cách dùng checklist

- Đánh `[x]` khi xong từng mục.
- Không nhảy cóc: phần sau phụ thuộc phần trước.
- Mỗi phần kết thúc bằng **Definition of Done** — chỉ sang bước tiếp theo khi đủ.

---

## Phase 0 — Setup & hiểu bài toán

### 0.1 Môi trường

- [x] Tạo virtualenv / conda riêng cho project
- [x] Cài: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `jupyter`
- [ ] (Khuyến nghị) thêm: `xgboost` hoặc `lightgbm`
- [x] Tạo `requirements.txt` và pin version đã chạy ổn
- [ ] Chạy notebook bằng kernel của env này (không lẫn Colab/local lung tung)

### 0.2 Hiểu competition

- [ ] Đọc [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) — Evaluation = RMSLE
- [ ] Đọc `data_description.txt` (đặc biệt cột NA mang nghĩa “không có”: Pool, Garage, Basement, Alley, Fence…)
- [ ] Xác nhận: **train để học**, **test chỉ để predict + submit** (không dùng để tune)

### 0.3 Nguyên tắc làm việc

- [ ] Không drop `Id` khỏi test trước khi tạo submission
- [ ] Mọi imputation / encoding **fit trên train**, **transform trên test** (tránh data leakage)
- [ ] Target transform (`log1p`) thì predict xong phải `expm1` trước khi submit
- [ ] Đo bằng RMSLE (hoặc RMSE trên `log1p(SalePrice)`), không chỉ R²

**Definition of Done — Phase 0:** môi trường chạy được, hiểu metric, nắm quy tắc leakage.

---

## Phase 1 — Khởi tạo notebook sạch (sửa lỗi nền)

Notebook cũ có nhiều cell gãy. Nên tổ chức lại theo section rõ ràng.

### 1.1 Cấu trúc notebook đề xuất

- [ ] `0. Setup & imports`
- [ ] `1. Load data`
- [ ] `2. EDA`
- [ ] `3. Missing values & cleaning`
- [ ] `4. Feature engineering`
- [ ] `5. Encoding & scaling`
- [ ] `6. Model training & validation`
- [ ] `7. Predict test + submission`
- [ ] `8. (Optional) Error analysis & next experiments`

### 1.2 Load data đúng (thay cell wget lỗi)

- [ ] Xóa / bỏ `wget` URL GitHub `/blob/` (đó là HTML, không phải CSV)
- [ ] Load local:

```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

- [ ] Lưu `test_ids = test["Id"].copy()` trước khi xử lý
- [ ] Kiểm tra shape: train `(1460, 81)`, test `(1459, 80)`
- [ ] Tách `y = train["SalePrice"].copy()`; không để `SalePrice` lẫn trong feature matrix của test

### 1.3 Sửa các lỗi runtime đã biết (nếu giữ code cũ)

- [ ] Thay `df.dtypes.iteritems()` → `df.dtypes.items()`
- [ ] Thay `np.bool` → `bool` hoặc `np.bool_`
- [ ] Thay `sns.distplot` → `sns.histplot(..., kde=True)` (hoặc `sns.kdeplot`)
- [ ] Không `categorical_features.remove("MasVnrArea")` nếu cột không thuộc object
- [ ] Không dùng biến chưa định nghĩa: `train_sample`, `TotalPoints`, `train` (khi mới chỉ có `df`)

**Definition of Done — Phase 1:** load train/test local thành công, có `test_ids`, notebook chạy được từ đầu đến hết load.

---

## Phase 2 — EDA (Explore như DS, không chỉ vẽ cho đẹp)

### 2.1 Tổng quan dữ liệu

- [ ] `train.info()`, `describe()`, đếm dtype object vs numeric
- [ ] Missing rate theo cột (train và test so sánh cạnh nhau)
- [ ] Phân phối `SalePrice` (hist + QQ): skew / kurtosis
- [ ] Quyết định: dùng `y_log = np.log1p(y)` cho modeling

### 2.2 Target & outliers

- [ ] Scatter `GrLivArea` vs `SalePrice` — ghi nhận outlier diện tích lớn / giá thấp
- [ ] Quyết định có drop outlier trên **train only** (ví dụ `GrLivArea > 4000` và giá thấp) và **ghi rõ lý do**
- [ ] Không “xóa outlier” trên test

### 2.3 Quan hệ feature–target

- [ ] Corr numeric với `SalePrice` / `y_log` (dùng full data hoặc CV-aware; tránh sample 25% nếu không cần)
- [ ] Boxplot ordinal quan trọng: `OverallQual`, `ExterQual`, `KitchenQual`, `BsmtQual`…
- [ ] Categorical: neighborhood, zoning — median price theo nhóm
- [ ] Phân biệt NA mang nghĩa “không có tiện ích” vs missing thật

### 2.4 Ghi chú quyết định (bắt buộc)

Trong markdown notebook, ghi ngắn:

- [ ] Cột nào impute bằng gì (None / 0 / median / mode)
- [ ] Cột nào ordinal encode vs one-hot
- [ ] Feature mới dự kiến (TotalSF, TotalBath, Age, RemodAge…)
- [ ] Cột gần như hằng / quá nhiều NA → cân nhắc bỏ

**Definition of Done — Phase 2:** có bảng missing + vài chart có insight + danh sách quyết định preprocessing viết ra giấy/markdown.

---

## Phase 3 — Preprocessing (train/test cùng một pipeline)

> Nguyên tắc vàng: viết hàm `fit` trên train, `transform` train & test giống nhau.

### 3.1 Missing values — sửa bug fillna cũ

Checklist sửa các lỗi đã phát hiện:

- [ ] **Không** `fillna(series.mode())` trực tiếp → dùng `mode()[0]` hoặc `fillna(value)`
- [ ] **Không** `fillna(df[col] == 0)` → dùng số `0` hoặc literal đúng nghĩa
- [ ] Cột “không có X” (theo data_description): fill `"None"` (hoặc một token thống nhất)
  - Ví dụ: `Alley`, `Bsmt*`, `FireplaceQu`, `Garage*`, `PoolQC`, `Fence`, `MiscFeature`
- [ ] Numeric liên quan: `MasVnrArea→0`, `GarageYrBlt→0` hoặc median (chọn 1 chiến lược và giữ nhất quán)
- [ ] `LotFrontage`: median theo `Neighborhood` (tốt hơn mean toàn cục)
- [ ] `Electrical`: `mode()[0]` (đừng hard-code nếu đã tính mode)
- [ ] Sau fill: `train.isna().sum().sum() == 0` và `test.isna().sum().sum() == 0` (hoặc chỉ còn cột cố ý để sau)

### 3.2 Ordinal mapping — khớp key với giá trị đã fill

- [ ] Nếu fill `"None"` thì dict map phải có key `"None"` (không để key `"NA"` trong khi data là `"No Garage"`)
- [ ] Kiểm tra sau map: `isna().sum()` của các cột ordinal = 0
- [ ] Ordinal gợi ý: Qual/Cond/QC, `BsmtExposure`, `BsmtFinType*`, `GarageFinish`, `Functional`, `Fence`, `LandSlope`…

### 3.3 Feature engineering (tối thiểu nên có)

- [ ] `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
- [ ] `TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
- [ ] `Age = YrSold - YearBuilt` (hoặc year reference nhất quán)
- [ ] `RemodAge = YrSold - YearRemodAdd`
- [ ] `HasGarage`, `HasBsmt`, `HasPool` (binary) nếu hữu ích
- [ ] Kiểm tra không tạo số âm vô lý / inf

### 3.4 Encoding & scale

- [ ] Nominal (không thứ tự): `OneHotEncoder(handle_unknown="ignore")` hoặc `pd.get_dummies` **align cột train/test**
- [ ] Ordinal: map số hoặc `OrdinalEncoder`
- [ ] Scale: `RobustScaler` / `StandardScaler` — chủ yếu hữu ích cho Linear/Ridge/Lasso/ElasticNet; tree model thường không bắt buộc
- [ ] Đảm bảo train và test cùng số cột, cùng thứ tự tên cột

### 3.5 (Khuyến nghị mạnh) Dùng `sklearn` Pipeline / ColumnTransformer

- [ ] Gói preprocessing vào 1 object `preprocess`
- [ ] `preprocess.fit(X_train)` rồi `transform` validation/test
- [ ] Không copy-paste logic fillna rời rạc dễ lệch train vs test

**Definition of Done — Phase 3:** từ raw CSV → `X_train`, `X_test_kaggle`, `y_log` sạch; không NaN; cột train/test khớp; không leakage rõ ràng.

---

## Phase 4 — Validation đúng cách

### 4.1 Split / CV

- [ ] Dùng `KFold` hoặc `train_test_split` trên **train only** (ví dụ 5-fold)
- [ ] Không dùng hàng `SalePrice == 0` để giả làm test Kaggle
- [ ] Mỗi fold: fit preprocess trên fold-train, transform fold-val

### 4.2 Metric

- [ ] Implement RMSLE:

```python
from sklearn.metrics import mean_squared_error
import numpy as np

def rmsle(y_true, y_pred):
    # y_true, y_pred ở thang giá gốc
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
```

- [ ] Nếu train trên `y_log`: predict → `np.expm1(pred_log)` rồi tính RMSLE, **hoặc** RMSE trực tiếp trên log (tương đương)

### 4.3 Baseline trước model phức tạp

- [ ] Baseline 1: dự đoán = median `SalePrice` (để biết “sàn”)
- [ ] Baseline 2: `LinearRegression` / `Ridge` trên vài feature mạnh (`OverallQual`, `GrLivArea`, `TotalSF`…)
- [ ] Ghi CV score baseline vào bảng kết quả

**Definition of Done — Phase 4:** có CV RMSLE baseline + hàm metric tái sử dụng được.

---

## Phase 5 — Modeling (tăng dần độ phức tạp)

### 5.1 Model tối thiểu nên chạy

- [ ] Ridge / Lasso / ElasticNet (kèm tuning `alpha`)
- [ ] RandomForestRegressor
- [ ] GradientBoostingRegressor **hoặc** XGBoost / LightGBM

### 5.2 So sánh công bằng

- [ ] Cùng preprocessing, cùng CV folds
- [ ] Bảng kết quả:

| Model           | CV RMSLE (mean ± std) | Notes |
| --------------- | --------------------- | ----- |
| Median baseline |                       |       |
| Ridge           |                       |       |
| RF              |                       |       |
| GBM/XGB         |                       |       |

- [ ] Chọn model (hoặc blend) theo CV, **không** theo 1 lần `score()` R² trên holdout nhỏ

### 5.3 Hyperparameter (vừa đủ)

- [ ] `GridSearchCV` / `RandomizedSearchCV` trên CV
- [ ] Không over-tune trên toàn bộ train rồi tự khen trên cùng data
- [ ] Tree: kiểm soát `max_depth`, `n_estimators`, `learning_rate`, `min_samples_leaf`

### 5.4 (Optional) Ensemble

- [ ] Average / weighted average dự đoán của 2–3 model tốt nhất
- [ ] Chỉ ensemble nếu CV thực sự cải thiện

**Definition of Done — Phase 5:** có ≥2 model, bảng CV RMSLE, chọn 1 hướng submit chính.

---

## Phase 6 — Predict Kaggle test & submission

### 6.1 Train lại trên full train

- [ ] Fit preprocess + model trên **toàn bộ** `train` (sau khi đã chọn config bằng CV)
- [ ] Predict `test` → `pred_log` → `SalePrice = np.expm1(pred_log)`
- [ ] Kiểm tra pred: không âm, không NaN, không Inf; phân phối hợp lý so với train

### 6.2 Xuất file

- [ ] Tạo submission:

```python
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": pred_prices
})
submission.to_csv("submission.csv", index=False)
```

- [ ] Kiểm tra: đúng 1459 dòng, đúng 2 cột, `Id` khớp `test.csv`
- [ ] Nộp lên Kaggle (hoặc lưu để nộp sau)

### 6.3 Nhật ký thí nghiệm

- [ ] Ghi: ngày, model, CV RMSLE, public LB (nếu có), thay đổi so với lần trước
- [ ] Không submit spam; mỗi thay đổi lớn 1–2 submission có kiểm soát

**Definition of Done — Phase 6:** có `submission.csv` hợp lệ + điểm CV gắn với file đó.

---

## Phase 7 — Làm việc như Data Scientist (chất lượng deliverable)

### 7.1 Notebook “đọc được”

- [ ] Markdown giải thích **vì sao** làm từng bước (không chỉ “vẽ chart”)
- [ ] Xóa cell chết / code comment dài không chạy / title `"aaa"`
- [ ] Mỗi chart có 1 câu insight
- [ ] Section Tree-Based / model phụ không để heading trống

### 7.2 Reproducibility

- [ ] `random_state` cố định
- [ ] `requirements.txt`
- [ ] README: cách chạy từ zero → ra `submission.csv`
- [ ] Không phụ thuộc file tải tay từ URL hỏng

### 7.3 Repo hygiene

- [ ] README mô tả bài toán, metric, cách chạy, kết quả CV tốt nhất
- [ ] Không commit file rác / output wget HTML
- [ ] (Nếu cần) tách `src/preprocess.py`, `src/train.py` khi notebook quá dài

**Definition of Done — Phase 7:** người khác clone repo, làm theo README, ra được submission tương tự.

---

## Thứ tự sửa lỗi cũ (ưu tiên nếu refactor notebook hiện tại)

Làm lần lượt:

1. [ ] Load `train.csv` / `test.csv` local; bỏ wget sai
2. [ ] Giữ `Id` test; tách `y`
3. [ ] Sửa fillna (`mode()[0]`, `0`, `"None"`)
4. [ ] Sửa ordinal map cho khớp token fill
5. [ ] Bỏ logic `SalePrice == 0` làm test giả
6. [ ] Pipeline encode align train/test
7. [ ] CV + RMSLE
8. [ ] Train model → `expm1` → `submission.csv`
9. [ ] Dọn EDA / markdown / README

---

## Definition of Done — toàn project

Project được xem là **hoàn thành mức Data Scientist** khi:

- [ ] Notebook chạy top → bottom không lỗi
- [ ] Preprocess không leakage (fit train / transform test)
- [ ] Có CV RMSLE rõ ràng cho model cuối
- [ ] Có `submission.csv` đúng format Kaggle
- [ ] README + requirements đủ để tái lập
- [ ] Biết giải thích: feature nào quan trọng, giới hạn model, hướng cải thiện tiếp

---

## Gợi ý timeline (nếu làm trong vài buổi)

| Buổi | Việc                                            |
| ---- | ----------------------------------------------- |
| 1    | Phase 0–2: setup, EDA, quyết định preprocessing |
| 2    | Phase 3–4: pipeline sạch + baseline CV          |
| 3    | Phase 5–6: model tốt hơn + submission           |
| 4    | Phase 7: dọn notebook, README, nhật ký kết quả  |

---

## Tài liệu tham chiếu nhanh trong repo

- `data_description.txt` — ý nghĩa cột & NA đặc biệt
- `train.csv` / `test.csv` — dữ liệu chính thức
- `Final_Project_HousePricing.ipynb` — bản cũ (tham khảo EDA; **đừng** copy nguyên phần modeling đang gãy)

Chúc bạn hoàn thành và có submission sạch. Khi cần, có thể nhờ AI sửa notebook theo đúng thứ tự Phase 1 → 6 trong file này.
