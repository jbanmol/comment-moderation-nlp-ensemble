# Historical experiment retained for inspection; known validation leakage.
# Use evaluate.py for the supported leakage-safe evaluation.
if __name__ == "__main__":
    raise SystemExit("Historical evaluation is disabled: use evaluate.py; see EVALUATION.md")

# %% [markdown]
# # Comment Category Prediction — v5 (Pushing Past 0.90)
# 
# **Key improvements over v3:**
# 1. Sentence-transformer embeddings (384-d) for semantic text understanding
# 2. Multi-model ensemble: LightGBM + XGBoost + CatBoost
# 3. Per-class threshold optimization for minority class (Label 3)
# 4. Enhanced feature engineering with interaction terms & post-level aggregations

# %% Setup & Data Loading
# 🔴 Set to True for a quick 30-second run to verify no code/syntax errors exist
SMOKE_TEST = False

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import lightgbm as lgb
from scipy.sparse import hstack
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

try:
    import subprocess
    subprocess.run(['nvidia-smi'], capture_output=True)
    USE_GPU = True
    print('✅ GPU available!')
except:
    USE_GPU = False
    print('⚠️ CPU mode')

# Try importing optional packages
try:
    import xgboost as xgb
    HAS_XGB = True
    print('✅ XGBoost available')
except ImportError:
    HAS_XGB = False
    print('⚠️ XGBoost not available')

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
    print('✅ CatBoost available')
except ImportError:
    HAS_CB = False
    print('⚠️ CatBoost not available')

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
    print('✅ sentence-transformers available')
except ImportError:
    HAS_ST = False
    print('⚠️ sentence-transformers not available — will use TF-IDF only')

# %% Load Data
train  = pd.read_csv('/kaggle/input/comment-category-prediction-challenge/train.csv')
test   = pd.read_csv('/kaggle/input/comment-category-prediction-challenge/test.csv')

if SMOKE_TEST:
    print('\n🚨 RUNNING IN SMOKE TEST MODE (Subset Data) 🚨')
    train = train.sample(500, random_state=42).reset_index(drop=True)
    test = test.head(100).copy()

sample = pd.read_csv('/kaggle/input/comment-category-prediction-challenge/Sample.csv')
print(f'Train: {train.shape} | Test: {test.shape}')
print(train['label'].value_counts(normalize=True).sort_index())

# %% [markdown]
# ## Feature Engineering (Enhanced v4)

# %% Feature Engineering
def engineer_features(df, is_train=True, post_id_stats=None, label_enc=None,
                      post_agg_stats=None):
    df = df.copy()
    df['comment'] = df['comment'].fillna('')

    # ═══════════════════════════════════════════════════════════════
    # GROUP A: if_2 & if_1 — THE MOST POWERFUL FEATURES
    # ═══════════════════════════════════════════════════════════════
    df['if2_is_4']  = (df['if_2'] == 4).astype(int)
    df['if2_is_10'] = (df['if_2'] == 10).astype(int)
    df['if2_is_5']  = (df['if_2'] == 5).astype(int)
    df['if2_is_6']  = (df['if_2'] == 6).astype(int)
    df['log_if2']   = np.log1p(df['if_2'])
    df['if2_gt_10'] = (df['if_2'] > 10).astype(int)

    df['if1_is_0']    = (df['if_1'] == 0).astype(int)
    df['if1_is_4']    = (df['if_1'] == 4).astype(int)
    df['if1_is_10']   = (df['if_1'] == 10).astype(int)
    df['if1_is_6']    = (df['if_1'] == 6).astype(int)
    df['log_if1']     = np.log1p(df['if_1'])
    df['if1_nonzero'] = (df['if_1'] > 0).astype(int)

    # Combined
    df['if2_4_if1_0']   = ((df['if_2'] == 4) & (df['if_1'] == 0)).astype(int)
    df['if2_10_if1_nz'] = ((df['if_2'] == 10) & (df['if_1'] > 0)).astype(int)
    df['if_sum']        = df['if_1'] + df['if_2']
    df['log_if_sum']    = np.log1p(df['if_sum'])

    # ★ NEW v4: interaction features
    df['if_product']    = df['if_1'] * df['if_2']
    df['if_ratio']      = df['if_2'] / (df['if_1'] + 1)
    df['if_diff']       = df['if_2'] - df['if_1']
    df['if_max']        = df[['if_1', 'if_2']].max(axis=1)
    df['if_min']        = df[['if_1', 'if_2']].min(axis=1)

    # ═══════════════════════════════════════════════════════════════
    # GROUP B: Identity columns
    # ═══════════════════════════════════════════════════════════════
    df['race_is_nan']    = df['race'].isna().astype(int)
    df['religion_is_nan']= df['religion'].isna().astype(int)
    df['gender_is_nan']  = df['gender'].isna().astype(int)
    df['all_id_nan']     = (df['race_is_nan'] & df['religion_is_nan'] & df['gender_is_nan']).astype(int)
    df['n_id_nans']      = df['race_is_nan'] + df['religion_is_nan'] + df['gender_is_nan']

    df['race']     = df['race'].fillna('__nan__')
    df['religion'] = df['religion'].fillna('__nan__')
    df['gender']   = df['gender'].fillna('__nan__')

    # Race flags
    df['race_black']   = (df['race'] == 'black').astype(int)
    df['race_white']   = (df['race'] == 'white').astype(int)
    df['race_asian']   = (df['race'] == 'asian').astype(int)
    df['race_latino']  = (df['race'] == 'latino').astype(int)
    df['race_none']    = (df['race'] == 'none').astype(int)
    df['race_nonzero'] = (df['race'].isin(['black','white','asian','latino','other'])).astype(int)

    # Religion flags
    df['religion_muslim']    = (df['religion'] == 'muslim').astype(int)
    df['religion_christian'] = (df['religion'] == 'christian').astype(int)
    df['religion_jewish']    = (df['religion'] == 'jewish').astype(int)
    df['religion_nonzero']   = (df['religion'].isin(['muslim','christian','jewish','atheist','buddhist','hindu','other'])).astype(int)

    # Gender flags
    df['gender_female']      = (df['gender'] == 'female').astype(int)
    df['gender_male']        = (df['gender'] == 'male').astype(int)
    df['gender_transgender'] = (df['gender'] == 'transgender').astype(int)
    df['gender_nonzero']     = (df['gender'].isin(['female','male','transgender','other'])).astype(int)

    df['any_identity_mentioned'] = (df['race_nonzero'] | df['religion_nonzero'] | df['gender_nonzero']).astype(int)
    df['if2_10_race_nz']  = ((df['if_2'] == 10) & (df['race_nonzero'] == 1)).astype(int)
    # ★ NEW v4: more combo signals
    df['if2_10_relig_nz'] = ((df['if_2'] == 10) & (df['religion_nonzero'] == 1)).astype(int)
    df['if2_10_gender_nz']= ((df['if_2'] == 10) & (df['gender_nonzero'] == 1)).astype(int)
    df['identity_count']  = df['race_nonzero'] + df['religion_nonzero'] + df['gender_nonzero']

    df['disability'] = df['disability'].astype(int)

    for col in ['race', 'religion', 'gender']:
        if label_enc and col in label_enc:
            le = label_enc[col]
            df[f'{col}_enc'] = df[col].map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )
        else:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
            if label_enc is not None:
                label_enc[col] = le

    # ═══════════════════════════════════════════════════════════════
    # GROUP C: Engagement
    # ═══════════════════════════════════════════════════════════════
    df['log_upvote']      = np.log1p(df['upvote'])
    df['log_downvote']    = np.log1p(df['downvote'])
    df['total_votes']     = df['upvote'] + df['downvote']
    df['log_total_votes'] = np.log1p(df['total_votes'])
    df['vote_ratio']      = df['upvote'] / (df['total_votes'] + 1)
    df['controversy']     = df['downvote'] / (df['upvote'] + 1)
    # ★ NEW v4
    df['vote_diff']       = df['upvote'] - df['downvote']
    df['log_vote_diff']   = np.log1p(np.abs(df['vote_diff'])) * np.sign(df['vote_diff'])

    # ═══════════════════════════════════════════════════════════════
    # GROUP D: Text stats
    # ═══════════════════════════════════════════════════════════════
    df['char_count']       = df['comment'].str.len()
    df['word_count']       = df['comment'].str.split().str.len().fillna(0).astype(int)
    df['avg_word_len']     = df['char_count'] / (df['word_count'] + 1)
    df['unique_words']     = df['comment'].apply(lambda x: len(set(str(x).lower().split())))
    df['unique_ratio']     = df['unique_words'] / (df['word_count'] + 1)
    df['caps_ratio']       = df['comment'].str.count(r'[A-Z]') / (df['char_count'] + 1)
    df['caps_word_cnt']    = df['comment'].apply(
        lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1)
    )
    df['special_char_cnt'] = df['comment'].str.count(r'[^\w\s]')
    df['exclamation_cnt']  = df['comment'].str.count('!')
    df['question_cnt']     = df['comment'].str.count(r'\?')
    df['digit_cnt']        = df['comment'].str.count(r'\d')
    df['sentence_cnt']     = df['comment'].str.count(r'[.!?]') + 1
    df['avg_sent_len']     = df['word_count'] / (df['sentence_cnt'] + 1)
    df['ellipsis_cnt']     = df['comment'].str.count(r'\.{3}')
    df['url_cnt']          = df['comment'].str.count(r'http|www\.')
    # ★ NEW v4: additional text signals
    df['hashtag_cnt']      = df['comment'].str.count(r'#\w+')
    df['mention_cnt']      = df['comment'].str.count(r'@\w+')
    df['punct_ratio']      = df['special_char_cnt'] / (df['char_count'] + 1)
    df['has_url']          = (df['url_cnt'] > 0).astype(int)
    df['short_comment']    = (df['word_count'] <= 5).astype(int)
    df['long_comment']     = (df['word_count'] >= 50).astype(int)

    # ═══════════════════════════════════════════════════════════════
    # GROUP E: Temporal
    # ═══════════════════════════════════════════════════════════════
    df['created_date'] = pd.to_datetime(df['created_date'], utc=True)
    df['hour']         = df['created_date'].dt.hour
    df['day_of_week']  = df['created_date'].dt.dayofweek
    df['month']        = df['created_date'].dt.month
    df['year']         = df['created_date'].dt.year
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin']     = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']     = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']      = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']      = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin']    = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']    = np.cos(2 * np.pi * df['month'] / 12)

    # ═══════════════════════════════════════════════════════════════
    # GROUP F: Emoticons & post_id
    # ═══════════════════════════════════════════════════════════════
    df['emoticon_sum']  = df['emoticon_1'] + df['emoticon_2'] + df['emoticon_3']
    df['has_emoticon']  = (df['emoticon_sum'] > 0).astype(int)
    # ★ NEW v4: emoticon diversity
    df['emoticon_diversity'] = ((df['emoticon_1'] > 0).astype(int) +
                                (df['emoticon_2'] > 0).astype(int) +
                                (df['emoticon_3'] > 0).astype(int))

    # Post-ID target encoding
    if is_train:
        global_mean = df['label'].mean() if 'label' in df.columns else 0.794
        alpha = 10
        stats = df.groupby('post_id')['label'].agg(['mean', 'count'])
        stats['smoothed'] = (stats['mean'] * stats['count'] + global_mean * alpha) / (stats['count'] + alpha)
        post_id_stats = stats['smoothed'].to_dict()
    df['post_id_encoded'] = df['post_id'].map(post_id_stats).fillna(
        np.mean(list(post_id_stats.values()))
    )

    # ★ NEW v4: Post-level aggregated features
    all_data = pd.concat([train[['post_id','if_2','upvote','downvote']],
                          test[['post_id','if_2','upvote','downvote']]], ignore_index=True)
    if is_train and post_agg_stats is None:
        post_agg = all_data.groupby('post_id').agg(
            post_comment_count=('if_2', 'count'),
            post_mean_if2=('if_2', 'mean'),
            post_mean_upvote=('upvote', 'mean'),
            post_std_upvote=('upvote', 'std'),
        ).fillna(0)
        post_agg_stats = post_agg.to_dict()

    if post_agg_stats is not None:
        for col_name in ['post_comment_count', 'post_mean_if2', 'post_mean_upvote', 'post_std_upvote']:
            if col_name in post_agg_stats:
                df[col_name] = df['post_id'].map(post_agg_stats[col_name]).fillna(0)

    # ── Feature list ──
    feature_cols = [
        # if_2 flags
        'if_2', 'log_if2', 'if2_is_4', 'if2_is_10', 'if2_is_5', 'if2_is_6', 'if2_gt_10',
        # if_1 flags
        'if_1', 'log_if1', 'if1_is_0', 'if1_is_4', 'if1_is_10', 'if1_is_6', 'if1_nonzero',
        # Combined if
        'if2_4_if1_0', 'if2_10_if1_nz', 'if_sum', 'log_if_sum',
        'if_product', 'if_ratio', 'if_diff', 'if_max', 'if_min',
        # Race
        'race_is_nan', 'race_black', 'race_white', 'race_asian', 'race_latino', 'race_none', 'race_nonzero',
        # Religion
        'religion_is_nan', 'religion_muslim', 'religion_christian', 'religion_jewish', 'religion_nonzero',
        # Gender
        'gender_is_nan', 'gender_female', 'gender_male', 'gender_transgender', 'gender_nonzero',
        # Combined identity
        'all_id_nan', 'n_id_nans', 'any_identity_mentioned',
        'if2_10_race_nz', 'if2_10_relig_nz', 'if2_10_gender_nz', 'identity_count',
        'race_enc', 'religion_enc', 'gender_enc', 'disability',
        # Engagement
        'upvote', 'downvote', 'log_upvote', 'log_downvote',
        'total_votes', 'log_total_votes', 'vote_ratio', 'controversy',
        'vote_diff', 'log_vote_diff',
        # Text stats
        'char_count', 'word_count', 'avg_word_len', 'unique_words', 'unique_ratio',
        'caps_ratio', 'caps_word_cnt', 'special_char_cnt',
        'exclamation_cnt', 'question_cnt', 'digit_cnt',
        'sentence_cnt', 'avg_sent_len', 'ellipsis_cnt', 'url_cnt',
        'hashtag_cnt', 'mention_cnt', 'punct_ratio', 'has_url',
        'short_comment', 'long_comment',
        # Temporal
        'hour', 'day_of_week', 'month', 'year', 'is_weekend',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        # Emoticons + post_id
        'emoticon_1', 'emoticon_2', 'emoticon_3', 'emoticon_sum', 'has_emoticon',
        'emoticon_diversity',
        'post_id', 'post_id_encoded',
    ]

    # Add post-level features if available
    for col_name in ['post_comment_count', 'post_mean_if2', 'post_mean_upvote', 'post_std_upvote']:
        if col_name in df.columns:
            feature_cols.append(col_name)

    return df, feature_cols, post_id_stats, post_agg_stats


label_enc = {}
train_fe, feature_cols, post_id_stats, post_agg_stats = engineer_features(
    train, is_train=True, label_enc=label_enc
)
test_fe, _, _, _ = engineer_features(
    test, is_train=False,
    post_id_stats=post_id_stats,
    label_enc=label_enc,
    post_agg_stats=post_agg_stats,
)

print(f'Total structured features: {len(feature_cols)}')
print('\n=== Sanity check: if2==4 by label ===')
print(train_fe.groupby('label')['if2_is_4'].mean())
print('\n=== Sanity check: if1_nonzero by label ===')
print(train_fe.groupby('label')['if1_nonzero'].mean())

# %% [markdown]
# ## Text Embeddings (Dual Pathway)

# %% TF-IDF + SVD
print('Building TF-IDF + SVD text features...')

tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2), max_features=50_000,
    lowercase=True, strip_accents='unicode',
    analyzer='word', min_df=2, sublinear_tf=True,
)
tfidf_char = TfidfVectorizer(
    ngram_range=(2, 4), max_features=25_000,
    lowercase=True, analyzer='char_wb',
    min_df=3, sublinear_tf=True,
)

tfidf_word.fit(train_fe['comment'])
tfidf_char.fit(train_fe['comment'])

X_text_train = hstack([tfidf_word.transform(train_fe['comment']),
                        tfidf_char.transform(train_fe['comment'])])
X_text_test  = hstack([tfidf_word.transform(test_fe['comment']),
                        tfidf_char.transform(test_fe['comment'])])

print(f'TF-IDF matrix: {X_text_train.shape}')

SVD_N = 10 if SMOKE_TEST else 300
svd = TruncatedSVD(n_components=SVD_N, random_state=42, n_iter=7)
X_svd_train = svd.fit_transform(X_text_train)
X_svd_test  = svd.transform(X_text_test)
print(f'SVD({SVD_N}) explained variance: {svd.explained_variance_ratio_.sum():.1%}')

# %% Sentence-Transformer Embeddings
ST_DIM = 0
X_st_train = np.empty((len(train), 0), dtype=np.float32)
X_st_test  = np.empty((len(test), 0),  dtype=np.float32)

if HAS_ST:
    print('Computing sentence-transformer embeddings (BAAI/bge-base-en-v1.5)...')
    try:
        st_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda' if USE_GPU else 'cpu')
        X_st_train = st_model.encode(
            train_fe['comment'].tolist(), batch_size=256,
            show_progress_bar=True, normalize_embeddings=True
        ).astype(np.float32)
        X_st_test = st_model.encode(
            test_fe['comment'].tolist(), batch_size=256,
            show_progress_bar=True, normalize_embeddings=True
        ).astype(np.float32)
        ST_DIM = X_st_train.shape[1]
        print(f'✅ Sentence-transformer embeddings: {X_st_train.shape}')
        del st_model
        import gc; gc.collect()
        if USE_GPU:
            import torch; torch.cuda.empty_cache()
    except Exception as e:
        print(f'⚠️ Sentence-transformer failed: {e}')
        print('Continuing with TF-IDF only...')
        HAS_ST = False
else:
    print('Skipping sentence-transformer embeddings')

# %% Combine all features
X_struct_train = train_fe[feature_cols].values.astype(np.float32)
X_struct_test  = test_fe[feature_cols].values.astype(np.float32)

components = [X_struct_train, X_svd_train.astype(np.float32)]
components_test = [X_struct_test, X_svd_test.astype(np.float32)]

if ST_DIM > 0:
    components.append(X_st_train)
    components_test.append(X_st_test)

X_full_train = np.hstack(components)
X_full_test  = np.hstack(components_test)

print(f'Combined matrix: {X_full_train.shape}')
print(f'  Structured: {len(feature_cols)} | SVD: {SVD_N} | ST: {ST_DIM}')

# %% [markdown]
# ## Multi-Model Training (5-Fold Stratified CV)

# %% LightGBM Training
y = train['label'].values
n_classes = 4
N_FOLDS = 2 if SMOKE_TEST else 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# ─── Model 1: LightGBM ───
print('\n' + '='*60)
print('MODEL 1: LightGBM')
print('='*60)

lgb_params = dict(
    objective='multiclass', num_class=n_classes,
    metric='multi_logloss',
    n_estimators=10 if SMOKE_TEST else 4000, learning_rate=0.03,
    num_leaves=31 if SMOKE_TEST else 255, max_depth=-1,
    min_child_samples=10,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.05, reg_lambda=0.1,
    class_weight='balanced',
    random_state=42, n_jobs=-1, verbose=-1,
    device='gpu' if USE_GPU else 'cpu',
)

oof_lgb  = np.zeros((len(train), n_classes))
test_lgb = np.zeros((len(test),  n_classes))
lgb_fold_scores = []
lgb_best_iters  = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full_train, y)):
    print(f'\n── Fold {fold+1}/{N_FOLDS} ──')
    Xtr, Xval = X_full_train[tr_idx], X_full_train[val_idx]
    ytr, yval = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)],
              callbacks=[lgb.early_stopping(200, verbose=False),
                         lgb.log_evaluation(500)])

    oof_lgb[val_idx] = model.predict_proba(Xval)
    test_lgb        += model.predict_proba(X_full_test) / N_FOLDS
    lgb_best_iters.append(model.best_iteration_)

    score = f1_score(yval, oof_lgb[val_idx].argmax(1), average='macro')
    lgb_fold_scores.append(score)
    print(f'  macro F1 = {score:.5f}  (best_iter={model.best_iteration_})')

lgb_oof_f1 = f1_score(y, oof_lgb.argmax(1), average='macro')
print(f'\n✅ LightGBM OOF macro F1: {lgb_oof_f1:.5f} ± {np.std(lgb_fold_scores):.5f}')
print(classification_report(y, oof_lgb.argmax(1)))

# %% Feature Importance
all_feat_names = feature_cols + [f'svd_{i}' for i in range(SVD_N)]
if ST_DIM > 0:
    all_feat_names += [f'st_{i}' for i in range(ST_DIM)]
fi = pd.DataFrame({'feature': all_feat_names,
                   'importance': model.feature_importances_}).sort_values(
    'importance', ascending=False
)
plt.figure(figsize=(10, 9))
plt.barh(fi['feature'].head(35)[::-1], fi['importance'].head(35)[::-1])
plt.title('Top 35 Feature Importances (v4)')
plt.tight_layout(); plt.show()
print('Top 10 features:')
print(fi.head(10)[['feature','importance']].to_string())

# %% XGBoost Training
oof_xgb  = np.zeros((len(train), n_classes))
test_xgb = np.zeros((len(test),  n_classes))
xgb_fold_scores = []

if HAS_XGB:
    print('\n' + '='*60)
    print('MODEL 2: XGBoost')
    print('='*60)

    # Compute sample weights for class imbalance
    class_counts = np.bincount(y)
    total = len(y)
    class_weights_arr = total / (n_classes * class_counts)
    sample_weights_train = np.array([class_weights_arr[yi] for yi in y])

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full_train, y)):
        print(f'\n── Fold {fold+1}/{N_FOLDS} ──')
        Xtr, Xval = X_full_train[tr_idx], X_full_train[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]

        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=n_classes,
            n_estimators=10 if SMOKE_TEST else 3000, learning_rate=0.03,
            max_depth=4 if SMOKE_TEST else 8, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.05, reg_lambda=0.1,
            tree_method='hist', device='cuda' if USE_GPU else 'cpu',
            random_state=42, n_jobs=-1, verbosity=0,
            eval_metric='mlogloss',
            early_stopping_rounds=200,
        )
        xgb_model.fit(
            Xtr, ytr,
            sample_weight=sample_weights_train[tr_idx],
            eval_set=[(Xval, yval)],
            verbose=500,
        )

        oof_xgb[val_idx] = xgb_model.predict_proba(Xval)
        test_xgb        += xgb_model.predict_proba(X_full_test) / N_FOLDS

        score = f1_score(yval, oof_xgb[val_idx].argmax(1), average='macro')
        xgb_fold_scores.append(score)
        print(f'  macro F1 = {score:.5f}')

    xgb_oof_f1 = f1_score(y, oof_xgb.argmax(1), average='macro')
    print(f'\n✅ XGBoost OOF macro F1: {xgb_oof_f1:.5f} ± {np.std(xgb_fold_scores):.5f}')
    print(classification_report(y, oof_xgb.argmax(1)))

# %% CatBoost Training
oof_cb  = np.zeros((len(train), n_classes))
test_cb = np.zeros((len(test),  n_classes))
cb_fold_scores = []

if HAS_CB:
    print('\n' + '='*60)
    print('MODEL 3: CatBoost')
    print('='*60)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full_train, y)):
        print(f'\n── Fold {fold+1}/{N_FOLDS} ──')
        Xtr, Xval = X_full_train[tr_idx], X_full_train[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]

        cb_model = CatBoostClassifier(
            iterations=10 if SMOKE_TEST else 3000, learning_rate=0.05,
            depth=4 if SMOKE_TEST else 8, l2_leaf_reg=3,
            auto_class_weights='Balanced',
            random_seed=42, verbose=500,
            task_type='GPU' if USE_GPU else 'CPU',
            early_stopping_rounds=200,
            eval_metric='TotalF1:average=Macro',
        )
        cb_model.fit(Xtr, ytr, eval_set=(Xval, yval), verbose=500)

        oof_cb[val_idx] = cb_model.predict_proba(Xval)
        test_cb        += cb_model.predict_proba(X_full_test) / N_FOLDS

        score = f1_score(yval, oof_cb[val_idx].argmax(1), average='macro')
        cb_fold_scores.append(score)
        print(f'  macro F1 = {score:.5f}')

    cb_oof_f1 = f1_score(y, oof_cb.argmax(1), average='macro')
    print(f'\n✅ CatBoost OOF macro F1: {cb_oof_f1:.5f} ± {np.std(cb_fold_scores):.5f}')
    print(classification_report(y, oof_cb.argmax(1)))

# %% Logistic Regression on Text
print('\n' + '='*60)
print('MODEL 4: Logistic Regression (Text)')
print('='*60)

# Combine SVD + transformer for LR input
lr_components = [X_svd_train]
lr_components_test = [X_svd_test]
if ST_DIM > 0:
    lr_components.append(X_st_train)
    lr_components_test.append(X_st_test)

X_lr_train = np.hstack(lr_components)
X_lr_test  = np.hstack(lr_components_test)

scaler = StandardScaler()
X_lr_train_s = scaler.fit_transform(X_lr_train)
X_lr_test_s  = scaler.transform(X_lr_test)

oof_lr  = np.zeros((len(train), n_classes))
test_lr = np.zeros((len(test),  n_classes))
lr_fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full_train, y)):
    lr = LogisticRegression(
        C=3.0, max_iter=10 if SMOKE_TEST else 2000, solver='lbfgs',
        class_weight='balanced', random_state=42,
        multi_class='multinomial', n_jobs=-1
    )
    lr.fit(X_lr_train_s[tr_idx], y[tr_idx])
    oof_lr[val_idx] = lr.predict_proba(X_lr_train_s[val_idx])
    test_lr        += lr.predict_proba(X_lr_test_s) / N_FOLDS
    score = f1_score(y[val_idx], oof_lr[val_idx].argmax(1), average='macro')
    lr_fold_scores.append(score)
    print(f'  Fold {fold+1}: {score:.5f}')

lr_oof_f1 = f1_score(y, oof_lr.argmax(1), average='macro')
print(f'\n✅ LR OOF macro F1: {lr_oof_f1:.5f}')
print(classification_report(y, oof_lr.argmax(1)))

# %% [markdown]
# ## Ensemble + Threshold Optimization

# %% Ensemble Weight Search
print('\n' + '='*60)
print('ENSEMBLE WEIGHT OPTIMIZATION')
print('='*60)

# Collect available OOF predictions
model_oofs = [('LGB', oof_lgb)]
model_tests = [('LGB', test_lgb)]
if HAS_XGB:
    model_oofs.append(('XGB', oof_xgb))
    model_tests.append(('XGB', test_xgb))
if HAS_CB:
    model_oofs.append(('CB', oof_cb))
    model_tests.append(('CB', test_cb))
model_oofs.append(('LR', oof_lr))
model_tests.append(('LR', test_lr))

n_models = len(model_oofs)
print(f'Ensembling {n_models} models: {[m[0] for m in model_oofs]}')

# Grid search over weights
best_ensemble_f1 = 0
best_weights = None

if n_models == 4:
    # 4-model grid search (coarse then fine)
    step = 0.1
    for w1 in np.arange(0, 1.01, step):
        for w2 in np.arange(0, 1.01 - w1, step):
            for w3 in np.arange(0, 1.01 - w1 - w2, step):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < -0.01:
                    continue
                w4 = max(0, w4)
                blend = (w1 * model_oofs[0][1] + w2 * model_oofs[1][1] +
                         w3 * model_oofs[2][1] + w4 * model_oofs[3][1])
                score = f1_score(y, blend.argmax(1), average='macro')
                if score > best_ensemble_f1:
                    best_ensemble_f1 = score
                    best_weights = [w1, w2, w3, w4]

    # Fine-tune around best
    print(f'Coarse search best: {best_ensemble_f1:.5f} weights={[f"{w:.2f}" for w in best_weights]}')
    fine_step = 0.02
    center = best_weights.copy()
    for dw1 in np.arange(-0.1, 0.11, fine_step):
        for dw2 in np.arange(-0.1, 0.11, fine_step):
            for dw3 in np.arange(-0.1, 0.11, fine_step):
                w1 = max(0, center[0] + dw1)
                w2 = max(0, center[1] + dw2)
                w3 = max(0, center[2] + dw3)
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0:
                    continue
                blend = (w1 * model_oofs[0][1] + w2 * model_oofs[1][1] +
                         w3 * model_oofs[2][1] + w4 * model_oofs[3][1])
                score = f1_score(y, blend.argmax(1), average='macro')
                if score > best_ensemble_f1:
                    best_ensemble_f1 = score
                    best_weights = [w1, w2, w3, w4]

elif n_models == 2:
    for w in np.arange(0, 1.01, 0.02):
        blend = w * model_oofs[0][1] + (1 - w) * model_oofs[1][1]
        score = f1_score(y, blend.argmax(1), average='macro')
        if score > best_ensemble_f1:
            best_ensemble_f1 = score
            best_weights = [w, 1 - w]
else:
    # For 3 models
    step = 0.05
    for w1 in np.arange(0, 1.01, step):
        for w2 in np.arange(0, 1.01 - w1, step):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            blend = (w1 * model_oofs[0][1] + w2 * model_oofs[1][1] +
                     w3 * model_oofs[2][1])
            score = f1_score(y, blend.argmax(1), average='macro')
            if score > best_ensemble_f1:
                best_ensemble_f1 = score
                best_weights = [w1, w2, w3]

print(f'\n✅ Best ensemble weights: {[f"{w:.3f}" for w in best_weights]}')
print(f'   Ensemble OOF macro F1: {best_ensemble_f1:.5f}')

# Create blended OOF
oof_blend = sum(w * oofs[1] for w, oofs in zip(best_weights, model_oofs))
print('\nEnsemble Classification Report:')
print(classification_report(y, oof_blend.argmax(1)))

# Confusion matrix
cm = confusion_matrix(y, oof_blend.argmax(1))
print('Confusion Matrix:')
print(pd.DataFrame(cm, index=[f'True {i}' for i in range(4)],
                       columns=[f'Pred {i}' for i in range(4)]))

# %% Per-Class Threshold Optimization
print('\n' + '='*60)
print('PER-CLASS THRESHOLD OPTIMIZATION')
print('='*60)

def apply_thresholds(probs, thresholds):
    """Apply per-class thresholds and return predictions."""
    adjusted = probs.copy()
    for c in range(probs.shape[1]):
        adjusted[:, c] = adjusted[:, c] + thresholds[c]
    return adjusted.argmax(1)

def threshold_objective(thresholds, probs, y_true):
    """Negative macro F1 (we minimize)."""
    preds = apply_thresholds(probs, thresholds)
    return -f1_score(y_true, preds, average='macro')

# Search per-class threshold adjustments
print('Searching for optimal per-class thresholds...')
best_thresh_f1 = best_ensemble_f1
best_thresholds = [0.0] * n_classes

# Grid search
for t0 in np.arange(-0.15, 0.16, 0.03):
    for t1 in np.arange(-0.15, 0.16, 0.03):
        for t2 in np.arange(-0.15, 0.16, 0.03):
            for t3 in np.arange(-0.15, 0.16, 0.03):
                thresholds = [t0, t1, t2, t3]
                preds = apply_thresholds(oof_blend, thresholds)
                score = f1_score(y, preds, average='macro')
                if score > best_thresh_f1:
                    best_thresh_f1 = score
                    best_thresholds = thresholds

print(f'\n✅ Best thresholds: {[f"{t:.3f}" for t in best_thresholds]}')
print(f'   After threshold tuning: {best_thresh_f1:.5f}  (was {best_ensemble_f1:.5f})')

final_oof_preds = apply_thresholds(oof_blend, best_thresholds)
print('\nFinal Classification Report (after threshold tuning):')
print(classification_report(y, final_oof_preds))

# %% [markdown]
# ## Full Retrain + Submission

# %% Full Retrain
print('\n' + '='*60)
print('FULL RETRAIN + SUBMISSION')
print('='*60)

# Full retrain LightGBM
best_n_lgb = min(10, max(int(np.mean(lgb_best_iters) * 1.05), 5)) if SMOKE_TEST else max(int(np.mean(lgb_best_iters) * 1.05), 100)
print(f'Full retrain LightGBM: n_estimators={best_n_lgb}')
lgb_full = lgb.LGBMClassifier(**{**lgb_params, 'n_estimators': best_n_lgb})
lgb_full.fit(X_full_train_aug, y_aug, callbacks=[lgb.log_evaluation(200)])
test_lgb_full = lgb_full.predict_proba(X_full_test)

test_preds_list = [(best_weights[0], test_lgb_full)]
wi = 1

# Full retrain XGBoost
if HAS_XGB:
    print('Full retrain XGBoost...')
    xgb_full = xgb.XGBClassifier(
        objective='multi:softprob', num_class=n_classes,
        n_estimators=10 if SMOKE_TEST else 2500, learning_rate=0.03,
        max_depth=4 if SMOKE_TEST else 8, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.05, reg_lambda=0.1,
        tree_method='hist', device='cuda' if USE_GPU else 'cpu',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_full.fit(X_full_train_aug, y_aug, sample_weight=sample_weights_train_aug, verbose=200)
    test_xgb_full = xgb_full.predict_proba(X_full_test)
    test_preds_list.append((best_weights[wi], test_xgb_full))
    wi += 1

# Full retrain CatBoost
if HAS_CB:
    print('Full retrain CatBoost...')
    cb_full = CatBoostClassifier(
        iterations=10 if SMOKE_TEST else 2500, learning_rate=0.05,
        depth=4 if SMOKE_TEST else 8, l2_leaf_reg=3,
        auto_class_weights='Balanced',
        random_seed=42, verbose=200,
        task_type='GPU' if USE_GPU else 'CPU',
    )
    cb_full.fit(X_full_train_aug, y_aug, verbose=200)
    test_cb_full = cb_full.predict_proba(X_full_test)
    test_preds_list.append((best_weights[wi], test_cb_full))
    wi += 1

# Full retrain LR
print('Full retrain LR...')
lr_full = LogisticRegression(
    C=3.0, max_iter=10 if SMOKE_TEST else 2000, solver='lbfgs',
    class_weight='balanced', random_state=42,
    multi_class='multinomial', n_jobs=-1
)
lr_full.fit(X_lr_train_s_aug, y_aug)
test_lr_full = lr_full.predict_proba(X_lr_test_s)
test_preds_list.append((best_weights[wi], test_lr_full))

# Final blended test predictions
test_blend_full = sum(w * p for w, p in test_preds_list)
test_final_preds = apply_thresholds(test_blend_full, best_thresholds)

# %% Submission
print('\nTest prediction distribution:')
unique, counts = np.unique(test_final_preds, return_counts=True)
for u, c in zip(unique, counts):
    print(f'  Label {u}: {c:,} ({c/len(test_final_preds)*100:.1f}%)')

submission = sample.head(len(test_final_preds)).copy() if SMOKE_TEST else sample.copy()
submission['label'] = test_final_preds
submission.to_csv('submission.csv', index=False)

print(f'\n✅ submission.csv saved | Shape: {submission.shape}')
print(submission.head())

print('\n' + '='*60)
print('FINAL RESULTS SUMMARY')
print('='*60)
print(f'  LightGBM OOF macro F1       : {lgb_oof_f1:.5f}')
if HAS_XGB:
    print(f'  XGBoost  OOF macro F1       : {xgb_oof_f1:.5f}')
if HAS_CB:
    print(f'  CatBoost OOF macro F1       : {cb_oof_f1:.5f}')
print(f'  LR (text) OOF macro F1      : {lr_oof_f1:.5f}')
print(f'  Ensemble OOF macro F1       : {best_ensemble_f1:.5f}')
print(f'  + Threshold tuning          : {best_thresh_f1:.5f}')
print(f'  Weights: {[f"{m[0]}={w:.3f}" for m, w in zip(model_oofs, best_weights)]}')
print(f'  Thresholds: {[f"{t:.3f}" for t in best_thresholds]}')
print('='*60)
print(f'  v3 Baseline (LB)            : 0.65656')
print('='*60)
