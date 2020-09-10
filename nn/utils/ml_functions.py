from sklearn.model_selection import KFold

def create_folds(N_SPLITS=5, SEED=42):
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    return folds