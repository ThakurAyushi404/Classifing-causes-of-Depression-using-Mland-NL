from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Combine oversampling and undersampling techniques
oversample = SMOTE(sampling_strategy={1: 500, 2: 500})  # Increase minority samples
undersample = RandomUnderSampler(sampling_strategy={0: 1500})  # Reduce majority class

pipeline = Pipeline(steps=[('o', oversample), ('u', undersample)])

# Apply resampling to the dataset
X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train_tfidf, y_train)

print(f"\n✅ Step 7: Dataset resampled. New Class Distribution:\n{pd.Series(y_train_resampled).value_counts()}")
