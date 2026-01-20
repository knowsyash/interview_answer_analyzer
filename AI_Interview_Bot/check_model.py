import joblib
import os

model_path = r'c:\Users\yashs\Desktop\New folder (2)\interview_answer_analyzer\AI_Interview_Bot\real_dataset_score\random_forest_model.joblib'

print(f"Loading model from: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

model_data = joblib.load(model_path)

print(f"\nModel data type: {type(model_data)}")

if isinstance(model_data, dict):
    print("\nModel is a dictionary with keys:")
    for key in model_data.keys():
        print(f"  - {key}")
    
    if 'model' in model_data:
        model = model_data['model']
        print(f"\nActual model type: {type(model)}")
        print(f"Features expected: {model.n_features_in_}")
        
    if 'feature_names' in model_data:
        print(f"\nFeature names ({len(model_data['feature_names'])}):")
        for i, name in enumerate(model_data['feature_names']):
            print(f"  {i+1}. {name}")
else:
    # Direct model object
    print(f"\nDirect model type: {type(model_data)}")
    if hasattr(model_data, 'n_features_in_'):
        print(f"Features expected: {model_data.n_features_in_}")
    if hasattr(model_data, 'classes_'):
        print(f"Classes: {model_data.classes_}")
