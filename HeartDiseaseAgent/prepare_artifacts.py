from pathlib import Path
import joblib
import shutil

src = Path(r'd:/Ml project/heart_disease_model (1).pkl')
dst = Path(r'd:/Ml project/HeartDiseaseAgent/models/heart_disease_model.pkl')
models_dir = dst.parent
models_dir.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src, dst)

class IdentityScaler:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

joblib.dump(IdentityScaler(), Path(r'd:/Ml project/HeartDiseaseAgent/models/scaler.pkl'))
print(dst.exists(), dst)
print(Path(r'd:/Ml project/HeartDiseaseAgent/models/scaler.pkl').exists())
