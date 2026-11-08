"""Where verdicts come from: the local model (default) or a remote VisionGuard API (Task 8)."""
import requests


class LocalPredictor:
    def __init__(self, model_path, threshold=None):
        from app.inference import VisionGuardModel   # TensorFlow is only imported in local mode
        self.model = VisionGuardModel(model_path, threshold)

    def predict(self, data, filename):
        return self.model.predict_bytes(data)        # raises ValueError if not decodable

    def health(self):
        return {"mode": "local", "model_loaded": True, "model_version": self.model.version,
                "threshold": self.model.threshold, "load_seconds": round(self.model.load_seconds, 1)}


class RemotePredictor:
    def __init__(self, url, api_key, timeout=60):
        self.url, self.api_key, self.timeout = url.rstrip("/"), api_key, timeout

    def predict(self, data, filename):
        try:
            r = requests.post(f"{self.url}/api/predict", files={"image": (filename, data)},
                              headers={"X-API-Key": self.api_key}, timeout=self.timeout)
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach the model API ({type(e).__name__}).") from e
        if r.status_code in (400, 413, 415, 422):
            raise ValueError(r.json().get("error", "The model API rejected the image."))
        if r.status_code != 200:
            raise RuntimeError(f"The model API answered HTTP {r.status_code}.")
        return r.json()

    def health(self):
        try:
            r = requests.get(f"{self.url}/health", timeout=10)
            return {"mode": "remote", "api_url": self.url, "api_health": r.json()}
        except (requests.RequestException, ValueError) as e:
            return {"mode": "remote", "api_url": self.url, "api_health": f"unreachable ({type(e).__name__})"}
