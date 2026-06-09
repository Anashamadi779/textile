import io
import requests
from PIL import Image

API = "http://127.0.0.1:8000/analyze"

# create a simple RGB image
img = Image.new('RGB', (640, 480), color=(200, 200, 200))
buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)

files = {'file': ('test.png', buf, 'image/png')}
params = {'conf': 0.25, 'iou': 0.45}

resp = requests.post(API, files=files, params=params, timeout=10)
print('status_code:', resp.status_code)
try:
    print(resp.json())
except Exception:
    print(resp.text)
