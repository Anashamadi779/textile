import io
import requests
from PIL import Image

img = Image.new('RGB', (640, 480), color=(200, 200, 200))
buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)

resp = requests.post(
    'http://127.0.0.1:8001/analyze',
    files={'file': ('test.png', buf, 'image/png')},
    params={'conf': 0.25, 'iou': 0.45},
    timeout=10,
)
print('status_code:', resp.status_code)
print(resp.text)
