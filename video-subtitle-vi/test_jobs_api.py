"""Kiểm tra xoá job, tạo job từ video đã tải, job kẹt khi khởi động (cần fastapi + httpx, chạy trong container): python3 test_jobs_api.py"""
import json
import os
import tempfile
from pathlib import Path

DATA = Path(tempfile.mkdtemp())
os.environ["DATA_DIR"] = str(DATA)


def put(kind, id_, data, video=None):
    d = DATA / kind / id_
    d.mkdir(parents=True)
    (d / ("job.json" if kind == "jobs" else "info.json")).write_text(json.dumps(data))
    if video:
        (d / video).write_bytes(b"fake")


put("jobs", "a" * 32, {"id": "a" * 32, "status": "ocr", "video": "input.mp4"})  # kẹt từ lần chạy trước
put("jobs", "b" * 32, {"id": "b" * 32, "status": "ready", "render_status": "rendering", "video": "input.mp4"})
put("downloads", "c" * 32, {"id": "c" * 32, "title": "Clip: hay/quá", "file": "video.mp4", "ext": ".mp4"}, "video.mp4")

from fastapi.testclient import TestClient  # noqa: E402
from app import main  # noqa: E402  (import = chạy fail_interrupted_jobs)

main.executor.submit = lambda *a, **k: None  # không chạy xử lý thật
c = TestClient(main.app)

# Job dở dang lúc khởi động -> lỗi, xoá được
assert c.get(f"/api/jobs/{'a' * 32}").json()["status"] == "error"
assert c.get(f"/api/jobs/{'b' * 32}").json()["render_status"] == "error"
assert c.delete(f"/api/jobs/{'a' * 32}").status_code == 200
assert not (DATA / "jobs" / ("a" * 32)).exists()

# Đang xử lý -> không cho xoá
put("jobs", "d" * 32, {"id": "d" * 32, "status": "transcribing", "video": "input.mp4"})
assert c.delete(f"/api/jobs/{'d' * 32}").status_code == 409
assert c.delete("/api/jobs/xyz").status_code == 404

# Tạo job từ video đã tải: chép file trên máy chủ, đặt tên từ tiêu đề
r = c.post("/api/jobs", data={"download_id": "c" * 32})
assert r.status_code == 200, r.text
job = c.get(f"/api/jobs/{r.json()['id']}").json()
assert job["filename"] == "Clip hay quá.mp4", job["filename"]
assert (DATA / "jobs" / job["id"] / "input.mp4").read_bytes() == b"fake"
assert c.post("/api/jobs", data={"download_id": "0" * 32}).status_code == 404
assert c.post("/api/jobs", data={}).status_code == 400
print("ok")
