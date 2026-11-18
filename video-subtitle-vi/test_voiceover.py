"""Kiểm tra xếp clip giọng đọc lên timeline + tốc độ đọc: python3 test_voiceover.py"""
from app.pipeline import TTS_RATE as R, _atempo, place_clips, timeline_map, voice_key

# Đúng mốc khi không chồng nhau
assert place_clips([(0.0, R), (2.0, R)]) == [0, 2 * R]
# Câu trước đọc dài -> câu sau lùi lại, không đè tiếng
assert place_clips([(0.0, 3 * R), (2.0, R), (2.5, R)]) == [0, 3 * R, 4 * R]
assert place_clips([]) == []

# Giọng: khóa cũ edge-tts đổi sang VieNeu, khóa lạ -> tắt giọng
assert voice_key("nam") == "Hải Đăng" and voice_key("Thục Đoan") == "Thục Đoan"
assert voice_key("../x") == "" and voice_key("") == ""
# Kéo dài video: đoạn [10, 12) chậm 2 lần -> mọi thứ sau giây 12 lùi 2s, trước 10 giữ nguyên
st = [(10.0, 12.0, 2.0)]
assert timeline_map(5, st) == 5 and timeline_map(11, st) == 12 and timeline_map(20, st) == 22
assert timeline_map(20, st + [(15.0, 16.0, 1.5)]) == 22.5
assert _atempo(0.8) == "atempo=0.80000" and _atempo(0.2) == "atempo=0.5,atempo=0.5,atempo=0.80000"
# Âm lượng tiếng gốc / nhạc nền: kẹp 0..200, rác -> mặc định
from app.pipeline import TTS_BG_VOLUME, valid_bg_volume
assert valid_bg_volume("5") == 5 and valid_bg_volume(0) == 0 and valid_bg_volume(300) == 200
assert valid_bg_volume(None) == TTS_BG_VOLUME == 15
# Giọng nhân bản: file data/voices/<tên>.wav thành giọng chọn được, khóa cache kèm mtime file mẫu
import importlib, os, tempfile
import app.pipeline as P
with tempfile.TemporaryDirectory() as d:
    os.makedirs(f"{d}/voices")
    open(f"{d}/voices/Giọng thử.wav", "wb").close()
    os.environ["DATA_DIR"] = d
    P = importlib.reload(P)
    assert P.voice_key("Giọng thử") == "Giọng thử" and P.voice_id("Giọng thử").startswith("Giọng thử@")
    assert P.voice_id("Hải Đăng") == "Hải Đăng"
# Bỏ giọng gốc: chỉ nhận 3 chế độ; auto chọn theo máy (GPU -> Roformer, CPU -> Demucs)
from app.pipeline import valid_bgm, bgm_method, has_gpu
assert valid_bgm("fast") == "fast" and valid_bgm("x") == "" and valid_bgm(None) == ""
assert bgm_method("auto") == ("clean" if has_gpu() else "fast") and bgm_method("clean") == "clean"
print("ok")
