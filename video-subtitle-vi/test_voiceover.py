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
# Âm lượng tiếng gốc: kẹp 0..100, rác -> mặc định
from app.pipeline import TTS_BG_VOLUME, valid_bg_volume
assert valid_bg_volume("5") == 5 and valid_bg_volume(0) == 0 and valid_bg_volume(300) == 100
assert valid_bg_volume(None) == TTS_BG_VOLUME == 15
print("ok")
