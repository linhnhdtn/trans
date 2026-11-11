"""Kiểm tra xếp clip giọng đọc lên timeline: python3 test_voiceover.py"""
from app.pipeline import TTS_RATE as R, place_clips

# Đúng mốc khi không chồng nhau
assert place_clips([(0.0, R), (2.0, R)]) == [0, 2 * R]
# Câu trước đọc dài -> câu sau lùi lại, không đè tiếng
assert place_clips([(0.0, 3 * R), (2.0, R), (2.5, R)]) == [0, 3 * R, 4 * R]
assert place_clips([]) == []
print("ok")
