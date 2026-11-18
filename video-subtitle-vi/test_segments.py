"""Phụ đề ngắt theo cụm có nghĩa, không cắt giữa cụm từ: python3 test_segments.py"""
from app.pipeline import build_segments


def words(tokens, step=0.3):
    return [{"start": i * step, "end": i * step + step - 0.02, "text": t, "seg_start": i == 0}
            for i, t in enumerate(tokens)]


# Câu thật trong video lỗi: trước đây bị cắt "…很奇怪 | 的梦…" và "…沙发上 | 醒过来…"
zh = ["那天", "晚上,", "我", "做了", "一个", "很", "奇怪", "的", "梦,", "梦里,", "我", "从", "一张", "沙发", "上",
      "醒过来,", "发现", "自己", "身处", "在", "一个", "完全", "陌生", "的", "房间。"]
got = [s["src"] for s in build_segments(words(zh), "zh")]
assert got == ["那天晚上,我做了一个很奇怪的梦,", "梦里,我从一张沙发上醒过来,", "发现自己身处在一个完全陌生的房间。"], got

# Không có dấu câu nào thì vẫn phải cắt, không để 1 đoạn dài vô hạn
long = build_segments(words(["一二三"] * 20, step=0.1), "zh")
assert len(long) > 1 and all(len(s["src"]) <= 21 for s in long), [s["src"] for s in long]
print("ok")
