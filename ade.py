import re

import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from loguru import logger

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 初始化模型
model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    # vad_model="fsmn-vad",  # 可选：语音活动检测
    device=device,  # 使用GPU加速
    disable_update=True,
)

# 识别音频文件
audio_path = "./input/vad_example.wav"
res = model.generate(
    input=audio_path,
    language="auto",  # 自动检测语言
    use_itn=True,  # 逆文本正则化
    batch_size_s=60,
    merge_vad=True,
)


def parse_events_from_text(raw_text):
    """提取音频事件和情感标签"""
    # 定义事件类型
    event_list = ["Speech", "Applause", "BGM", "Laughter", "Cry", "Sneeze", "Breath", "Cough"]
    emotion_list = ["NEUTRAL", "HAPPY", "ANGRY", "SAD", "FEARFUL", "DISGUSTED", "SURPRISED"]

    # 提取事件标签
    event_pattern = r"<\|(" + "|".join(event_list) + r")\|>"
    emotion_pattern = r"<\|(" + "|".join(emotion_list) + r")\|>"

    events = re.findall(event_pattern, raw_text)
    emotions = re.findall(emotion_pattern, raw_text)

    # 清理标签，获取纯文本
    clean_text = rich_transcription_postprocess(raw_text)

    return {
        "clean_text": clean_text,
        "events": list(set(events)),  # 去重
        "emotions": list(set(emotions)),
    }


# 获取原始结果（包含事件标签）
raw_text = res[0].get("text", "")
logger.info("原始输出: {}", raw_text)

# 使用解析函数
result = parse_events_from_text(raw_text)
logger.info(f"转写文本: {result['clean_text']}")
logger.info(f"检测事件: {result['events']}")  # 如: ['Applause', 'Laughter']
logger.info(f"情感标签: {result['emotions']}")  # 如: ['HAPPY']
