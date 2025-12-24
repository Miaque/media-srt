import os
import tempfile

import librosa
import soundfile as sf
import torch
from funasr import AutoModel
from loguru import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 检测可用设备
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"使用设备: {device}")

# ==================== ASR模型配置 ====================
# paraformer-zh: 阿里达摩院开源的中文语音识别大模型
# 集成了VAD、标点预测、说话人识别等功能

# VAD配置：优化语音活动检测
vad_kwargs = {
    "max_single_segment_time": 30000,  # 单段最长30秒，防止过长片段
    "speech_noise_thres": 0.8,  # 语音/噪音阈值，0.8较为平衡（范围0-1，越高越严格）
}

# 说话人识别配置：优化多人对话场景
spk_kwargs = {
    "sv_threshold": 0.9465,  # 说话人相似度阈值（默认0.9465）
    # 如果说话人区分不够清晰，可以降低到0.90-0.92
    # 如果误判太多（同一人被识别为多人），可以提高到0.95-0.97
}

logger.info("初始化ASR模型...")
model = AutoModel(
    # 使用大模型以获得最佳识别效果
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    # VAD模型：语音活动检测，过滤静音和噪音
    vad_model="fsmn-vad",
    vad_kwargs=vad_kwargs,
    # 标点预测模型：提高可读性
    punc_model="ct-punc",
    # 说话人识别模型：区分不同说话人
    spk_model="cam++",
    spk_kwargs=spk_kwargs,
    # 禁用自动更新模型
    disable_update=True,
    # 设备配置
    device=device,
)
logger.info("ASR模型初始化完成")

# ==================== 音频文件配置 ====================
audio_file = "./input/1729694733.wav"

# 检查音频文件是否存在
if not os.path.exists(audio_file):
    logger.error(f"音频文件不存在: {audio_file}")
    raise FileNotFoundError(f"音频文件不存在: {audio_file}")

logger.info(f"开始处理音频文件: {audio_file}")

# ==================== ASR识别参数优化 ====================
# 根据音频特点调整参数以获得最佳效果

# 热词配置：提高特定词汇的识别准确度
# 支持多个热词，用空格分隔，例如："三茂老师 张三 李四 人工智能"
hotwords = "三茂老师"

res = model.generate(
    input=audio_file,
    # ===== 批处理配置 =====
    batch_size_s=300,  # 批处理大小（秒），长音频建议300-600
    # ===== VAD配置 =====
    merge_vad=True,  # 合并VAD检测到的语音段
    merge_length_s=15,  # 合并间隔小于15秒的语音段
    # 说明：
    # - 会议/多人对话场景：建议10-15秒（更精确的分段）
    # - 演讲/单人场景：建议20-30秒（减少片段数）
    # - 短对话场景：建议5-10秒（快速响应）
    # ===== 热词增强 =====
    hotword=hotwords,  # 提高特定词汇识别准确度
    # ===== 其他可选参数 =====
    # batch_size=1,  # 并行处理的音频数量
    # use_itn=True,  # 逆文本归一化（数字、日期等格式化）
)

logger.info("=" * 60)
logger.info("ASR识别完成")
logger.info("=" * 60)
logger.info("识别结果:")
logger.info(res)

# 加载完整音频
audio, sr = librosa.load(audio_file, sr=16000)
logger.info(f"音频采样率: {sr}, 时长: {len(audio) / sr:.2f}秒")

# 初始化情感识别pipeline
inference_pipeline = pipeline(task=Tasks.emotion_recognition, model="iic/emotion2vec_plus_large")

# 创建临时目录存储音频片段
temp_dir = tempfile.mkdtemp()


def merge_short_segments(segments, min_duration_ms=2000, max_duration_ms=15000, max_gap_ms=500):
    """
    合并较短的语音片段
    
    Args:
        segments: 原始片段列表
        min_duration_ms: 最小片段时长(毫秒)，短于此时长的片段会尝试合并
        max_duration_ms: 合并后的最大时长(毫秒)，避免合并后片段过长
        max_gap_ms: 最大间隔时长(毫秒)，片段间隔小于此值时更容易合并
    
    Returns:
        合并后的片段列表
    """
    if not segments:
        return []

    merged = []
    current_segment = None

    for segment in segments:
        duration = segment["end"] - segment["start"]

        if current_segment is None:
            # 第一个片段
            current_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "spk": segment.get("spk", "unknown"),
                "segments": [segment],  # 保存原始片段信息
            }
        else:
            current_duration = current_segment["end"] - current_segment["start"]
            same_speaker = current_segment["spk"] == segment.get("spk", "unknown")
            gap = segment["start"] - current_segment["end"]  # 计算片段间隔

            # 判断是否应该合并：
            # 条件1：必须是同一说话人
            # 条件2：满足以下任一情况
            #   - 当前片段很短（< min_duration）
            #   - 新片段很短（< min_duration）
            #   - 两个片段都较短且间隔很小
            # 条件3：合并后不会太长
            both_short_and_close = (
                    current_duration < min_duration_ms * 1.5
                    and duration < min_duration_ms * 1.5
                    and gap < max_gap_ms
            )
            total_duration = (
                    current_segment["end"] - current_segment["start"] + segment["end"] - segment["start"]
            )

            should_merge = (
                    same_speaker
                    and (current_duration < min_duration_ms or duration < min_duration_ms or both_short_and_close)
                    and total_duration <= max_duration_ms
            )

            # 调试日志（可选）
            # if duration < min_duration_ms or current_duration < min_duration_ms:
            #     logger.debug(
            #         f"片段: curr={current_duration}ms, new={duration}ms, gap={gap}ms, "
            #         f"same_spk={same_speaker}, merge={should_merge}"
            #     )

            if should_merge:
                # 合并片段
                current_segment["end"] = segment["end"]
                current_segment["text"] += segment["text"]
                current_segment["segments"].append(segment)
            else:
                # 保存当前片段，开始新片段
                merged.append(current_segment)
                current_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "spk": segment.get("spk", "unknown"),
                    "segments": [segment],
                }

    # 添加最后一个片段
    if current_segment is not None:
        merged.append(current_segment)

    return merged


# 对每个VAD分段进行情感识别
# 检查res是否为列表且包含结果
if res and isinstance(res, list) and len(res) > 0:
    result_dict = res[0]
    if isinstance(result_dict, dict) and "sentence_info" in result_dict:
        sentence_info = result_dict["sentence_info"]
        logger.info(f"\n原始片段数: {len(sentence_info)}")

        # 合并较短的片段
        merged_segments = merge_short_segments(
            sentence_info,
            min_duration_ms=2000,  # 短于2秒的片段会尝试合并
            max_duration_ms=15000,  # 合并后最长15秒
            max_gap_ms=500  # 间隔小于0.5秒的短片段更容易合并
        )
        logger.info(f"合并后片段数: {len(merged_segments)}")
        logger.info(f"\n开始对 {len(merged_segments)} 个语音片段进行情感识别:")

        for idx, segment in enumerate(merged_segments):
            start_ms = segment["start"]  # 毫秒
            end_ms = segment["end"]  # 毫秒
            text = segment["text"]
            spk = segment.get("spk", "unknown")

            # 转换为秒并提取音频片段
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)

            segment_audio = audio[start_sample:end_sample]

            # 保存临时音频片段
            temp_audio_path = os.path.join(temp_dir, f"segment_{idx}.wav")
            sf.write(temp_audio_path, segment_audio, sr)

            # 对该片段进行情感识别
            try:
                emotion_result = inference_pipeline(
                    input=temp_audio_path,
                    granularity="utterance",  # 因为已经是单个片段了
                    extract_embedding=False,
                )

                # 提取情感标签和分数
                if emotion_result and isinstance(emotion_result, list) and len(emotion_result) > 0:
                    # emotion_result 是一个列表，取第一个元素
                    first_result = emotion_result[0]
                    emotions = first_result.get("labels", [])
                    scores = first_result.get("scores", [])
                    # 找到分数最高的情感
                    if emotions and scores:
                        max_idx = scores.index(max(scores))
                        top_emotion = emotions[max_idx]
                        top_score = scores[max_idx]
                    else:
                        top_emotion = "unknown"
                        top_score = 0.0
                else:
                    top_emotion = "unknown"
                    top_score = 0.0
                    emotions = []
                    scores = []

                duration = end_sec - start_sec
                original_count = len(segment.get("segments", [segment]))

                logger.info(f"\n片段 {idx + 1}:")
                logger.info(f"  时间: {start_sec:.2f}s - {end_sec:.2f}s (时长: {duration:.2f}s)")
                logger.info(f"  说话人: {spk}")
                if original_count > 1:
                    logger.info(f"  合并自: {original_count} 个原始片段")
                logger.info(f"  文本: {text}")
                logger.info(f"  情感: {top_emotion} (置信度: {top_score:.3f})")
                if len(emotions) > 1:
                    logger.info(f"  其他情感: {list(zip(emotions[1:3], scores[1:3]))}")

            except Exception as e:
                logger.error(f"片段 {idx + 1} 情感识别失败: {e}")

            # 删除临时文件
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        # 清理临时目录
        os.rmdir(temp_dir)
else:
    logger.warning("未找到sentence_info,尝试整体识别")
    rec_result = inference_pipeline(
        input=audio_file,
        granularity="utterance",
        extract_embedding=False,
    )
    logger.info(rec_result)

if __name__ == '__main__':
    pass
