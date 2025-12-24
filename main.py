# example_asr.py
import os

import soundfile
from funasr import AutoModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

audio_file = "./input/vad_example.wav"  # 替换为你的音频文件路径


def parse_diarization_result(diarization_segments):
    """解析说话人分离模型返回的 [[start, end, id]] 格式列表。"""
    speaker_segments = []
    if not isinstance(diarization_segments, list):
        return []
    for segment in diarization_segments:
        if isinstance(segment, list) and len(segment) == 3:
            try:
                start_sec, end_sec = float(segment[0]), float(segment[1])
                speaker_id = (
                    f"说话人{int(segment[2]) + 1}"  # 转换为“说话人1”、“说话人2”格式
                )
                speaker_segments.append(
                    {"speaker": speaker_id, "start": start_sec, "end": end_sec}
                )
            except (ValueError, TypeError) as e:
                print(f"警告：跳过格式错误的分离片段: {segment}。错误: {e}")
    return speaker_segments


def merge_results_and_to_srt(asr_sentences, speaker_segments, output_file):
    """合并ASR结果和说话人分离结果，并生成SRT文件"""
    srt_content = ""
    subtitle_index = 1
    for sentence in asr_sentences:
        if "start" not in sentence or "end" not in sentence:
            continue

        sentence_start_sec = sentence["start"] / 1000.0
        sentence_end_sec = sentence["end"] / 1000.0
        found_speaker = "未知"
        best_overlap = 0

        # 寻找与当前句子时间重叠最长的说话人片段
        for seg in speaker_segments:
            overlap_start = max(sentence_start_sec, seg["start"])
            overlap_end = min(sentence_end_sec, seg["end"])
            overlap_duration = max(0, overlap_end - overlap_start)
            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                found_speaker = seg["speaker"]

        start_time = format_time(sentence["start"])
        end_time = format_time(sentence["end"])
        text = sentence.get("text", "").strip()

        srt_content += f"{subtitle_index}\n{start_time} --> {end_time}\n[{found_speaker}] {text}\n\n"
        subtitle_index += 1

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(srt_content)
    print(f"SRT字幕文件已保存至: {output_file}")


def format_time(milliseconds):
    """将毫秒数格式化为SRT时间戳格式 (HH:MM:SS,ms)"""
    seconds = milliseconds / 1000.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


try:
    data, sample_rate = soundfile.read(audio_file)
    if sample_rate != 16000:
        print(
            f"警告：音频采样率为 {sample_rate}Hz。为了获得最佳效果，建议使用16kHz采样率的音频。"
        )
except Exception as e:
    print(
        f"错误：无法读取音频文件 {audio_file}。请确保文件存在且格式正确。错误信息: {e}"
    )
    exit()

print("初始化说话人分离模型 (CAM++)...")
diarization_pipeline = pipeline(
    task=Tasks.speaker_diarization,
    model="iic/speech_campplus_speaker-diarization_common",
    # model_revision="v1.0.0",
)

print(f"开始处理音频文件: {audio_file}")
print("开始执行说话人分离...")
# 如果已知说话人数量，可以添加 oracle_num 参数以提高准确率
num_speakers = 2  # 根据实际情况调整或注释掉
diarization_result = diarization_pipeline(audio_file, oracle_num=num_speakers)
diarization_output = diarization_result["text"]
print("说话人分离完成。")
print(diarization_output)

print("初始化语音识别模型 (Paraformer)...")
asr_model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="fsmn-vad",
    punc_model="ct-punc-c",
    device="cuda",
)

print("开始执行语音识别...")
# 使用模型内置的VAD进行智能分句，直接获取句子列表
res = asr_model.generate(input=audio_file, sentence_timestamp=True)
print("语音识别完成。")
print(res)

# 解析结果
speaker_info = parse_diarization_result(diarization_output)
sentence_list = []
if res and "sentence_info" in res[0]:
    sentence_list = res[0]["sentence_info"]
else:
    print("警告：无法从ASR结果中获取'sentence_info'。")

# 生成SRT
output_srt_file = "output_with_speakers.srt"
merge_results_and_to_srt(sentence_list, speaker_info, output_srt_file)

print("\n--- 处理完成 ---")

if __name__ == "__main__":
    pass
