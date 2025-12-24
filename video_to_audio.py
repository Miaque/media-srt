import os
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger


def video_to_audio(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    audio_format: str = "wav",
    overwrite: bool = False,
) -> str:
    """
    将视频文件转换为适合语音识别的音频文件

    参数说明:
        video_path: 输入视频文件路径（支持mp4、avi、mkv等格式）
        output_path: 输出音频文件路径（可选，默认为视频同目录下的同名.wav文件）
        sample_rate: 采样率（Hz），默认16000（16kHz，语音识别标准）
        channels: 声道数，默认1（单声道，语音识别推荐）
        audio_format: 音频格式，默认'wav'（无损，最适合ASR）
        overwrite: 是否覆盖已存在的文件，默认False

    返回值:
        输出音频文件的路径

    参数说明:
        - 采样率16kHz: 大多数语音识别模型的标准输入
        - 单声道: 语音识别不需要立体声，单声道可以减少文件大小
        - WAV格式: 无损格式，保证最佳识别效果
        - PCM编码: 16位深度，平衡音质和文件大小
    """

    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 如果没有指定输出路径，自动生成
    if output_path is None:
        video_path_obj = Path(video_path)
        output_path = str(video_path_obj.with_suffix(f".{audio_format}"))

    # 检查输出文件是否已存在
    if os.path.exists(output_path) and not overwrite:
        logger.warning(f"输出文件已存在: {output_path}")
        user_input = input("是否覆盖？(y/n): ").strip().lower()
        if user_input != "y":
            logger.info("已取消转换")
            return output_path

    # 构建ffmpeg命令
    # 参数说明:
    # -i: 输入文件
    # -vn: 不处理视频流（只提取音频）
    # -acodec pcm_s16le: 使用PCM 16位小端编码（无损，最适合ASR）
    # -ar: 音频采样率
    # -ac: 声道数
    # -y: 覆盖输出文件（如果存在）

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",  # 不处理视频
        "-acodec",
        "pcm_s16le",  # PCM 16位编码（无损）
        "-ar",
        str(sample_rate),  # 采样率16kHz
        "-ac",
        str(channels),  # 单声道
    ]

    # 如果需要覆盖，添加-y参数
    if overwrite or os.path.exists(output_path):
        cmd.append("-y")

    cmd.append(output_path)

    logger.info(f"开始转换视频: {video_path}")
    logger.info(f"输出音频参数: 采样率={sample_rate}Hz, 声道={channels}, 格式={audio_format}")

    try:
        # 执行ffmpeg命令
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",  # 忽略编码错误
        )

        logger.success(f"转换完成！输出文件: {output_path}")

        # 显示文件大小
        file_size = os.path.getsize(output_path)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"文件大小: {file_size_mb:.2f} MB")

        return output_path

    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg转换失败")
        logger.error(f"错误信息: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("未找到ffmpeg，请确保已安装ffmpeg并添加到系统PATH")
        logger.info("安装方法:")
        logger.info("  Windows: 下载ffmpeg并添加到PATH，或使用: winget install ffmpeg")
        logger.info("  macOS: brew install ffmpeg")
        logger.info("  Linux: sudo apt install ffmpeg 或 sudo yum install ffmpeg")
        raise


def batch_convert(video_dir: str, output_dir: Optional[str] = None, **kwargs):
    """
    批量转换目录下的所有视频文件

    参数:
        video_dir: 视频文件所在目录
        output_dir: 输出目录（可选，默认为视频同目录）
        **kwargs: 传递给video_to_audio的其他参数
    """
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"}
    video_dir_path = Path(video_dir)

    if not video_dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {video_dir}")

    # 查找所有视频文件
    video_files = [f for f in video_dir_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]

    if not video_files:
        logger.warning(f"未在目录 {video_dir} 中找到视频文件")
        return

    logger.info(f"找到 {len(video_files)} 个视频文件")

    # 创建输出目录
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_path = video_dir_path

    # 批量转换
    success_count = 0
    for video_file in video_files:
        try:
            output_path = output_dir_path / f"{video_file.stem}.wav"
            video_to_audio(str(video_file), str(output_path), **kwargs)
            success_count += 1
        except Exception as e:
            logger.error(f"转换失败: {video_file.name}, 错误: {e}")

    logger.info(f"批量转换完成: 成功 {success_count}/{len(video_files)}")


if __name__ == "__main__":
    # 使用示例

    # 示例1: 单个视频转换
    video_file = "./input/1729694733.mp4"  # 替换为你的视频文件路径

    if os.path.exists(video_file):
        audio_output = video_to_audio(
            video_path=video_file,
            output_path="./output/1729694733.wav",  # 自动生成输出路径
            sample_rate=16000,  # 16kHz采样率（ASR标准）
            channels=1,  # 单声道
            audio_format="wav",  # WAV格式
            overwrite=True,  # 自动覆盖
        )
        logger.info(f"音频文件已保存: {audio_output}")
    else:
        logger.warning(f"示例视频文件不存在: {video_file}")
        logger.info("请修改代码中的video_file路径为实际的视频文件路径")

    # 示例2: 批量转换
    # video_directory = "videos"  # 视频文件夹路径
    # if os.path.exists(video_directory):
    #     batch_convert(
    #         video_dir=video_directory,
    #         output_dir="audio_output",  # 输出到单独的文件夹
    #         overwrite=True,
    #     )
