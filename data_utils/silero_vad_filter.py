import torch
import torchaudio
import argparse
import os
from tqdm import tqdm

def load_vad_model():
    """
    加载 Silero VAD 模型和相关工具函数。
    """
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False # 设置为True可强制重新下载
        )
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        return model, get_speech_timestamps
    except Exception as e:
        print("错误：无法加载 Silero VAD 模型。")
        print("请检查您的网络连接，或尝试设置 force_reload=True。")
        print(f"原始错误: {e}")
        exit(1)

def refine_timestamps_with_vad(input_file, audio_dir, output_file, min_pause_duration_ms):
    """
    使用 Silero VAD 对时间戳文件进行二次过滤，并识别内部的停顿。

    Args:
        input_file (str): 格式为 <audio name>\t<start>\t<end>\t<channel>\t<text> 的文件。
        audio_dir (str): 存放音频文件的目录。
        output_file (str): 输出结果的文件路径。
        min_pause_duration_ms (int): 识别为停顿的最小静音时长（毫秒）。
    """
    print("正在加载 Silero VAD 模型...")
    vad_model, get_speech_timestamps = load_vad_model()
    print("模型加载成功。")

    audio_cache = {}
    min_pause_samples = 0 # 将在获取采样率后设置

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # 写入表头，方便理解
        f_out.write("audio_name\tstart_time\tend_time\tchannel_id\ttype\tinfo\n")

        lines = f_in.readlines()
        for line in tqdm(lines, desc="处理音频片段"):
            try:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                audio_name, orig_start_str, orig_end_str, channel_id, text = parts
                orig_start_sec = float(orig_start_str)
                orig_end_sec = float(orig_end_str)

                audio_path = os.path.join(audio_dir, audio_name)

                # --- 音频加载与缓存 ---
                if audio_path not in audio_cache:
                    if not os.path.exists(audio_path):
                        print(f"\n警告：找不到音频文件 {audio_path}，跳过此行。")
                        continue
                    
                    waveform, sample_rate = torchaudio.load(audio_path)
                    audio_cache[audio_path] = (waveform, sample_rate)
                    
                    # 仅在第一次加载音频时设置采样率相关的阈值
                    if min_pause_samples == 0:
                        min_pause_samples = int((min_pause_duration_ms / 1000.0) * sample_rate)

                else:
                    waveform, sample_rate = audio_cache[audio_path]

                # --- 提取原始片段 ---
                start_sample = int(orig_start_sec * sample_rate)
                end_sample = int(orig_end_sec * sample_rate)
                
                # 确保切片不越界
                segment_waveform = waveform[int(channel_id):int(channel_id)+1, start_sample:end_sample]
                
                if segment_waveform.shape[1] == 0:
                    continue

                # --- 运行 VAD ---
                # 使用默认的VAD参数，对于大多数场景效果良好
                speech_timestamps = get_speech_timestamps(
                    segment_waveform,
                    vad_model,
                    sampling_rate=sample_rate,
                    min_speech_duration_ms=50,
                    min_silence_duration_ms=min_pause_duration_ms
                )

                if not speech_timestamps:
                    # 如果VAD未检测到任何语音，则将整个原始片段标记为PAUSE
                    duration_ms = (orig_end_sec - orig_start_sec) * 1000
                    f_out.write(f"{audio_name}\t{orig_start_sec:.3f}\t{orig_end_sec:.3f}\t{channel_id}\tPAUSE\t{duration_ms:.0f}ms_full_segment\n")
                    continue

                # --- 处理 VAD 结果并写入文件 ---
                last_speech_end_sample = 0
                for i, speech in enumerate(speech_timestamps):
                    # VAD返回的时间是相对于片段开始的采样点
                    relative_start_sample = speech['start']
                    relative_end_sample = speech['end']
                    
                    # 检查并写入上一个语音结束到当前语音开始之间的停顿
                    if relative_start_sample > last_speech_end_sample:
                        pause_duration_samples = relative_start_sample - last_speech_end_sample
                        if pause_duration_samples >= min_pause_samples:
                            pause_start_sec = orig_start_sec + (last_speech_end_sample / sample_rate)
                            pause_end_sec = orig_start_sec + (relative_start_sample / sample_rate)
                            pause_duration_ms = pause_duration_samples / sample_rate * 1000
                            f_out.write(f"{audio_name}\t{pause_start_sec:.3f}\t{pause_end_sec:.3f}\t{channel_id}\tPAUSE\t{pause_duration_ms:.0f}ms\n")

                    # 写入本次检测到的语音片段
                    speech_start_sec = orig_start_sec + (relative_start_sample / sample_rate)
                    speech_end_sec = orig_start_sec + (relative_end_sample / sample_rate)
                    
                    # 将原始文本与第一个语音片段关联
                    info = text if i == 0 else ""
                    f_out.write(f"{audio_name}\t{speech_start_sec:.3f}\t{speech_end_sec:.3f}\t{channel_id}\tSPEECH\t{info}\n")
                    
                    last_speech_end_sample = relative_end_sample

            except Exception as e:
                print(f"\n处理行 '{line.strip()}' 时发生错误: {e}")

    print("处理完成！")
    print(f"结果已保存至: '{os.path.abspath(output_file)}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="使用 Silero VAD 对时间戳进行二次过滤并识别停顿。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="输入的 `timestamps_output.txt` 文件路径。"
    )
    parser.add_argument(
        '--audio-dir', 
        type=str, 
        required=True,
        help="包含WAV或其他格式音频文件的目录路径。"
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='./timestamps_vad_refined.txt',
        help="输出文件的路径。\n(默认: ./timestamps_vad_refined.txt)"
    )
    parser.add_argument(
        '--min-pause-ms',
        type=int,
        default=100,
        help="被识别为有效停顿的最小静音时长（毫秒）。\n(默认: 200)"
    )
    args = parser.parse_args()

    refine_timestamps_with_vad(args.input, args.audio_dir, args.output, args.min_pause_ms)