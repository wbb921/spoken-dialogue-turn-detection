import pandas as pd
import numpy as np
import os
import argparse # 导入 argparse 模块

def convert_log_file_to_state_matrix(filepath: str, frame_rate: float = 12.5) -> dict[str, np.ndarray]:
    """
    从一个日志文本文件中读取数据，并将其转换为一个字典，
    其中键是音频文件的名称，值是对应的 2xN 状态矩阵。
    此版本会忽略最后一列 'info' 以避免解析错误，并假设时间戳以秒为单位。
    根据前后说话人切换情况，动态判断 'pause' 和 'gap' 状态。

    Args:
        filepath (str): 日志文件的路径 (例如 'my_data.txt')。
        frame_rate (float): 矩阵的分辨率，单位为 Hz (每秒帧数)。

    Returns:
        dict[str, np.ndarray]: 一个字典，每个键是 audio_name，对应一个 2xN 的整数矩阵，
                                 表示该音频文件中每个通道在每一帧的状态。
    """
    # 状态码定义
    STATE_MAP = {
        'speech': 0,
        'another_speech': 1,
        'gap': 2,
        'pause': 3,
        'overlap': 4
    }
    
    # 定义我们要读取的列名
    columns_to_read = ['audio_name', 'start_time', 'end_time', 'channel_id', 'type']

    try:
        # 使用pandas从指定的文本文件路径读取数据
        df = pd.read_csv(filepath, sep='\s+', header=0, usecols=columns_to_read)
        
    except FileNotFoundError:
        print(f"错误：找不到文件 '{filepath}'。请检查文件路径是否正确。")
        return {}
    except Exception as e:
        print(f"读取或解析文件时发生错误: {e}")
        return {}

    if df.empty:
        print("警告：输入文件为空或未包含有效数据。")
        return {}

    # 用于存储所有音频文件的结果
    all_audio_matrices = {}

    # 遍历每个唯一的 audio_name
    for audio_name, group_df in df.groupby('audio_name'):
        print(f"\n正在处理音频文件: {audio_name}")

        # 确定当前音频文件的矩阵尺寸
        if group_df.empty:
            continue
            
        max_time = group_df['end_time'].max()
        num_frames = int(np.ceil(max_time * frame_rate))
        print(f"计算得出的总帧数: {num_frames}")
        
        if num_frames == 0:
            print(f"警告：文件 '{audio_name}' 的持续时间过短，无法生成帧。")
            all_audio_matrices[audio_name] = np.empty((2, 0), dtype=int)
            continue

        # 初始化一个临时的布尔矩阵来标记语音活动
        is_speaking = np.zeros((2, num_frames), dtype=bool)

        # 首先处理 SPEECH 事件
        speech_events = group_df[group_df['type'] == 'SPEECH'].copy()
        speech_events['start_frame'] = (speech_events['start_time'] * frame_rate).astype(int)
        speech_events['end_frame'] = (speech_events['end_time'] * frame_rate).astype(int).clip(upper=num_frames)

        for _, row in speech_events.iterrows():
            start_f, end_f = row['start_frame'], row['end_frame']
            channel = row['channel_id']
            if 0 <= channel < 2 and start_f < num_frames:
                is_speaking[channel, start_f:end_f] = True

        # 初始化最终结果矩阵，默认状态为 'gap' (2)
        result_matrix = np.full((2, num_frames), STATE_MAP['gap'], dtype=int)

        # 逐帧确定 speech, another_speech, and overlap 状态
        for frame in range(num_frames):
            ch0_speaks = is_speaking[0, frame]
            ch1_speaks = is_speaking[1, frame]

            if ch0_speaks and ch1_speaks:
                result_matrix[:, frame] = STATE_MAP['overlap']
            elif ch0_speaks:
                result_matrix[0, frame] = STATE_MAP['speech']
                result_matrix[1, frame] = STATE_MAP['another_speech']
            elif ch1_speaks:
                result_matrix[0, frame] = STATE_MAP['another_speech']
                result_matrix[1, frame] = STATE_MAP['speech']

        # 优化 'pause' 和 'gap' 状态
        silent_frames = np.where(~is_speaking[0, :] & ~is_speaking[1, :])[0]

        if silent_frames.size > 0:
            silent_segment_starts = silent_frames[np.insert(np.diff(silent_frames) != 1, 0, True)]
            silent_segment_ends = silent_frames[np.append(np.diff(silent_frames) != 1, True)]

            for start_idx, end_idx in zip(silent_segment_starts, silent_segment_ends):
                prev_frame = start_idx - 1
                next_frame = end_idx + 1

                speakers_before = set()
                if prev_frame >= 0:
                    if is_speaking[0, prev_frame]: speakers_before.add(0)
                    if is_speaking[1, prev_frame]: speakers_before.add(1)

                speakers_after = set()
                if next_frame < num_frames:
                    if is_speaking[0, next_frame]: speakers_after.add(0)
                    if is_speaking[1, next_frame]: speakers_after.add(1)
                
                # 如果说话人集合发生变化（包括从有人说到没人说），则为 'gap'
                # 否则（说话人集合未变，例如A说完话，停顿后还是A说话），则为 'pause'
                if speakers_before != speakers_after:
                    result_matrix[:, start_idx : end_idx + 1] = STATE_MAP['gap']
                else:
                    result_matrix[:, start_idx : end_idx + 1] = STATE_MAP['pause']
        
        all_audio_matrices[audio_name] = result_matrix
            
    return all_audio_matrices

def main():
    """
    主函数，用于解析命令行参数并驱动整个处理流程。
    """
    # 1. 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从日志文件转换生成对话状态矩阵，并保存为 NumPy 文件。"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="必需：待处理的日志文本文件路径。"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="speech_activity_matrix", # 提供一个默认值
        help="可选：用于保存输出 .npy 文件的目录。默认为 'speech_activity_matrix'。"
    )
    args = parser.parse_args()

    # 2. 调用核心函数处理文件
    state_matrices = convert_log_file_to_state_matrix(args.input_file)

    # 3. 打印结果并保存每个文件的矩阵
    if state_matrices:
        print(f"\n成功处理文件: '{args.input_file}'")
        print(f"共生成 {len(state_matrices)} 个对话状态矩阵。")

        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"结果将保存到目录: {args.output_dir}")

        for audio_name, matrix in state_matrices.items():
            print(f"\n--- 音频文件: {audio_name} ---")
            if matrix.size > 0:
                print("生成的对话状态矩阵 (部分预览):\n", matrix[:, :20])
                print(f"矩阵形状 (2 x N): {matrix.shape}")

                # 保存结果为 .npy 文件
                base_audio_name = audio_name.rsplit('.', 1)[0] if '.' in audio_name else audio_name
                output_filename = os.path.join(args.output_dir, f"{base_audio_name}.npy")
                np.save(output_filename, matrix)
                print(f"结果已保存到 {output_filename}")
            else:
                print("该音频文件未生成有效状态矩阵 (可能时间过短或无数据)。")
    else:
        print(f"文件 '{args.input_file}' 未能生成任何对话状态矩阵。")

if __name__ == "__main__":
    # 当脚本被直接执行时，调用 main 函数
    main()
