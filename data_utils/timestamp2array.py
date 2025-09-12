import pandas as pd
import numpy as np
import os # Import os module for path operations

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
        # 1. 使用pandas从指定的文本文件路径读取数据
        # sep='\s+' 表示使用一个或多个空白字符作为分隔符。
        # usecols 参数指定只加载这几列，'info' 列被完全忽略，解决了包含空格的问题。
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

        # 2. 确定当前音频文件的矩阵尺寸
        if group_df.empty:
            continue
            
        max_time = group_df['end_time'].max()
        num_frames = int(np.ceil(max_time * frame_rate))
        print(f"计算得出的总帧数: {num_frames}")
        
        if num_frames == 0:
            print(f"警告：文件 '{audio_name}' 的持续时间过短，无法生成帧。")
            all_audio_matrices[audio_name] = np.empty((2, 0), dtype=int)
            continue

        # 3. 初始化一个临时的布尔矩阵来标记语音活动
        # 0: Channel 0, 1: Channel 1
        is_speaking = np.zeros((2, num_frames), dtype=bool)

        # Process SPEECH events first
        speech_events = group_df[group_df['type'] == 'SPEECH'].copy()
        speech_events['start_frame'] = (speech_events['start_time'] * frame_rate).astype(int)
        speech_events['end_frame'] = (speech_events['end_time'] * frame_rate).astype(int).clip(upper=num_frames)

        for _, row in speech_events.iterrows():
            start_f = row['start_frame']
            end_f = row['end_frame']
            channel = row['channel_id']
            if 0 <= channel < 2 and start_f < num_frames: # Basic check to avoid index errors
                is_speaking[channel, start_f:end_f] = True

        # 4. 初始化最终结果矩阵，默认状态为 'gap' (2)
        # We start by assuming all non-speech are 'gap' and then refine 'pause'
        result_matrix = np.full((2, num_frames), STATE_MAP['gap'], dtype=int)

        # 5. 逐帧确定 speech, another_speech, and overlap states
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
            # If neither speaks, it remains 'gap' for now, to be refined later

        # 6. Refine 'pause' and 'gap' states
        # Identify silent regions (where both channels are not speaking)
        silent_frames = np.where(~is_speaking[0, :] & ~is_speaking[1, :])[0]

        if silent_frames.size > 0:
            # Group consecutive silent frames to identify silent segments
            # Find the start of each silent segment
            silent_segment_starts = silent_frames[np.insert(np.diff(silent_frames) != 1, 0, True)]
            # Find the end of each silent segment
            silent_segment_ends = silent_frames[np.append(np.diff(silent_frames) != 1, True)]

            for start_idx, end_idx in zip(silent_segment_starts, silent_segment_ends):
                # The silent segment is from start_idx to end_idx (inclusive)
                # We need to check frames *before* start_idx and *after* end_idx for speaker change

                # Determine the frame immediately before the silent segment
                prev_frame = start_idx - 1
                # Determine the frame immediately after the silent segment
                next_frame = end_idx + 1

                speaker_change = False

                # Check for speaker change before the silent segment
                if prev_frame >= 0:
                    # Get the speaker status just before silence
                    ch0_prev = is_speaking[0, prev_frame]
                    ch1_prev = is_speaking[1, prev_frame]
                else:
                    # If silence starts at the beginning, no prior speaker
                    ch0_prev, ch1_prev = False, False

                # Check for speaker change after the silent segment
                if next_frame < num_frames:
                    # Get the speaker status just after silence
                    ch0_next = is_speaking[0, next_frame]
                    ch1_next = is_speaking[1, next_frame]
                else:
                    # If silence ends at the end, no subsequent speaker
                    ch0_next, ch1_next = False, False

                # A speaker change occurs if:
                # - A channel was speaking before and isn't after (or vice-versa)
                # - One channel was speaking before, and a *different* channel is speaking after
                # We need to consider the active speaker (speech or another_speech) at prev_frame and next_frame
                
                # Simple check: if the active speaking channels are different
                # We'll use 0 for Ch0, 1 for Ch1, -1 for no one speaking
                active_speaker_prev = -1
                if ch0_prev and not ch1_prev: active_speaker_prev = 0
                elif not ch0_prev and ch1_prev: active_speaker_prev = 1

                active_speaker_next = -1
                if ch0_next and not ch1_next: active_speaker_next = 0
                elif not ch0_next and ch1_next: active_speaker_next = 1
                
                # Speaker change if active speaker changes or if one goes from speaking to non-speaking and vice-versa
                # and if both were not silent at the transitions
                if active_speaker_prev != active_speaker_next and \
                   (active_speaker_prev != -1 or active_speaker_next != -1):
                    speaker_change = True
                
                # Specifically handle cases where one channel was speaking before silence,
                # and the *other* channel speaks after silence.
                if (ch0_prev and ch1_next and not ch1_prev and not ch0_next) or \
                   (ch1_prev and ch0_next and not ch0_prev and not ch1_next):
                    speaker_change = True
                
                # If both were speaking (overlap) and then silence, and then a single speaker, consider a change.
                # If single speaker before, then silence, then another single speaker or overlap, consider a change.
                # The most robust way is to compare the *set* of active speakers.
                
                speakers_before = set()
                if ch0_prev: speakers_before.add(0)
                if ch1_prev: speakers_before.add(1)

                speakers_after = set()
                if ch0_next: speakers_after.add(0)
                if ch1_next: speakers_after.add(1)
                
                if speakers_before != speakers_after:
                    speaker_change = True

                # Apply the state based on speaker_change
                if speaker_change:
                    # This segment is a 'gap'
                    result_matrix[:, start_idx : end_idx + 1] = STATE_MAP['gap']
                else:
                    # This segment is a 'pause'
                    result_matrix[:, start_idx : end_idx + 1] = STATE_MAP['pause']
        
        all_audio_matrices[audio_name] = result_matrix
            
    return all_audio_matrices

# --- How to use ---

# 1. Assume your data is saved in a file named `timestamps_vad_refined.txt`.
file_to_process = 'timestamps_vad_refined_test.txt'

# Create a dummy file for testing if it doesn't exist
if not os.path.exists(file_to_process):
    print(f"Creating a dummy file: {file_to_process} for demonstration.")
    dummy_content = """audio_name start_time end_time channel_id type info
audio_file_01.wav 0.0 1.0 0 SPEECH A
audio_file_01.wav 1.5 2.5 1 SPEECH B
audio_file_01.wav 3.0 4.0 0 SPEECH C
audio_file_01.wav 4.0 5.0 1 SPEECH D
audio_file_01.wav 5.5 6.0 0 SPEECH E
audio_file_01.wav 6.5 7.0 1 SPEECH F
audio_file_01.wav 7.5 8.0 0 PAUSE G # This will be re-evaluated
audio_file_01.wav 8.5 9.0 1 PAUSE H # This will be re-evaluated
audio_file_01.wav 9.5 10.0 0 SPEECH I
audio_file_01.wav 10.5 11.0 1 SPEECH J
audio_file_02.wav 0.0 0.5 0 SPEECH X
audio_file_02.wav 0.5 1.0 1 SPEECH Y
audio_file_02.wav 1.0 1.5 0 PAUSE Z # This will be re-evaluated
audio_file_02.wav 1.5 2.0 1 PAUSE AA # This will be re-evaluated
audio_file_02.wav 2.0 2.5 0 SPEECH BB
"""
    with open(file_to_process, 'w') as f:
        f.write(dummy_content)
    print(f"Dummy file '{file_to_process}' created.")

# 2. Call the function to process the file
# state_matrices is now a dictionary where keys are audio filenames and values are the corresponding numpy arrays
state_matrices = convert_log_file_to_state_matrix(file_to_process)

# 3. Print results and save each file's matrix
if state_matrices:
    print(f"\n成功处理文件: '{file_to_process}'")
    print(f"共生成 {len(state_matrices)} 个对话状态矩阵。")

    # Ensure the 'matrix' directory exists
    output_dir = 'speech_activity_matrix_test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    for audio_name, matrix in state_matrices.items():
        print(f"\n--- 音频文件: {audio_name} ---")
        if matrix.size > 0:
            print("生成的对话状态矩阵 (部分预览):\n", matrix[:, :20]) # Print first 20 frames as preview
            print(f"矩阵形状 (2 x N): {matrix.shape}")

            # Save the result to a NumPy file (.npy)
            # Remove the .wav extension from the original filename (if present), then add .npy
            base_audio_name = audio_name.rsplit('.', 1)[0] if '.' in audio_name else audio_name
            output_filename = os.path.join(output_dir, f"output_matrix_{base_audio_name}.npy")
            np.save(output_filename, matrix)
            print(f"结果已保存到 {output_filename}")
        else:
            print("该音频文件未生成有效状态矩阵 (可能时间过短或无数据)。")
else:
    print(f"文件 '{file_to_process}' 未能生成任何对话状态矩阵。")