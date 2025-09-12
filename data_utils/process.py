import json
import os
import argparse
from tqdm import tqdm

def process_data(json_path, output_path):
    """
    处理 SpokenWoz 或类似结构的数据集，提取每句话的起止时间、通道和内容。
    
    该脚本智能地处理两种时间戳格式：
    1. 优先使用 'words' 列表中的词级别时间戳（毫秒）。
    2. 如果 'words' 列表不存在，则回退使用 'span_info' 中的句子级别时间戳（秒）。

    通道/角色将被映射为数字：USER -> 0, SYSTEM -> 1。

    Args:
        json_path (str): 输入的 data.json 文件路径。
        output_path (str): 输出文件的路径。
    """
    print(f"正在从 '{json_path}' 读取数据...")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{json_path}'。请检查路径是否正确。")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 '{json_path}' 不是一个有效的 JSON 文件。")
        return

    print(f"数据读取成功，共包含 {len(data)} 个对话。")
    print(f"正在处理数据并写入到 '{output_path}'...")

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for dialogue_id, dialogue_content in tqdm(data.items(), desc="处理对话中"):
            audio_name = f"{dialogue_id}.wav"
            turns = dialogue_content.get('log', [])

            for turn in turns:
                text = turn.get('text', '').strip()
                
                # --- 主要修改点在这里 ---
                # 1. 获取原始的角色字符串 (USER/SYSTEM)
                role_str = turn.get('tag', 'UNKNOWN').upper()
                
                # 2. 将角色字符串映射到数字ID
                if role_str == 'USER':
                    channel_id = 0
                elif role_str == 'SYSTEM':
                    channel_id = 1
                else:
                    channel_id = -1 # 为未知角色设置一个默认值

                if not text:
                    continue

                start_time_sec = None
                end_time_sec = None

                words_list = turn.get('words', [])
                span_info_list = turn.get('span_info', [])

                if words_list:
                    try:
                        start_time_ms = float(words_list[0]['BeginTime'])
                        end_time_ms = float(words_list[-1]['EndTime'])
                        start_time_sec = start_time_ms / 1000.0
                        end_time_sec = end_time_ms / 1000.0
                    except (KeyError, IndexError, ValueError) as e:
                        print(f"警告：处理 'words' 列表时出错: {e}。对话ID: {dialogue_id}")
                        start_time_sec = None
                        end_time_sec = None
                
                elif span_info_list:
                    min_start = float('inf')
                    max_end = float('-inf')
                    try:
                        for span in span_info_list:
                            min_start = min(min_start, float(span[4]))
                            max_end = max(max_end, float(span[5]))
                        
                        if min_start != float('inf') and max_end != float('-inf'):
                            start_time_sec = min_start
                            end_time_sec = max_end
                    except (IndexError, ValueError) as e:
                        print(f"警告：处理 'span_info' 列表时出错: {e}。对话ID: {dialogue_id}")
                        pass

                if start_time_sec is not None and end_time_sec is not None:
                    # 3. 在输出时使用映射后的 channel_id
                    output_line = f"{audio_name}\t{start_time_sec:.3f}\t{end_time_sec:.3f}\t{channel_id}\t{text}\n"
                    out_f.write(output_line)

    print("处理完成！")
    print(f"输出文件已保存至: '{os.path.abspath(output_path)}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="处理 SpokenWoz 或类似数据集，智能提取句子级别的时间戳。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--json_path', 
        type=str, 
        required=True,
        help='输入的 data.json 文件路径。\n例如: /data/luhaitian/datasets/spokenwoz/text_5700_test/data.json'
    )
    
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='./timestamps_test.txt',
        help='输出文件的路径。'
    )

    args = parser.parse_args()
    process_data(args.json_path, args.output_path)