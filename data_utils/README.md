# Data Preparation

There are three steps for processing the SpokenWOZ dataset:

1. Process data.json in the SpokenWOZ dataset, generate utterance timestamps in txt format:
   ```bash
   python process.py --json_path /path/to/your/json --output_path ./timestamps_raw.txt
   ```
   The output txt file should contain content like this:
   ```bash
   MUL0001.wav	2.570	3.570	0	hello .
   MUL0001.wav	3.780	5.440	1	hello , how can i help .
   MUL0001.wav	6.550	9.010	0	yes , i'm looking for restaurant .
   MUL0001.wav	9.750	11.830	1	okay , any requirement .
   ...
   ```
   the columns refer to audio_file start_time end_time channel text

2. Use silero_vad to refine the timestamps:
   ```bash
   python silero_vad_filter.py --input ./timestamps_raw.txt --audio_dir /path/to/your/audio_dir --output ./timestamps_vad_refined.txt 
   ```
   the output should contain:
   ```bash
    audio_name	start_time	end_time	channel_id	type	info
    MUL0001.wav	2.570	2.924	0	PAUSE	354ms
    MUL0001.wav	2.924	3.336	0	SPEECH	hello .
    MUL0001.wav	3.780	4.102	1	PAUSE	322ms
    MUL0001.wav	4.102	5.282	1	SPEECH	hello , how can i help .
    MUL0001.wav	6.550	6.872	0	PAUSE	322ms
    MUL0001.wav	6.872	8.756	0	SPEECH	yes , i'm looking for restaurant .
    MUL0001.wav	9.750	10.072	1	PAUSE	322ms
    MUL0001.wav	10.072	11.604	1	SPEECH	okay , any requirement .
    MUL0001.wav	11.780	12.102	0	PAUSE	322ms
    MUL0001.wav	12.102	16.194	0	SPEECH	yes , i wanted to be in the west area serving international .
   ```

3. Convert the txt timestamp file into numpy array:
   ```bash
   python convert_log.py --input_file ./timestamps_vad_refined.txt  --output_dir my_results
   ```
   the generated numpy arrays will be in the output directory, with the same name as the audio files
   
