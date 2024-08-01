import gradio as gr
import random
import os
import sys
import shutil
import argparse
import numpy as np
import torch
import torchaudio
import librosa
import ffmpeg
import logging
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import uuid
import scipy.io.wavfile

# 设置warning 日志,有效的排除无用的信息
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''

max_val = 0.8
ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
reference_audio_path = os.path.join(ROOT_DIR,'参考音频')
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
reference_audio_choice = [file for file in os.listdir("./参考音频/")]
instruct_dict = {'预训练角色': '1. 输入合成的文本\n2. 选择预训练角色\n3. 点击生成音频按钮',
                 '3s极速复刻': '1. 输入合成的文本\n2. 选择prompt音频文件，或录入音频，若同时提供，优先选择prompt音频文件\n3. 输入prompt文本\n4. 点击生成音频按钮',
                 '跨语言转换': '1. 输入合成的文本\n2. 选择prompt音频文件，或录入音频，若同时提供，优先选择prompt音频文件\n3. 输入prompt文本\n4. 点击生成音频按钮',
                 '语言控制': '1. 选择预训练角色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
inference_mode_list = ['预训练角色','3s极速复刻','跨语言转换','语言控制']
reference_audio_choice = [ file for file in os.listdir(reference_audio_path) ]

def random_generate():
    random_seed = random.randint(1,100000000)
    return {
        '__type__':"update",
        "value":random_seed
    }


def reference_audio():
    reference_audio_choice = [ file for file in os.listdir(reference_audio_path) ]
    return {
        "__type__":"update",
        "choices":reference_audio_choice
    }
    



def refresh_role_name():
    audio_options = cosyvoice.list_avaliable_spks()
    for name in os.listdir("./voices/"):
        audio_options.append(name.replace('.pt',''))
    return {"choices":audio_options, "__type__": "update"}

def save_new_role_function(name):
    if not name or name == "":

        gr.Info('角色名字不能为空')
        return False
    shutil.copyfile("./output.pt",f"./voices/{name}.pt")
    gr.Info("音色保存成功,存放位置为voices目录")

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

def change_reference_audio_dropdown_function(audio_path):
    text = audio_path.replace('.wav', '').replace('.WAV', '').replace('.mp3', '')
    return f"./参考音频/{audio_path}",text


def speed_change(input_audio: np.ndarray, speed: float, sr: int):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")


    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio

    
def generate_audio_function(tts_text,speech_rate,
                            inference_mode,role_select,seed,prompt_text,instruct_text,
                            load_audio,record_audio,new_role_name):
    if load_audio is not None:
        prompt_wav = load_audio
    elif record_audio is not None:
        prompt_wav = record_audio
    else:
        prompt_wav = None
    if inference_mode in ['语言控制']:
        if cosyvoice.frontend.instruct is False:
            gr.Warning('您正在使用语言控制模式, {}模型不支持此模式, 请使用pretrained_models/CosyVoice-300M-Instruct模型'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            return (target_sr, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')
    if inference_mode in ['跨语言转换']:
        if cosyvoice.frontend.instruct is True:
            gr.Warning('您正在使用跨语言转换模式, {}模型不支持此模式, 请使用pretrained_models/CosyVoice-300M模型'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用跨语言复刻模式, instruct文本会被忽略')
        if prompt_wav is None:
            gr.Warning('您正在使用跨语言复刻模式, 请提供prompt音频')
            return (target_sr, default_data)
        gr.Info('您正在使用跨语言转换模式, 请确保合成文本和prompt文本为不同语言')
    if inference_mode in ['3s极速复刻', '跨语言转换']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            return (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            return (target_sr, default_data)
    if inference_mode in ['预训练角色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练角色模式，prompt文本/prompt音频/instruct文本会被忽略！')
    if inference_mode in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练角色/instruct文本会被忽略！')
    if inference_mode == '预训练角色':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        output = cosyvoice.inference_sft(tts_text,role_select,new_role_name)
    elif inference_mode == '3s极速复刻':
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    elif inference_mode == '跨语言转换':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        output = cosyvoice.inference_instruct(tts_text, role_select, instruct_text,new_role_name)
    if speech_rate != 1.0:
        try:
            numpy_array = output['tts_speech'].numpy()
            audio = (numpy_array * 32768).astype(np.int16) 
            audio_data = speed_change(audio, speed=speech_rate, sr=int(target_sr))
        except Exception as e:
            print(f"Failed to change speed of audio: \n{e}")
    else:
        audio_data = output['tts_speech'].numpy().flatten()


    return (target_sr, audio_data)
    

def inferce_mode_function(role_selct):
    return instruct_dict[role_selct]

def save_audio_recording(audio_path):
    sample_rate, data = scipy.io.wavfile.read(audio_path)
    
    save_dir = '音频文件'
    os.makedirs(save_dir, exist_ok=True)
    
    unique_filename = str(uuid.uuid4()) + '.wav'
    save_path = os.path.join(save_dir, unique_filename)
    
    # 保存音频文件
    scipy.io.wavfile.write(save_path, sample_rate, np.array(data))
    print(f"音频文件已保存到: {save_path}")

def main():
    global cosyvoice
    with gr.Blocks() as demo:
        gr.Markdown('#### 请按照下面的教程进行操作')
        with gr.Column():
            input_synthetic_text = gr.Textbox(label = '请输入需要合成的文本',
                    value = '这是,目前效果最好的,克隆语音模型'
                                        )
            speech_rate_adjust = gr.Slider(minimum=0.2,maximum=5,value =1.0,label = '语速调节'
                                           ,step =0.05,interactive=True)

        with gr.Row():
            inferce_mode_select = gr.Radio(choices=inference_mode_list,label = '选择模式',value =inference_mode_list[0])
            tutorial_text = gr.Text(label = '教程',value=instruct_dict[inference_mode_list[0]], scale=1)
            audio_options = cosyvoice.list_avaliable_spks()
            role_selection =  gr.Dropdown(choices=audio_options,label ='选择预训练角色',value=audio_options[0], scale=1,interactive=True)
            refresh_role_button = gr.Button('角色刷新')
            refresh_role_button.click(fn=refresh_role_name,outputs=[role_selection])
            with gr.Column():
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(label = '随机推理种子',value=0)
                seed_button.click(fn = random_generate,outputs=[seed])
        with gr.Row():
            reference_audio_dropdown = gr.Dropdown(label ="参考音频",choices =reference_audio_choice,interactive=True)
            refresh_audio_button = gr.Button('音频刷新')
            refresh_audio_button.click(fn=reference_audio, outputs=reference_audio_dropdown)
            load_audio = gr.Audio(type='filepath',label='prompt 音频')
            inter_record_audio = gr.Audio(sources='microphone',label ='录制音频', type="filepath")
            inter_record_audio.change(
                fn = save_audio_recording,
                inputs = [inter_record_audio]
            )
        with gr.Column():
            input_audio_text =  gr.Textbox(label = '输入prompt文本',placeholder=
                    '请输入音频中的文本，输入的文字要与音频中的文本保持一致')
            input_instruct_text = gr.Textbox(label = '输入instruct文本',placeholder=
                    '例子：在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。')
            new_role_name = gr.Textbox(label="音色名称", lines=1, placeholder="输入新的音色名称.", value='')
            save_new_role_button = gr.Button("保存新角色")
            save_new_role_button.click(fn = save_new_role_function,inputs=new_role_name)
            generate_audio_button = gr.Button('生成音频')
            audio_output = gr.Audio(label ='音频输出',value = None,interactive=False,autoplay=True,show_label=True,show_download_button=True)
            generate_audio_button.click(fn = generate_audio_function,
                                        inputs = [input_synthetic_text,speech_rate_adjust,inferce_mode_select,role_selection,seed,input_audio_text,input_instruct_text,load_audio,inter_record_audio,new_role_name],
                                        outputs=[audio_output]
                                        )
        inferce_mode_select.change(
            fn = inferce_mode_function,
            inputs = [inferce_mode_select],
            outputs=[tutorial_text]

        )
        reference_audio_dropdown.change(
                fn = change_reference_audio_dropdown_function,
                inputs = [reference_audio_dropdown],
                outputs=[load_audio,input_audio_text]
            )
    demo.queue(max_size=6,default_concurrency_limit=2)
    demo.launch(server_port=args.port,inbrowser=True,auth = ('admin','12234'))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type = int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type = str,
                        default='pretrained_models/CosyVoice-300M',
                        help = 'local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()
    



            
