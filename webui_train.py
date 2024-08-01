import json
import os
import gradio as gr
import subprocess
from pathlib import Path


def preprocess(train_input_path, val_input_path, output_path, pre_model_path):
    for state, input_path in zip(['train', 'val'], [train_input_path, val_input_path]):
        temp1 = Path(output_path)/state/'temp1'
        temp2 = Path(output_path)/state/'temp2'
        try:
            temp1.mkdir(parents=True)
            temp2.mkdir(parents=True)
        except Exception as e:
            pass

        print(input_path)
        print(temp1)

        subprocess.run([r'.\py311\python.exe', 'local/prepare_data.py', 
                        '--src_dir', input_path, 
                        '--des_dir', str(temp1)])
        
        print("第一步结束")

        subprocess.run([r'.\py311\python.exe', 'tools/extract_embedding.py', 
                        '--dir', str(temp1), 
                        '--onnx_path', "pretrained_models/CosyVoice-300M/campplus.onnx"
                        ])
        
        print("第二步结束")

        subprocess.run([r'.\py311\python.exe', 'tools/extract_speech_token.py', 
                        '--dir', str(temp1), 
                        '--onnx_path', "pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx"
                        ])
        
        print("第三步结束")
        subprocess.run([r'.\py311\python.exe', 'tools/make_parquet_list.py', 
                        '--num_utts_per_parquet', '100',
                        '--num_processes', '1',
                        '--src_dir', str(temp1),
                        '--des_dir', str(temp2),
                        ])
        return f'{state} make parquet list done!'


def refresh_voice(output_path):
    content = (Path(output_path)/'train'/'temp1'/'utt2spk').read_text()
    voices = []
    for item in content.split('\n'):
        voices.append(item.split(' ')[0])
    return gr.Dropdown(choices=voices)

    
def train(output_path, pre_model_path):
    train_list = os.path.join(output_path, 'train', 'temp2', 'data.list')
    val_list = os.path.join(output_path, 'val', 'temp2', 'data.list')
    model_dir = Path(output_path)/'models'
    model_dir.mkdir(exist_ok=True, parents=True)

    cmd = rf".\py311\Scripts\torchrun.exe --nnodes 1 --nproc_per_node 1 --rdzv_id 1986 --rdzv_backend c10d --rdzv_endpoint localhost:0 cosyvoice/bin/train.py --train_engine torch_ddp --config conf/cosyvoice.yaml --model llm --checkpoint pretrained_models/CosyVoice-300M/llm.pt --ddp.dist_backend gloo --num_workers 1 --prefetch 100 --pin_memory --train_data {train_list} --cv_data {val_list} --model_dir {model_dir} --tensorboard_dir {model_dir}"

    print(cmd)

    res = subprocess.Popen(cmd)

    res.wait()

    # subprocess.run([r'.\py311\torchrun', '--nnodes', '1', '--nproc_per_node', '1', '--rdzv_id', '1986',
    #                 '--rdzv_backend', "c10d", '--rdzv_endpoint', "localhost:0", 
    #                 'cosyvoice/bin/train.py','--train_engine','torch_ddp','--config','conf/cosyvoice.yaml', 
    #                 '--model','llm', '--checkpoint', os.path.join(pre_model_path, 'llm.pt'), 
    #                 '--ddp.dist_backend', 'nccl', '--num_workers', '1', '--prefetch', '100','--pin_memory', 
    #                 '--deepspeed_config', './conf/ds_stage2.json', '--deepspeed.save_states', 'model+optimizer', 
    #                 '--train_data', train_list, '--cv_data', val_list, 
    #                 '--model_dir', str(model_dir), '--tensorboard_dir', str(model_dir),
    #                 ])
    return 'Train done!'



def inference(mode, output_path, epoch, pre_model_path, text, voice):
    train_list = os.path.join(output_path, 'train', 'temp2', 'data.list')
    utt2data_list = Path(train_list).with_name('utt2data.list')
    llm_model = os.path.join(output_path, 'models', f'epoch_{epoch}_whole.pt')
    flow_model = os.path.join(pre_model_path, 'flow.pt')
    hifigan_model = os.path.join(pre_model_path, 'hift.pt')

    res_dir = Path(output_path)/'outputs'
    res_dir.mkdir(exist_ok=True, parents=True)

    json_path = str(Path(res_dir)/'tts_text.json')
    with open(json_path, 'wt', encoding='utf-8') as f:
        json.dump({voice:[text]}, f)

    subprocess.run([r'.\pyr11\python.exe', 'cosyvoice/bin/inference.py', 
      '--mode', mode,
      '--gpu', '0', '--config', 'conf/cosyvoice.yaml',
      '--prompt_data', train_list, 
      '--prompt_utt2data', str(utt2data_list), 
      '--tts_text', json_path,
      '--llm_model', llm_model, 
      '--flow_model', flow_model,
      '--hifigan_model', hifigan_model, 
      '--result_dir', str(res_dir)])
    output_path = str(Path(res_dir)/f'{voice}_0.wav')
    return output_path
    

with gr.Blocks() as demo:
    pretrained_model_path = gr.Text('./pretrained_models/CosyVoice-300M', label='Pretrained model dir')
    output_dir = gr.Text(label='Output dir',value="./output")
    with gr.Tab('Train'):
        train_input_path = gr.Text(label='Train input path',value="./train")
        val_input_path = gr.Text(label='Val input path',value="./val")
        preprocess_btn = gr.Button('Preprocess', variant='primary')
        train_btn = gr.Button('Train', variant='primary')
        status = gr.Text(label='Status')
    with gr.Tab('Inference'):
        with gr.Row():
            voices = gr.Dropdown(label='Voices')
            refresh = gr.Button('Refresh voices', variant='primary')
            mode = gr.Dropdown(choices=['sft', 'zero_shot'], label='Mode')
            epoch = gr.Number(value=8, interactive=True, precision=0, label='Epochs')
        text = gr.Text()
        inference_btn = gr.Button('Inference', variant='primary')
        out_audio = gr.Audio()

    preprocess_btn.click(preprocess, inputs=[train_input_path, val_input_path, output_dir, pretrained_model_path], outputs=status)
    train_btn.click(train, inputs=[output_dir, pretrained_model_path], outputs=status)
    inference_btn.click(inference, inputs=[mode, output_dir, epoch, pretrained_model_path, text, voices], outputs=out_audio)
    refresh.click(refresh_voice, inputs=output_dir, outputs=voices)

demo.launch(server_name='0.0.0.0',server_port=9883,inbrowser=True)