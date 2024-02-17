from openai import OpenAI
import os
import shutil
import torch
import whisper
import gradio as gr
import pytube
from pytube import YouTube
import subprocess
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import fitz
import random

# 使用Colab環境變亮獲取API Key
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

voctemp_dir = 'temp_files'
Path(voctemp_dir).mkdir(parents=True, exist_ok=True)

#AUDIO-簡化版YOUTUBE連結
def download_youtube_audio(youtube_link, current_time):
    try:
        yt = YouTube(youtube_link) # use_oauth=True, allow_oauth_cache=True
        video = yt.streams.filter(only_audio=True).first()

        video_title = yt.title[:20]  #取YT名稱
        safe_title = "".join([c for c in video_title if c.isalpha() or c.isdigit() or c in (' ','_')]).rstrip()  # 安全文件名
        # 组合文件名
        filename = f"{current_time}_YT_{safe_title}.mp3"
        download_path = video.download(output_path=voctemp_dir, filename=filename)

        return os.path.join(voctemp_dir, filename), None
    except Exception as e:
        error_message = f"下載YouTube錯誤，可能是版權或年齡限制: {str(e)}"
        return None, error_message

# AUDIO--抓取網頁
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text, None
    except requests.exceptions.RequestException as e:
        return None, "提取網頁文本出錯: {str(e)}"

# Text --抓取PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

#AUDIO- REC錄音+SAVE
def save_uploaded_audio(uploaded_audio, current_time):
    audio_save_path = os.path.join(voctemp_dir, f"{current_time}_original.mp3")
    # Gradio 1.8.0及以上版本将上传的文件作为临时文件路径提供
    if isinstance(uploaded_audio, str):  # 如果uploaded_audio是文件路径的字符串
        shutil.copy(uploaded_audio, audio_save_path)
    else: # 如果uploaded_audio是文件对象，尝试读取内容（不推荐，仅为兼容旧版本）
        with open(audio_save_path, "wb") as file_out:
            file_out.write(uploaded_audio.read())

    return audio_save_path

# AUDIO->TEXT: 使用Whisper進行語音轉文字的函數，自動檢查並使用GPU
def transcribe_audio(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium" if device == "cuda" else "small", device=device)
    result = model.transcribe(audio_path)
    return result["text"]

# 使用OpenAI GPT進行摘要
def summarize_text(text, language, current_time, user_prompt):
    original_text_filename = os.path.join(voctemp_dir, f"{current_time}_original.txt")
    summary_filename = os.path.join(voctemp_dir, f"{current_time}_summary.txt")

    # 原始文本保存到文件
    with open(original_text_filename, "w", encoding="utf-8") as file:
        file.write(text)

    prompt=f'''
     {user_prompt} in "{language}" from this text "{text}". '''
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=messages,
        temperature=0.1,
    )

    summary = response.choices[0].message.content

    # 在摘要添加额外的文字
    rec_year=  datetime.now().strftime("%Y")
    rec_month=  datetime.now().strftime("%m")
    rec_day=  datetime.now().strftime("%d")
    full_summary = summary + "這是西元"+ rec_year+"年"+rec_month+"月"+ rec_day+"日"+"由Hugo Chen工作室的語言轉換摘要AI工具，可以生成短語音朗讀報告幫助視障人士，如果覺得好用或任何建議歡迎寄email給hugocc@gmail.com"

    # 儲存摘要
    with open(summary_filename, "w", encoding="utf-8") as file:
        file.write(full_summary)

    return full_summary

# OUTPUT- 使用OpenAI的TTS语音模型
def text_to_speech(text, current_time, language):
    filename = os.path.join(voctemp_dir, f"{current_time}_otts.mp3")
    # 使用OpenAI TTS API生成语音
    voice = random.choice(["onyx", "alloy", "nova"])

    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    speech_file_path = Path(filename)
    response.stream_to_file(speech_file_path)

    return str(speech_file_path)


# Gradio流程與新版v0.5頁面
import gradio as gr
from datetime import datetime

def process_input(input_link, uploaded_audio, uploaded_file, text_input, record_audio, language, prompt_options): #Gradio邏輯
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    wav_path, summary = None, ""
    error_message = None

    if input_link:  # 處理YouTube連結 >> 文字網頁連結
        if "youtube.com" in input_link or "youtu.be" in input_link:
            audio_path, error_message = download_youtube_audio(input_link, current_time)
            if error_message:
                return None, error_message
            transcribed_text = transcribe_audio(audio_path)
        else: #處理文字網頁
            try:
                transcribed_text, error_message = extract_text_from_url(input_link)
                if error_message:
                    return None, error_message
            except Exception as e:
                return None, str(e)
    elif uploaded_audio:  # 處理上傳音檔
        audio_save_path = save_uploaded_audio(uploaded_audio, current_time)
        transcribed_text = transcribe_audio(uploaded_audio)
    elif record_audio:  # 處理錄REC音檔
        audio_save_path = save_uploaded_audio(record_audio, current_time)
        transcribed_text = transcribe_audio(audio_save_path)
    elif uploaded_file:  # 處理上傳的文檔PDF/TXT
        if uploaded_file.name.endswith('.pdf'):
            transcribed_text = extract_text_from_pdf(uploaded_file)
        else:
            transcribed_text = uploaded_file.read().decode("utf-8")
    elif text_input:  # 處理直接輸入的文本
        transcribed_text = text_input
    else:
        return None, "未提供有效輸入。"

    #用戶選擇prompt
    if prompt_options == "Bullets-Journalist":  # 列點摘要
        user_prompt= "According to the context of podcast or article, Provide me point wise and comprehensive summary with proper title"
    elif prompt_options == "Speech-Lecturer":  # 大學老師
        user_prompt= "You are a scholar in university. According to the context of podcast or article, Provide me comprehensive and  academic report with proper title. 專有名詞使用原文或英文. "
    elif prompt_options == "Bilingual-Teacher":  # 中學老師雙語
        user_prompt= "You are a high school teacher. Based on the context of the provided article, provide a bilingual point wise summary with a suitable title. For each point made in the summary, present it first in its original language followed by its translation in "
    else:
        return None, "請選擇。"

    if transcribed_text: #文本 >> 摘要 >> TTS
        summary = summarize_text(transcribed_text, language, current_time, user_prompt)
        wav_path = text_to_speech(summary, current_time, language=language)

    if wav_path and summary:
        return wav_path, summary
    else:
        return None, error_message or "處理時發生錯誤，請檢查輸入。"

with gr.Blocks(gr.themes.Base()) as demo: #gr.themes.Soft()
    gr.Markdown("# 語言轉換-摘要工具 Voice Translation + Summary Tool - ver 0.6")

    language = gr.Radio(["zh-TW", "en-US", "ja-JP", "de-DE", "fr-FR"], label="選擇發聲語言: 台灣中文, 美式英文, 日文, 德文, 法文", value="zh-TW")

    with gr.Tab("YouTube連結"):
        yt_link = gr.Textbox(label="輸入YouTube連結", lines =3)
    with gr.Tab("文字網頁連結"):
        web_link = gr.Textbox(label="輸入文字網頁連結", lines=3)
    with gr.Tab("上傳音檔"):
        uploaded_audio = gr.File(label="上傳音訊檔案 (MP3/WAV)")
    with gr.Tab("上傳文字檔"):
        uploaded_doc = gr.File(label="上傳文字檔案 (PDF/txt)")
    with gr.Tab("打字"):
        text_input = gr.Textbox(label="直接輸入發音文字", lines=8)
    with gr.Tab("錄音"):
        record_audio = gr.Audio(
                                  format="mp3", type="filepath", sources=["microphone"],
                                  show_download_button=True,
                                  waveform_options=gr.WaveformOptions(
                                  waveform_color="#01C6FF",
                                  waveform_progress_color="#0066B4",
                                  skip_length=2,
                                  show_controls=False
                                  ))

    prompt_options = gr.Radio(
        ["Bullets-Journalist", "Speech-Lecturer", "Bilingual-Teacher"], value="Bullets-Journalist",
        label="輸出Output", info="選擇你想要的摘要方法")
    submit_btn = gr.Button("Submit")
    output_audio = gr.Audio(label="播放摘要語音")
    output_summary = gr.Markdown(label="顯示文字摘要")
    clr_btn = gr.ClearButton()

    def aggregate_inputs(yt_link, web_link, uploaded_audio, uploaded_doc, text_input, record_audio, language, prompt_options):
        audio_path = record_audio if record_audio else None
        if yt_link:
            return process_input(yt_link, None, None, None, None, language, prompt_options)
        elif web_link:
            return process_input(web_link, None, None, None, None, language, prompt_options)
        elif uploaded_audio:
            return process_input(None, uploaded_audio, None, None, None, language, prompt_options)
        elif uploaded_doc:
            return process_input(None, None, uploaded_doc, None, None, language, prompt_options)
        elif text_input:
            return process_input(None, None, None, text_input, None, language, prompt_options)
        elif record_audio:
            return process_input(None, None, None, None, audio_path, language, prompt_options)
        else:
            return None, "請至少在一個標籤頁中提供輸入(或是輸入無法被處理)。"

    submit_btn.click(fn=aggregate_inputs,
                     inputs=[yt_link, web_link, uploaded_audio, uploaded_doc, text_input, record_audio, language, prompt_options],
                     outputs=[output_audio, output_summary])

    clr_btn.click(inputs=[yt_link, web_link, uploaded_audio, uploaded_doc, text_input, record_audio],
                  outputs=[output_audio, output_summary])

demo.launch(share=True, debug=True)