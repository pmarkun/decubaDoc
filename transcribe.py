import os
import pandas as pd
from datetime import timedelta
from pyannote.audio import Pipeline
import whisper_timestamped as whisper
from tqdm import tqdm
from dotenv import load_dotenv
import ffmpeg
import tempfile
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook


load_dotenv()

# Importa HF_TOKEN do .env
HF_TOKEN = os.getenv("HF_TOKEN")

import ffmpeg
import tempfile
import os

model = whisper.load_model("base")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

def extract_audio_stream(mov_path, temp=False):
    if temp:
        # Cria um arquivo temporário com a extensão .wav
        audio_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = audio_output.name
    else:
        # Define o caminho de saída no mesmo diretório do .mov, substituindo .mov por .wav
        output_path = mov_path.replace(".mov", ".wav")

    print(f"Extraindo áudio de {mov_path} para {'arquivo temporário' if temp else output_path}...")
    try:
        # Extrai o áudio sem recodificação
        (
            ffmpeg
            .input(mov_path)
            .output(output_path, acodec="pcm_s16le", ar="16000")  # Converte para WAV com PCM 16-bit e 44.1kHz
            .overwrite_output()
            .run()
        )
        print(f"Áudio extraído com sucesso: {output_path}")
    except ffmpeg.Error as e:
        print("Erro ao extrair áudio:", e)
        if temp:
            audio_output.close()
            os.unlink(output_path)  # Remove o arquivo temporário em caso de erro
        return None

    # Retorna o caminho do arquivo de áudio (temporário ou permanente)
    return output_path


def transcribe_and_diarize_with_vad(audio_path):
    print("Carregando modelo Whisper com timestamp e VAD...")
    # Carregar o modelo Whisper

    # Realizar a transcrição com VAD habilitado
    print("Iniciando a transcrição com VAD habilitado...")
    result = whisper.transcribe(model, audio_path, vad=True, verbose=True)

    # Salvar a transcrição completa em um arquivo de texto
    transcript_txt_path = audio_path.replace('.wav', '_transcript.txt')
    with open(transcript_txt_path, 'w') as f:
        f.write(result['text'])
    print(f"Transcrição completa salva em: {transcript_txt_path}")

    # Configurar o pipeline de diarização do pyannote
    print("Carregando o modelo de diarização...")
    

    # Realizar a diarização
    
    with ProgressHook() as hook:
        diarization = pipeline(audio_path, hook=hook)

    print("Processando segmentos com mudança de falante e VAD...")


    # Preparar para salvar em intervalos baseados na mudança de falante
    intervals = []
    current_speaker = None
    interval_text = ""
    current_start = 0

    # Adicionar barra de progresso para o loop de segmentos de palavras
    for segment in tqdm(result['segments'], desc="Processando transcrição por palavra"):
        for word in segment['words']:
            # Só processar palavras com alta confiança
            if word['confidence'] < 0.6:
                continue

            word_start_ms = int(word['start'] * 1000)
            word_text = word['text']

            # Identificar o falante atual usando os intervalos de diarização
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= word['start'] <= turn.end:
                    # Quando o falante muda, salva o intervalo
                    if speaker != current_speaker:
                        if interval_text:
                            intervals.append((current_start, word_start_ms, current_speaker, interval_text.strip()))
                        current_start = word_start_ms
                        interval_text = f"{word_text}"
                        current_speaker = speaker
                    else:
                        interval_text += f" {word_text}"

    # Adicionar o último intervalo
    if interval_text:
        intervals.append((current_start, word_start_ms, current_speaker, interval_text.strip()))

    print("Convertendo intervalos de milissegundos para o formato HH:MM:SS e salvando no CSV...")

    # Converter milissegundos para formato HH:MM:SS
    intervals_formatted = [
        (
            str(timedelta(milliseconds=start_ms)),
            str(timedelta(milliseconds=end_ms)),
            speaker,
            text
        ) for start_ms, end_ms, speaker, text in intervals
    ]

    # Salvar os intervalos em um arquivo CSV
    csv_path = audio_path.replace('.wav', '_transcript_intervals.csv')
    df = pd.DataFrame(intervals_formatted, columns=['Start (HH:MM:SS)', 'End (HH:MM:SS)', 'Speaker', 'Transcription'])
    df.to_csv(csv_path, index=False)

    print(f"Transcrição segmentada salva em CSV em: {csv_path}")

    return transcript_txt_path, csv_path
