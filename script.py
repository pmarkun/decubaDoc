import os
import ffmpeg
from transcribe import extract_audio_stream, transcribe_and_diarize_with_vad

def find_mov_files(target_dir):
    mov_files = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                if "._" not in file:
                    mov_files.append(os.path.join(root, file))
    return mov_files

def process_file(mov_path):
        if mov_path.endswith(".mp3"):
            formato = ".mp3"
        elif mov_path.endswith(".mov"):
            formato = ".mov"
        # Define o caminho do arquivo _transcript.txt
        transcript_path = mov_path.replace(formato, "_transcript.txt")
        
        # Verifica se o arquivo de transcrição já existe
        if os.path.exists(transcript_path):
            print(f"Transcrição já existe para: {mov_path}, pulando...")
            return None
        
        # Caso não exista, extrai o áudio e realiza a transcrição
        print(f"Processando o arquivo: {mov_path}")
        
        # Extrai o áudio em formato WAV para transcrição
        if formato != ".mp3":
            audio_path = extract_audio_stream(mov_path)
        else:
            audio_path = mov_path
        
        if audio_path:
            # Transcreve e diariza o áudio extraído
            
            transcript_txt_path, csv_path = transcribe_and_diarize_with_vad(audio_path)
            print(f"Transcrição concluída para: {mov_path}")
            
            # Remove o arquivo temporário de áudio
            
            if formato != ".mp3":
                os.unlink(audio_path)
        else:
            print(f"Falha na extração de áudio para: {mov_path}")

# Defina o diretório de destino (substitua TARGET_DIR pelo seu caminho)
TARGET_DIR = "/media/markun/MEDIAARTS21/LEX_PEDRO"
mov_files_list = find_mov_files(TARGET_DIR)

# Processa os arquivos encontrados
for mov_files_list in mov_files_list:
    try:
        process_file(mov_files_list)
    except Exception as e:
        print(f"Erro ao processar {mov_files_list}:", e)
        continue
