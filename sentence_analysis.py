import whisper
from whisper.utils import get_writer
from konlpy.tag import Komoran, Kkma
import datetime
import pysubs2
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import xlsxwriter

title = 'police_30min'
base_dir = '/Users/1110165/duke/whisper'
wav_fname = title+'.wav'
srt_fname = wav_fname+'.srt'
csv_fname = title+'.csv'
xlsx_fname = title+'.xlsx'
xlsx_path = os.path.join(base_dir, xlsx_fname)

model_name = "medium" #"large-v2"

def timestamp():
    now = datetime.datetime.now()
    print(now.strftime('start %Y-%m-%d %H:%M:%S'))

transcription = ""

def func_whisper():
    global transcription

    model = whisper.load_model(model_name)

    options = dict(language="Korean", beam_size=5, best_of=5)
    transcribe_options = dict(task="transcribe", **options)

    timestamp()

    transcription = model.transcribe(wav_fname, **transcribe_options) #["text"]
    writer = get_writer("srt", "./")

    writer(transcription, wav_fname)
    print(transcription["text"])
    #func_konlpy(transcription)

    timestamp()
    return transcription["text"]

def func_komoran(transcription):
    komoran = Komoran()
    text = komoran.nouns(transcription)

    return text

def func_kkma(transcription):
    kkma = Kkma()
    text = kkma.nouns(transcription)

    return text

#text = func_whisper()
#print(len(text))

subs = pysubs2.load(srt_fname, encoding='utf-8')
text_list = []
index = 0
for line in subs:
    time_len = line.end - line.start

    text_list.append([])
    text_list[index].append(line.text)
    text_list[index].append(len(line.text))

    text_list[index].append(line.start/1000)
    text_list[index].append(line.end/1000)
    text_list[index].append(time_len/1000)

    index += 1

text_pd = pd.DataFrame(text_list, columns=['text', 'len', 'start', 'end', 'time'])
#text_pd.to_csv(csv_fname, index=False)

print(text_pd)
#Path(xlxs_path).touch()
text_pd.to_excel(xlsx_path,
                 sheet_name = 'Sheet1',
                 float_format = "%.3f",
                 header = True,
                 index = False,
                 engine = 'xlsxwriter'
                 )
#df = pd.read_csv ('mywav.csv'2t)
text_pd.plot(title='sentence length', x='end', y='len')
plt.show()
