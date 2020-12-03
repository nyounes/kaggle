import tqdm
import pandas as pd

# from googletrans import Translator
from translate import Translator
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()


train = pd.read_csv('datas/train.csv')


# def convert_text(text, src_language, inter_language, dest_language):
#     translator = Translator()
#
#     text_inter = translator.translate(text, src=src_language, dest=inter_language).text
#
#     text_final = translator.translate(text_inter, src=inter_language, dest=dest_language).text
#     print(text)
#     print(text_final)
#     return text_final


def convert_text(text, src_language, inter_language, dest_language):
    translator1 = Translator(from_lang=src_language, to_lang=inter_language)
    translator2 = Translator(from_lang='fr', to_lang='en')

    text_inter = translator1.translate(text)
    text_final = translator2.translate(text_inter)

    return text_final

from time import time
start_time = time()
train = train.head(1850000).tail(50000)
train['comment_text'] = train['comment_text'].apply(lambda x: convert_text(x, 'en', 'fr', 'en'))
end_time = time()
print(end_time - start_time)

train.to_csv('datas/train_extended_fr_37.csv', index=False)
