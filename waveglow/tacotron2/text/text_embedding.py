import os
import csv
import re


def split_text(text, max_word):
    sens_out = []
    sen_out = ''
    for sen in text.split('.'):
        sen = sen.strip()
        if sen:
            sen = sen + ' . '
            if max_word > len(sen.split()):
                if len(sen_out.split()) < max_word - len(sen.split()):
                    sen_out += sen
                else:
                    sens_out.append(sen_out[:-1])
                    sen_out = sen
            else:
                sens_out.append(sen_out[:-1])
                sen_out = ''
                sens_out.append(sen[:-1])
    sens_out.append(sen_out[:-1])
    return sens_out

def split_long_sentence(text, max_words):
    result = []
    for sub_sen in text.strip().split(','):
        sub_sen = sub_sen.strip()
        tokens = []
        for word in sub_sen.split():
            tokens.append(word)
            if len(tokens) % max_words == 0:
                tokens.append(",")
        result.append(' '.join(tokens))
    text = ','.join(result)
    result = []
    sen = ""
    for sub_sen in text.strip().split(','):
        sub_sen = sub_sen.strip()
        if len((sen + " " + sub_sen).split()) > max_words:
            result.append(sen)
            sen = ""
        if len(sen) > 0:
            sen += " , "
        sen += sub_sen
    if len(sen) > 0:
        result.append(sen)
    return result

def word2phone(phone_dict_path, coda_nucleus_and_semivowel=['iz', 'pc', 'nz', 'tc', 'ngz', 'kc', 'uz', 'mz', 'aa', 'ee', 'ea', 'oa',
                                                            'aw', 'ie', 'uo', 'a', 'wa', 'oo', 'e', 'i', 'o', 'u', 'ow', 'uw', 'w']):
    word2phone_dict = {}
    if os.path.isfile(phone_dict_path):
        with open(phone_dict_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line and line.strip() and not line.startswith("#"):
                    items = line.strip().split()
                    syllable = items[0]
                    if len(items) > 1:
                        curr_tone = items[1]
                        phonemes = items[2:]
                        result = []
                        for phoneme in phonemes:
                            if phoneme in coda_nucleus_and_semivowel:
                                result.append('@' + phoneme + curr_tone)
                            elif phoneme.isdigit():
                                curr_tone = phoneme
                            else:
                                result.append('@' + phoneme)
                        word2phone_dict[syllable] = result
                    else:
                        word2phone_dict[syllable] = {}
    return word2phone_dict

def word2phone_2(phone_vn_path, phone_oov_path, coda_nucleus_and_semivowel):
    vn2phone_dict = word2phone(phone_vn_path, coda_nucleus_and_semivowel)
    oov2phone_dict = word2phone(phone_oov_path, coda_nucleus_and_semivowel)
    word2phone_dict = {**vn2phone_dict, **oov2phone_dict}
    return word2phone_dict

def phone2numeric(word2phone_dict):
    phone_lst = sorted(set([_phoneme for word in word2phone_dict.values() for _phoneme in word]))
    phone2numeric_dict = {phoneme: i for i, phoneme in enumerate(phone_lst)}
    return phone2numeric_dict

def oov2syllables(oov2syllables_path):
    oov2syllables_dict = {}
    with open(oov2syllables_path) as f:
        data = [line.strip() for line in f]
        for line in data:
            if line and line[0] != "#" and "[" not in line:
                temp = (re.sub(" +", " ", line)).split()
                if len(temp) > 1:
                    word, syllables = temp[0], temp[1:]
                    oov2syllables_dict[word] = syllables
    return oov2syllables_dict

def syllable2phonemes(phone_vn_path):
    syllable2phonemes_dict = {}
    with open(phone_vn_path) as f:
        data = [line.strip() for line in f]
        for line in data:
            if line and line[0] != "#":
                temp = line.strip().split()
                syllable, phonemes = temp[0], temp[1:]
                syllable2phonemes_dict[syllable] = phonemes
    return syllable2phonemes_dict

def get_oov_phone_dict(oov2syllables_path, phone_vn_path):
    oov2syllables_dict = oov2syllables(oov2syllables_path)
    syllable2phonemes_dict = syllable2phonemes(phone_vn_path)
    if oov2syllables_dict:
        phone_oov_dict_path = os.path.join(os.path.dirname(oov2syllables_path), 'phone_oov')
        phone_oov_dict_file = open(phone_oov_dict_path, 'w')
        oov2syllables_error = ''
        for oov_word in oov2syllables_dict:
            phone_list = []
            for syllable in oov2syllables_dict[oov_word]:
                if syllable not in syllable2phonemes_dict.keys():
                    oov2syllables_error += oov_word + " " + " ".join(oov2syllables_dict[oov_word]) + "\n"
                    phone_list = []
                    break
                else:
                    phone_list.append(" ".join(syllable2phonemes_dict[syllable]))
                    continue
            if phone_list:
                phone_oov_dict_file.write(oov_word + " " + " ".join(phone_list) + "\n")
        print(
            f'Created phone_oov at {os.path.dirname(oov2syllables_path)}')

        if oov2syllables_error:
            oov2syllables_error_path = oov2syllables_path.replace('.txt', '_error.txt')
            with open(oov2syllables_error_path, 'w') as f:
                f.write(oov2syllables_error)
            print(
                f'Created {os.path.basename(oov2syllables_error_path)} at {os.path.dirname(oov2syllables_path)}')

def read_metadata(metadata_path, delimiter='|'):
    extension = os.path.splitext(metadata_path)[1]
    metadata = []
    with open(metadata_path, encoding='utf-8') as f:
        if extension == '.csv':
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                audio_name = row[0]
                text = row[1]
                if len(row) > 2:
                    id = row[2]
                    metadata.append([audio_name, text, id])
                else:
                    metadata.append([audio_name, text])
        else:
            metadata = [line.strip().split(delimiter) for line in f if line]
    return metadata


import random


wind_sound_full_dict = {'sờ': 's',
                    'xờ': 'x',
                    'dờ': 'd',
                    'tờ': 't',
                    'vờ': 'v',
                    'bờ': 'b',
                    'pờ': 'p',
                    'cờ': 'c',
                    'đờ': 'đ',
                    'gờ': 'g',
                    'chờ': 'ch',
                    'lờ': 'l',
                    'nờ': 'n',
                    'phờ': 'ph',
                    'thờ': 'th'}

end_wind_sound_list = ['xờ', 'sờ']

class TextEmbedding:
    def __init__(self, config, word2phone_dict={}, symbol2numeric_dict={}, load_g2s=False):
        if config is None:
            config = {
                        "letters":"aáàạãảăắằặẵẳâấầậẫẩbcdđeéèẹẽẻêếềệễểghiíìịĩỉklmnoóòọõỏôốồộỗổơớờợỡởpqrstuúùụũủưứừựữửvxyýỳỵỹỷfjzw",
                        "coda_nucleus_and_semivowel": ["iz", "pc", "nz", "tc", "ngz", "kc", "uz", "mz", "aa", "ee", "ea", "oa", "aw", "ie", "uo", "a", "wa", "oo", "e", "i", "o", "u", "ow", "uw", "w"],
                        "special":"-",
                        "eos":"#",
                        "punctuation":"~,.*",
                        "p_phone_mix":0.0,
                        "phone_vn_train":"./text/phone_vn_north.txt",
                        "phone_oov_train":"",
        }
        self.p_phone_mix = config['p_phone_mix']
        self.punctuation = config['punctuation']
        self.eos = config['eos']
        if word2phone_dict:
            self.word2phone_dict = word2phone_dict
        else:
            self.word2phone_dict = word2phone_2(config['phone_vn_train'], config['phone_oov_train'], config['coda_nucleus_and_semivowel'])
        if symbol2numeric_dict:
            self.symbol2numeric_dict = symbol2numeric_dict
        else:
            self.symbol2numeric_dict = self.symbol2numeric(config)
        self.load_g2s = load_g2s
        if self.load_g2s:
            self.g2s_model, self.g2s_dict = load_model()
        else:
            self.g2s_model, self.g2s_dict = None, None
        self.connect = config['special']
        self.wind_sound_full_dict = wind_sound_full_dict

    def symbol2numeric(self, config):
        letters_lst = list(config['letters'])
        phone2numeric_dict = phone2numeric(self.word2phone_dict)
        phonemes_lst = list(phone2numeric_dict.keys())
        if 1 > self.p_phone_mix > 0:
            symbols = letters_lst + phonemes_lst
        elif self.p_phone_mix >= 1:
            symbols = phonemes_lst
        else:
            symbols = letters_lst
        symbols = list(' ' + self.punctuation + config['special']) + symbols
        if config['eos'] not in symbols:
            symbols = symbols + list(config['eos'])
        symbol2numeric_dict = {s: i for i, s in enumerate(symbols)}
        return symbol2numeric_dict

    def text_norm(self, text, end_ws_list=end_wind_sound_list, ws_list=wind_sound_full_dict.keys()):
        text = text.strip()
        # print(self.eos, text[-1] in self.eos)
        while text[-1] in (self.punctuation + self.eos + ' '):
            text = text[:-1]
        # print(text,"ok")
        text_out = ''
        for word in text.split():
            if self.connect in word:
                word_out = ''
                syl_list = word.split(self.connect)
                if end_ws_list:
                    while syl_list[-1] in wind_sound_full_dict.values() and syl_list[-1] not in [wind_sound_full_dict[ws] for ws in end_ws_list]:
                        syl_list = syl_list[:-1]
                for syl in syl_list:
                    if syl not in self.wind_sound_full_dict.values() or syl in [wind_sound_full_dict[ws] for ws in ws_list]:
                        word_out += syl + '-'
                text_out += word_out[:-1] + ' '
            else:
                text_out += word + ' '
        if self.eos:
            text_out = text_out + self.eos
        else:
            text_out = text_out[:-1]
        return text_out

    def g2s(self, text, end_ws_list=end_wind_sound_list, ws_list=wind_sound_full_dict.keys()):
        if self.g2s_model is None:return text
        text_output = ''
        words = [word for word in text.split() if word]
        for word in words:
            if word not in self.word2phone_dict.keys() and self.connect not in word and word not in (self.punctuation + self.eos):
                result = G2S(word, self.g2s_model, self.g2s_dict)
                if result != word:
                    syl_list = result[0].split()[1:]
                    if syl_list[0] != word and len([syl for syl in syl_list if '(' in syl]) != len(syl_list):
                        if end_ws_list:                            
                            while syl_list and '(' in syl_list[-1] and syl_list[-1].replace('(', '').replace(')', '') not in end_ws_list:
                                syl_list = syl_list[:-1]
                        if ws_list:
                            syllables = ''
                            for syl in syl_list:
                                if '(' in syl:
                                    syl = syl.replace('(', '').replace(')', '')
                                    if syl in ws_list:
                                        syl = self.wind_sound_full_dict[syl]
                                    else:
                                        syl = ''
                                if syl:
                                    syllables += syl + self.connect
                            syllables = syllables[:-1]

                        else:
                            syllables = self.connect.join([syl for syl in syl_list if '(' not in syl])
                        if syllables:
                            text_output += syllables + ' '
            else:
                text_output += word + ' '
        return text_output[:-1]

    def text2seq(self, text):
        sequence = []
        words_of_text = [word for word in text.split() if word]
        for word in words_of_text:
            append = False
            if random.random() < self.p_phone_mix:
                if word in self.word2phone_dict.keys() and self.word2phone_dict[word]:
                    phonemes = self.word2phone_dict[word]
                    for phoneme in phonemes:
                        sequence.append(self.symbol2numeric_dict[phoneme])
                elif self.p_phone_mix >= 1:
                    if word in self.symbol2numeric_dict.keys():
                        sequence.append(self.symbol2numeric_dict[word])
                        append = True
                    else:
                        print(("{} not in phone_train_dict".format(word)))
            else:
                for symbol in word:
                    if symbol in self.symbol2numeric_dict.keys():
                        sequence.append(self.symbol2numeric_dict[symbol])
                        append = True
                    else:
                        print(("{} not in symbols_dict".format(symbol)))
                        print(text)
            if append:
                sequence.append(self.symbol2numeric_dict[' '])
        return sequence[:-1]