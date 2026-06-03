
import os, sys

from tqdm import tqdm
now_dir = os.getcwd()
sys.path.append(now_dir)

import re
import torch
import LangSegment
from core.tts.text import chinese   # if you use chinese text, uncomment this line
from typing import Dict, List, Tuple
from core.tts.text.cleaner import clean_text
from core.tts.text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from core.tts.TTS_infer_pack.text_segmentation_method import split_big_text, splits, get_method as get_seg_method

from core.tts.tools.i18n.i18n import I18nAuto, scan_language_list

language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)
punctuation = set(['!', '?', '…', ',', '.', '-'," "])

def get_first(text:str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def merge_short_text_in_array(texts:str, threshold:int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result





class TextPreprocessor:
    def __init__(self, bert_model:AutoModelForMaskedLM, 
                 tokenizer:AutoTokenizer, device:torch.device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        
    def preprocess(self, text:str, lang:str, text_split_method:str, version:str="v2")->List[Dict]:
        print(i18n("############ Segment text ############"))
        text = self.replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        print(i18n("############ Extracting text 'Bert' features ############"))
        for text in tqdm(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang, version)
            if phones is None or norm_text=="":
                continue
            res={
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    def pre_seg_text(self, text:str, lang:str, text_split_method:str):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if (text[0] not in splits and len(get_first(text)) < 4): 
            text = "。" + text if lang != "en" else "." + text
        print(i18n("The actual input target text:"))
        print(text)
        
        seg_method = get_seg_method(text_split_method)
        text = seg_method(text)
        
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = text.split("\n")
        _texts = self.filter_text(_texts)
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        
        for text in _texts:
            # 대상 텍스트의 빈 줄을 입력하여 발생하는 오류 문제를 해결하세요.
            if (len(text.strip()) == 0):
               continue
            if not re.sub("\W+", "", text):       
                # 확인하고 순수한 심볼이라면 건너뛰세요.
                continue
            if (text[-1] not in splits): text += "。" if lang != "en" else "."
            
            # Bert가 긴 문장으로 인해 오류를 보고하는 문제를 해결하세요.
            if (len(text) > 510):
                texts.extend(split_big_text(text))
            else:
                texts.append(text)
            
        print(i18n("The actual input target text (after sentence segmentation):"))
        print(texts)
        return texts
    
    def segment_and_extract_feature_for_text(self, text:str, language:str, version:str="v1")->Tuple[list, torch.Tensor, str]:
        return self.get_phones_and_bert(text, language, version)
        
    def get_phones_and_bert(self, text:str, language:str, version:str, final:bool=False):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_","")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 중국어, 일본어, 한국어 문자는 구분이 불가능하므로 사용자 입력을 우선시합니다.
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                if re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext,"zh",version)
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext,"yue",version)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float32,
                ).to(self.device)
        if language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist=[]
            langlist=[]
            LangSegment.setfilters(["zh","ja","en","ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 중국어, 일본어, 한국어 문자는 구분이 불가능하므로 사용자 입력을 우선시합니다.
                        langlist.append(language)
                    textlist.append(tmp["text"])
            # print(textlist)
            # print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text,language,version,final=True)

        return phones, bert, norm_text


    def get_bert_feature(self, text:str, word2ph:list)->torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T
    
    def clean_text_inf(self, text:str, language:str, version:str="v2"):
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones:list, word2ph:list, norm_text:str, language:str):
        language=language.replace("all_","")
        if language == "zh":
            feature = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            feature = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            ).to(self.device)

        return feature


    def filter_text(self,texts):
        _text=[]
        if all(text in [None, " ", "\n",""] for text in texts):
            raise ValueError(i18n("유효한 텍스트를 입력하세요"))
        for text in texts:
            if text in  [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text
    

    def replace_consecutive_punctuation(self,text):
        punctuations = ''.join(re.escape(p) for p in punctuation)
        pattern = f'([{punctuations}])([{punctuations}])+'
        result = re.sub(pattern, r'\1', text)
        return result



