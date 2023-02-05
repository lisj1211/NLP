import os
import re
import json
from pprint import pprint

from tqdm import tqdm
from transformers import BertTokenizerFast


class BaiduDataPreprocessor:
    """针对Baidu关系抽取数据集的预处理"""
    def __init__(self, pretrained_bert_path):
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_bert_path, do_lower_case=True)
        self.rm_space = lambda x: re.sub(r'\s+', '', x)

    def data_preprocess(self, input_path, output_path, dtype):
        """数据预处理"""
        processed_data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f'Parsing {dtype} dataset'):
                sample = json.loads(line)
                sample = self._parse_data(sample)
                if self._check_span(sample):
                    processed_data.append(sample)
                else:
                    pprint(sample)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

    def _parse_data(self, data):
        """解析单条数据"""
        text = data['text']
        text = self.rm_space(text)
        codes = self.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
        sample = {
            'text': text,
            'tokens': self.tokenizer.convert_ids_to_tokens(codes['input_ids']),
            'input_ids': codes['input_ids'],
            'offset_mapping': codes['offset_mapping']
        }
        self._parse_entities(data, sample)
        self._parse_relations(data, sample)

        return sample

    def _parse_entities(self, data, sample):
        """解析实体信息"""
        text = sample['text']
        entity_list = []
        entities = [(self.rm_space(spo['subject']), spo['subject_type']) for spo in data['spo_list']]
        entities.extend([(self.rm_space(spo['object']), spo['object_type']) for spo in data['spo_list']])
        for ent, ent_tp in entities:
            for m in re.finditer(re.escape(ent), text):
                # get char and token span
                char_span_start, char_span_end = m.span()[0], m.span()[1]
                token_span_start, token_span_end = self._parse_token_span(sample, char_span_start, char_span_end)
                if token_span_start is None or token_span_end is None:
                    print(f'invalid token span for entity: {ent}')
                    continue
                entity_list.append({
                    'text': ent,
                    'type': ent_tp,
                    'token_span': (token_span_start, token_span_end),
                    'char_span': (char_span_start, char_span_end)
                })

        sample.update({
            'entity_list': entity_list
        })

    @staticmethod
    def _parse_token_span(sample, start, end):
        """获得token level范围"""
        token_start, token_end = None, None
        for idx, (token, offset) in enumerate(zip(sample['tokens'], sample['offset_mapping'])):
            if offset[0] == start and offset[1] == end:
                return idx, idx + 1
            if offset[0] == start:
                token_start = idx
            if end == offset[1]:
                token_end = idx
            if token_start is not None and token_end is not None:
                return token_start, token_end + 1
        return token_start, token_end

    def _parse_relations(self, data, sample):
        """解析关系信息"""
        entity_list = sample['entity_list']
        relations_list = []
        for spo in data['spo_list']:
            subj_list = [ent_info for ent_info in entity_list if ent_info['text'] == self.rm_space(spo['subject'])]
            obj_list = [ent_info for ent_info in entity_list if ent_info['text'] == self.rm_space(spo['object'])]
            for subj_info in subj_list:
                for obj_info in obj_list:
                    relations_list.append({
                        'subject': subj_info['text'],
                        'object': obj_info['text'],
                        'subj_char_span': subj_info['char_span'],
                        'obj_char_span': obj_info['char_span'],
                        'subj_tok_span': subj_info['token_span'],
                        'obj_tok_span': obj_info['token_span'],
                        'predicate': spo['predicate'],
                    })

        sample.update({
            'relation_list': relations_list
        })

    @staticmethod
    def _check_span(sample):
        """对获得的范围进行校验"""
        tokens = sample['tokens']
        text = sample['text']
        if '[UNK]' in tokens:
            return True
        for entity in sample['entity_list']:
            true = entity['text']
            token_start, token_end = entity['token_span']
            token_based = ''.join(tokens[token_start:token_end])
            if '#' in token_based:
                token_based = re.sub(r'#+', '', token_based)
            char_start, char_end = entity['char_span']
            char_based = text[char_start:char_end]
            if true != token_based or true != char_based or token_based != char_based:
                return False

        return True


if __name__ == '__main__':
    bert_path = r'./bert-base-chinese'
    processor = BaiduDataPreprocessor(bert_path)
    os.makedirs(r'../data/baidu/preprocessed', exist_ok=True)
    processor.data_preprocess(r'../data/baidu/raw/train.json', r'../data/baidu/preprocessed/train.json', dtype='train')
    processor.data_preprocess(r'../data/baidu/raw/dev.json', r'../data/baidu/preprocessed/dev.json', dtype='dev')
    processor.data_preprocess(r'../data/baidu/raw/test.json', r'../data/baidu/preprocessed/test.json', dtype='test')
