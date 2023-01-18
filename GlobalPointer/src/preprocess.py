import json

from tqdm import tqdm
from transformers import BertTokenizerFast


class Preprocessor:
    """针对CLUENER2020数据集的预处理"""
    def __init__(self, pretrained_bert_path):
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_bert_path, do_lower_case=True)

    def data_preprocess(self, input_path, output_path, dtype):
        """数据预处理"""
        processed_data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f'Parsing {dtype} dataset'):
                sample = json.loads(line)
                sample = self._parse_data(sample)
                processed_data.append(sample)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

    def _parse_data(self, data):
        """解析单条数据"""
        text = data['text']
        codes = self.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
        sample = {
            'text': text,
            'tokens': self.tokenizer.convert_ids_to_tokens(codes['input_ids']),
            'input_ids': codes['input_ids'],
            'offset_mapping': codes['offset_mapping']
        }
        self._parse_entities(data, sample)

        return sample

    def _parse_entities(self, data, sample):
        """解析实体信息"""
        label = data['label']
        entity_list = []
        for entity_label, entity_dic in label.items():
            for entity, span_list in entity_dic.items():
                for char_span_start, char_span_end in span_list:
                    token_span_start, token_span_end = self._parse_token_span(sample, char_span_start, char_span_end + 1)
                    if token_span_start is None or token_span_end is None:
                        print(f'invalid token span for entity: {entity}')
                        continue
                    entity_list.append({
                        'text': entity,
                        'type': entity_label,
                        'token_span': (token_span_start, token_span_end),
                        'char_span': (char_span_start, char_span_end + 1)
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


if __name__ == '__main__':
    bert_path = r'./bert-base-chinese'
    processor = Preprocessor(bert_path)
    processor.data_preprocess(r'../data/raw/train.json', r'../data/preprocessed/train.json', dtype='train')
    processor.data_preprocess(r'../data/raw/dev.json', r'../data/preprocessed/dev.json', dtype='dev')
