

class SampleTruncator:
    """对大于给定长度的句子进行截断"""
    def __init__(self, max_sequence_length, window_size):
        self.max_sequence_length = max_sequence_length
        self.window_size = window_size

    def truncate(self, sample):
        all_samples = []

        text = sample['text']
        if len(sample['input_ids']) < self.max_sequence_length:
            all_samples.append({
                'text': text,
                'entity_list': sample['entity_list'],
                'relation_list': sample['relation_list'],
                'offset_mapping': self._adjust_offset_mapping(sample['offset_mapping'], 0),
                'token_offset': 0,
                'char_offset': 0,
            })
            return all_samples

        tokens = sample['tokens']
        offset = sample['offset_mapping']

        for start in range(0, len(tokens), self.window_size):
            # do not truncate word pieces
            while str(tokens[start]).startswith('##'):
                start -= 1
            end = min(start + self.max_sequence_length, len(tokens))
            range_offset_mapping = offset[start: end]
            char_span = [range_offset_mapping[0][0], range_offset_mapping[-1][1]]
            text_subs = text[char_span[0]:char_span[1]]

            token_offset = start
            char_offset = char_span[0]

            truncated_sample = {
                'text': text_subs,
                'entity_list': self._adjust_entity_list(sample, start, end, token_offset, char_offset),
                'relation_list': self._adjust_relation_list(sample, start, end, token_offset, char_offset),
                'offset_mapping': self._adjust_offset_mapping(range_offset_mapping, char_offset),
                'token_offset': token_offset,
                'char_offset': char_offset
            }
            all_samples.append(truncated_sample)

            if end >= len(tokens):
                break

        return all_samples

    @staticmethod
    def _adjust_entity_list(sample, start, end, token_offset, char_offset):
        entity_list = []
        for entity in sample['entity_list']:
            token_span = entity['token_span']
            char_span = entity['char_span']
            if token_span[0] < start or token_span[1] > end:
                continue
            entity_list.append({
                'text': entity['text'],
                'type': entity['type'],
                'token_span': [token_span[0] - token_offset, token_span[1] - token_offset],
                'char_span': [char_span[0] - char_offset, char_span[1] - char_offset]
            })
        return entity_list

    @staticmethod
    def _adjust_relation_list(sample, start, end, token_offset, char_offset):
        relation_list = []
        for relation in sample['relation_list']:
            subj_token_span, obj_token_span = relation['subj_tok_span'], relation['obj_tok_span']
            subj_char_span, obj_char_span = relation['subj_char_span'], relation['obj_char_span']
            if subj_token_span[0] >= start and subj_token_span[1] <= end \
                    and obj_token_span[0] >= start and obj_token_span[1] <= end:
                relation_list.append({
                    'subj_tok_span': [subj_token_span[0] - token_offset, subj_token_span[1] - token_offset],
                    'obj_tok_span': [obj_token_span[0] - token_offset, obj_token_span[1] - token_offset],
                    'subj_char_span': [subj_char_span[0] - char_offset, subj_char_span[1] - char_offset],
                    'obj_char_span': [obj_char_span[0] - char_offset, obj_char_span[1] - char_offset],
                    'subject': relation['subject'],
                    'object': relation['object'],
                    'predicate': relation['predicate'],
                })
        return relation_list

    @staticmethod
    def _adjust_offset_mapping(offset_mapping, char_offset, max_sequence_length=100):
        offsets = []
        for start, end in offset_mapping:
            offsets.append([start - char_offset, end - char_offset])
        # padding to max_sequence_length to avoid DataLoader runtime error
        while len(offsets) < max_sequence_length:
            offsets.append([0, 0])
        return offsets
