import torch
from transformers import BertTokenizer, BertModel, ErnieModel


def is_equal(convert_path, raw_path, sentence):
    my_tokenizer = BertTokenizer.from_pretrained(convert_path)
    my_model = BertModel.from_pretrained(convert_path)

    my_inputs = my_tokenizer(sentence, return_tensors='pt')
    my_model.eval()
    with torch.no_grad():
        my_encoded = my_model(**my_inputs)['last_hidden_state']

    ernie_tokenizer = BertTokenizer.from_pretrained(raw_path)
    ernie_model = ErnieModel.from_pretrained(raw_path)

    ernie_inputs = ernie_tokenizer(sentence, return_tensors='pt')
    ernie_model.eval()
    with torch.no_grad():
        ernie_encoded = my_model(**ernie_inputs)['last_hidden_state']

    return my_encoded.equal(ernie_encoded)


if __name__ == '__main__':
    test_str = "综合报道，卡塔尔当地时间22日，沙特阿拉伯队在世界杯小组赛中以2：1比分战胜阿根廷队后，沙特官方宣布全国放假一天，庆祝胜利。"
    my_path = 'convert_path'
    ernie_path = 'raw_path'
    print(is_equal(my_path, ernie_path, test_str))
