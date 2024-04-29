import argparse

def parseargs():
    parser = argparse.ArgumentParser(description='Config for using LLMs.')
    parser.add_argument('-mn', '--mname', help='The path or name from hugging face of the model.', required=True, type=str)
    parser.add_argument('-pe', '--peftmp', help='The path or name from hugging face of peft model.', required=False, type=str)
    parser.add_argument('-mt', '--mtype', help='The type of model: seq2seq, decoder', required=True, type=str)
    parser.add_argument('-qu', '--quantize', help='Whether to quantize the model or not', required=False, type=str)
    parser.add_argument('-hf', '--hftoken', help='Hugging face token to use for authentication', required=False, type=str)
    parser.add_argument('-tr', '--training', help='The type of model: seq2seq, decoder', required=False, type=str)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseargs()
    lines = [line for line in vars(args).values()]
    
    with open('Model/server_config.txt', 'w') as f:
        for line in lines:
            f.write(str(line) + '\n')

