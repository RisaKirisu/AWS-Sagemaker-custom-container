def prompt_llama(sys, usr):
    return '<s>[INST] <<SYS>>\n{}<</SYS>>\n\n{} [/INST]'.format(sys, usr)

def prompt_chatml(sys, usr):
    return '<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>'.format(sys, usr)

def stop_condition_chatml(tokenizer):
    return [tokenizer.eos_token_id,
            '<|im_end|>']

def stop_condition_llama(tokenizer):
    return [tokenizer.eos_token_id]

def parse_output_llama(text, usr):
    return text[text.find('[/INST]') + len('[/INST]'):].strip('\n ')

def parse_output_chatml(text, usr):
    return text[text.find(usr) + len(usr):].strip('\n ')

prompt_formatter = {
    "llama": prompt_llama,
    "chatml": prompt_chatml,
}

stop_condition = {
    "llama": stop_condition_llama,
    "chatml": stop_condition_chatml,
}

parse_output = {
    "llama": parse_output_llama,
    "chatml": parse_output_chatml,
}