# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import time
from datetime import datetime

import flask
import torch

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

from formats import prompt_formatter, stop_condition, parse_output

DEBUG = True

model_dir = "/opt/ml/model/Llama2-70B-chat-exl2"
gpu_split = [13, 13, 13, 13]
prompt_format = "llama"

def LOG(log: str):
    print("{}:".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), log)

def LOG_DEBUG(log: str):
    if DEBUG:
        LOG(log)


def load_model_args(model_kargs: dict = None) -> ExLlamaV2Sampler.Settings:
    settings = ExLlamaV2Sampler.Settings()
    if model_kargs is None:
        settings.temperature = 0.7
        settings.top_k = 40
        settings.top_p = 0.1
        settings.token_repetition_penalty = 1.176
    else:
        settings.temperature = model_kargs['temperature'] if model_kargs.get('temperature') is not None else 0.7
        settings.top_k = model_kargs['top_k'] if model_kargs.get('top_k') is not None else 40
        settings.top_p = model_kargs['top_p'] if model_kargs.get('top_p') is not None else 0.1
        settings.token_repetition_penalty = model_kargs['repetition_penalty'] if model_kargs.get('repetition_penalty') is not None else 1.176

    return settings

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class Llama(object):
    model = None  # Where we keep the model when it's loaded
    tokenizer = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            if not os.path.isdir(model_dir):
                raise FileNotFoundError("Model dir {} doesn't exist.".format(model_dir))
            else:
                # Load model using exllamav2
                config = ExLlamaV2Config()
                config.model_dir = model_dir
                config.prepare()
                # config.max_seq_len = 2048

                LOG("Loading model: " + model_dir)
                cls.model = ExLlamaV2(config)
                cls.tokenizer = ExLlamaV2Tokenizer(config)
                cls.model.load(gpu_split)

        return cls.model, cls.tokenizer

    @classmethod
    def predict(cls, input: str, model_kargs: dict = None) -> dict:
        """For the input, do the predictions and return them.

        Args:
            input: LLM Prompt
            model_kargs: model generator setting
        """
        model, tokenizer = cls.get_model()

        # Initialize cache
        cache = ExLlamaV2Cache(model)

        # Initialize generator
        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        generator.warmup()
 
        settings = load_model_args(model_kargs)
        max_tokens = model_kargs['max_tokens'] if model_kargs.get('max_tokens') is not None else 250

        return generator.generate_simple(input, settings, max_tokens, encode_special_tokens=True)

    @classmethod
    def get_stream_generator(cls, input: str, model_kargs: dict = None) -> ExLlamaV2StreamingGenerator:
        """For the input, return a stream generator that can be used to generate stream output.

        Args:
            input: LLM Prompt
            model_kargs: model generator setting
        """
        model, tokenizer = cls.get_model()

        # Initialize cache
        cache = ExLlamaV2Cache(model)

        # Initialize generator
        generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
        settings = load_model_args(model_kargs)
        generator.set_stop_conditions(stop_condition[prompt_format](tokenizer))

        # Encode prompt
        context = tokenizer.encode(input, add_bos=False, add_eos=False, encode_special_tokens=True)
        generator.begin_stream(context, settings)

        return generator


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = Llama.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    response = "model loaded." if health else "model not loaded"
    return flask.Response(response=response, status=status, mimetype="application/json")


# @app.route("/invocations", methods=["POST"])
# def invocations():
#     """
#     Do an inference on a single batch of data.
#     """
#     # Input Json format:
#     # {
#     # "inputs": [
#     #     [
#     #     {
#     #         "role": "system",
#     #         "content": "<content>"
#     #     },
#     #     {
#     #         "role": "user",
#     #         "content": "<prompt>"
#     #     }
#     #     ]
#     # ],
#     # "parameters": {**model_kargs}
#     # }

#     data = None

#     # Parse input json
#     if flask.request.content_type == "application/json":
#         data = flask.request.get_json()
#         if DEBUG:
#             print("==== Got Request ====")
#             print(json.dumps(data, indent=2))
#             print("=====================")

#     else:
#         return flask.Response(
#             response="This predictor only supports json", status=415, mimetype="text/plain"
#         )

#     # Construct prompt from request input
#     inputs = data["inputs"]
#     model_kargs = data["parameters"]

#     sys_msg = inputs[0][0]['content']
#     usr_msg = inputs[0][1]['content']
#     prompt = prompt_formatter[prompt_format](sys_msg, usr_msg)
#     if DEBUG:
#         print("====== Prompt ======")
#         print(prompt)
#         print("====================")

#     # Do the prediction
#     start_time = time.time()
#     prediction = Llama.predict(prompt, model_kargs)
#     end_time = time.time()

#     if DEBUG:
#         print("==== Generated Text ====")
#         print(prediction)
#         print("========================")

#     # Construct response
#     prediction = parse_output[prompt_format](prediction, usr_msg)

#     result = {
#         "text": prediction,
#         "prediction_time": end_time - start_time,
#     }
#     if DEBUG:
#         print("==== Result ====")
#         print(json.dumps(result, indent=2))
#         print("================")

#     result = json.dumps(result)

#     torch.cuda.empty_cache()

#     return flask.Response(response=result, status=200, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations_stream():

    data = None

    # Parse input json
    if flask.request.content_type == "application/json":
        data = flask.request.get_json()
        LOG("==== Got Request Streaming ====")
        LOG_DEBUG(json.dumps(data, indent=2))

    else:
        return flask.Response(
            response="This predictor only supports json", status=415, mimetype="text/plain"
        )

    # Construct prompt from request input
    prompt = data["prompt"]
    model_kargs = data["parameters"]

    LOG_DEBUG("==== Prompt ====")
    LOG_DEBUG(prompt)
    
    # Get stream generator (model)
    generator = Llama.get_stream_generator(prompt, model_kargs)
    
    max_tokens = model_kargs['max_tokens'] if model_kargs.get('max_tokens') is not None else 250
    print_timing = model_kargs['print_timing'] if model_kargs.get('print_timing') is not None else False
    
    LOG("==== Start Generating ====")

    start_time = time.time()

    def stream_output_formatter(token_texts: list):
        token_texts = {"outputs": token_texts}
        json_encoded_str = json.dumps(token_texts) + "\n"
        return json_encoded_str.encode("utf-8")

    # Generator function for streaming HTTP response
    def generate():
        response_tokens = 0

        # Generate tokens
        for response_tokens in range(max_tokens):
            response_text, eos, _ = generator.stream()
            
            if eos:
                break
            else:
                LOG_DEBUG(response_text, end='')
                yield stream_output_formatter([response_text])

        LOG('==== Finished Generating ====')

        time_passed = time.time() - start_time
        speed = (response_tokens + 1) / time_passed
        LOG(f"(Response: {response_tokens} tokens, {speed:.2f} tokens/second)")
        
        if print_timing:
            yield stream_output_formatter([f"\n\n(Response: {response_tokens} tokens, {speed:.2f} tokens/second)"])
    
    return flask.Response(flask.stream_with_context(generate()), status=200, mimetype="application/json", headers={'X-Accel-Buffering': 'no'})
