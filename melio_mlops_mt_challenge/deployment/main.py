"""Kserve inference script."""

import gc
import argparse
import re

from kserve import (
    InferOutput,
    InferRequest,
    InferResponse,
    Model,
    ModelServer,
    model_server,
)
from kserve.utils.utils import generate_uuid
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from torch import bfloat16


MODEL_DIR = "/app/saved_model"
TOKENIZER_CONFIG = {
    "src_lang": "dyu_Latn",
    "tgt_lang": "fra_Latn",
}
# Set `do_sample=False` for deterministic generation per the competition rules
DECODER_KWARGS = {
    "do_sample": False,
}
CHARS_TO_REMOVE_REGEX = r'[!"&\(\),-./:;=?+.\n\[\]]'


def clean_text(text: str) -> str:
    """
    Clean input text by removing special characters and converting
    to lower case.
    """
    return re.sub(CHARS_TO_REMOVE_REGEX, " ", text.lower()).strip()


class MyModel(Model):
    """Kserve inference implementation of model."""

    def __init__(self, name: str):
        """Initialise model."""
        super().__init__(name)
        self.name = name
        self.model = None
        self.tokenizer = None
        self.ready = False
        self.load()

    def load(self):
        """Reconstitute model from disk."""
        # Load model and tokenizer
        self.tokenizer = NllbTokenizer.from_pretrained(
            MODEL_DIR,
            src_lang=TOKENIZER_CONFIG["src_lang"],
            tgt_lang=TOKENIZER_CONFIG["tgt_lang"]
        )
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=bfloat16,
            use_safetensors=True
        )
        self.model.eval()
        self.ready = True

    def preprocess(self, payload: InferRequest, headers=None) -> str:
        """Preprocess inference request."""
        # Clean input sentence and add prefix
        return clean_text(payload.inputs[0].data[0])

    def predict(self, payload: str, headers=None) -> InferResponse:
        # """Pass inference request to model to make prediction."""
        a, b, max_input_length = 16, 1.5, 128
        tgt_lang = TOKENIZER_CONFIG["tgt_lang"]

        # Model prediction preprocessed sentence
        inputs = self.tokenizer(
            payload, return_tensors='pt', padding=True, truncation=True,
            max_length=max_input_length
        )
        result = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
            **DECODER_KWARGS
        )
        translation = self.tokenizer.decode(result[0], skip_special_tokens=True)

        response_id = generate_uuid()
        infer_output = InferOutput(
            name="output-0", shape=[1], datatype="STR", data=[translation]
        )
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        _ = gc.collect()
        return infer_response


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_name", default="model", help="The name that the model is served under."
)
cmd_args, _ = parser.parse_known_args()

if __name__ == "__main__":
    ModelServer().start([MyModel(cmd_args.model_name)])
