import torch
from transformers import AutoTokenizer, GenerationConfig
from transformers.models.t5 import T5ForConditionalGeneration
from argparse import ArgumentParser
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

def main(args):

    base_id = "QizhiPei/biot5-plus-base"
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = T5ForConditionalGeneration.from_pretrained(base_id)
    ckpt = torch.load(args.ckpt_path, weights_only=False, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    print(f"Original state_dict keys: {len(state_dict)}")

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("t5_model."):
            new_state_dict[k[len("t5_model."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    text_to_molecule_params = {
        "max_length": 512,
        "num_beams": 4,
        "do_sample": False,
        "early_stopping": True,
        "no_repeat_ngram_size": 0,
        "length_penalty": 1.0,
    }

    if getattr(model.config, "task_specific_params", None) is None:
        model.config.task_specific_params = {}
    model.config.task_specific_params["text_to_molecule_generation"] = text_to_molecule_params
    model.generation_config = GenerationConfig(**text_to_molecule_params)

    name_or_path_hub = f"Neeze/{args.model_name}"
    tokenizer.push_to_hub(name_or_path_hub, private=True)
    model.push_to_hub(name_or_path_hub, private=True, safe_serialization=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="biot5-plus-base-sft")
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
