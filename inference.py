import torch

import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    return generator

def run(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
):
    local_rank = 0
    world_size = 1
    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)

    chat_log = [
       ("Adam", " Hi, I'm an assistant chatbot, I am well versed in mathematics, geography, and cars."),
       ("User", " What is 2 + 2?"),
       ("Adam", " 2 + 2 = 4."),
    ]
    while True:
        # get user input
        user_input = input("User: ")
        chat_log += [("User", user_input)]

        # Prepare prompt for Adam
        prompt = "\n".join([f"{user}:{text}[EOS]" for user,text in chat_log])
        prompt += "\nAdam:"

        # do generation
        result = generator.generate([prompt], max_gen_len=256, temperature=temperature, top_p=top_p)[0]
        out = result[len(prompt):]
        out = out[:out.find("[EOS]"):]

        # append to the chat log
        chat_log += [("Adam:", out)]

        # print
        print(f"Adam:{out}")



def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/llama_data/7B")
    parser.add_argument("--tokenizer_path", type=str, default="/llama_data/tokenizer.model")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        temperature=0.8,
        top_p=0.95,
        max_seq_len=1024,
        max_batch_size=1
    )



