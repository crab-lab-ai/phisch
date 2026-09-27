# Usage
This is the long version of the Usage section in the [README](./README.md).   
Here you'll find every command, every flag, what goes in a config file and every file that training leaves behind. If you only want the quick pipeline, the README should be enough.

```sh
Usage: phisch <COMMAND>

Commands:
  config     Generate a default config
  tokenizer  Train the tokenizer
  train      Train a Phisch
  generate   Generate a sequence of text
  export     Export a checkpoint as safetensors
  help       Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

A typical pipeline looks like this:

1. `phisch config` to get a config file you can edit
2. `phisch tokenizer` to train a tokenizer on your corpus
3. `phisch train` to train the model with your config
4. `phisch generate` to generate a sequence of text
5. `phisch export` to turn a checkpoint into a safetensors file you can share

---

## Config

Generates a `default-config.json` with the default settings, so you have a base to edit instead of writing one from scratch.

```sh
Usage: phisch config [OPTIONS]

Options:
  -p, --path <PATH>  Directory where the config file will generate [default: configs/]
  -h, --help         Print help
```

| Flag | Default | Description |
|---|---|---|
| `-p`, `--path` | `configs/` | Directory where `default-config.json` gets written. Created if it doesn't exist. If a file with that name already exists, it's overwritten. |

Every setting inside the file is explained in [The config file](#the-config-file).

---

## Tokenizer

Trains a byte-level BPE tokenizer on your corpus. This is the vocabulary your model will speak, so train it on the same kind of text you're going to train the model on.

```sh
Usage: phisch tokenizer [OPTIONS] <FILES>...

Arguments:
  <FILES>...  Corpus files to train on

Options:
  -o, --out <OUT>      Path of the resulting tokenizer [default: data/tokenizer.json]
  -v, --vocab <VOCAB>  Size of the vocabulary [default: 4096]
  -h, --help           Print help
```

| Argument / Flag | Default | Description |
|---|---|---|
| `<FILES>...` | *(required)* | One or more text files to learn the vocabulary from. You can pass as many as you want, for example your stories and your chats together. |
| `-o`, `--out` | `data/tokenizer.json` | Where the tokenizer gets saved. Missing folders in the path are created. |
| `-v`, `--vocab` | `4096` | Size of the vocabulary, special tokens included. |

```sh
phisch tokenizer data/stories.txt data/chats.txt -v 8192 -o data/tokenizer-8192.json
```

The tokenizer always reserves these special tokens:

| Token | Used for |
|---|---|
| `<\|system\|>` | Starts a system message |
| `<\|user\|>` | Starts a user message |
| `<\|assistant\|>` | Starts a model message |
| `<\|end\|>` | Ends a message, generation stops when the model produces it |
| `<\|reserved_0\|>` … `<\|reserved_7\|>` | Free slots i'll use in the future, so new special tokens don't need a new tokenizer |

> [!NOTE]
> A pair of characters has to show up at least twice in your corpus to become a token. With a small corpus you can end up with **fewer** tokens than you asked for in `--vocab`.

> [!WARNING]
> The size of the embeddings comes from the tokenizer. If you train a new tokenizer, even with the same `--vocab`, your old checkpoints won't make sense anymore, because every token id now means something different. Train the model again after changing tokenizers.

---

## Train

Trains a model using the settings in a config file.

```sh
Usage: phisch train [OPTIONS] <CONFIG>

Arguments:
  <CONFIG>  Path of a training config

Options:
      --resume                 Continue from the last step checkpoint of this config
  -a, --artifacts <ARTIFACTS>  Directory where checkpoints and configs are written [default: artifacts/]
  -h, --help                   Print help
```

| Argument / Flag | Default | Description |
|---|---|---|
| `<CONFIG>` | *(required)* | Path of the training config, see [The config file](#the-config-file). |
| `--resume` | off | Continues an interrupted run from its last step checkpoint, see [Checkpoints and resuming](#checkpoints-and-resuming). |
| `-a`, `--artifacts` | `artifacts/` | Where every checkpoint and config of the run is written. Created if it doesn't exist. |

```sh
phisch train configs/my-config.json
phisch train configs/my-config.json -a runs/experiment-1
```

While training, phisch logs the epoch, step, learning rate and loss every 25 steps, and the validation loss at the end of every epoch.

### The config file

This is what `phisch config` generates:

```json
{
  "model": {
    "d_model": 1024,
    "d_ff": 2048,
    "seq_length": 256,
    "heads": 8,
    "layers": 6,
    "dropout": 0.1
  },
  "tokenizer": "your-tokenizer.json",
  "dataset": "your-corpus.txt",
  "batch": 4,
  "epochs": 5,
  "lr": 0.001,
  "valid": 0.05,
  "artifact": "model",
  "init": null,
  "seed": 50,
  "warmup": 200,
  "clip": 1.0,
  "checkpoint_minutes": 30,
  "valid_batch": 8
}
```

> [!WARNING]
> Every field has to be in the file, even if you don't want to change it. A missing field is an error when loading, it won't fall back to its default. The only exception is `init`, which you can leave out. That's why starting from the generated file is the easiest way.

#### The model

These decide the shape of the network. `d_model`, `d_ff`, `heads` and `layers` are fixed once you train, if you change any of them old checkpoints won't load anymore, since the weights have a different shape. `seq_length` and `dropout` have no weights, so those can be changed between trainings.

| Field | Default | Description |
|---|---|---|
| `d_model` | `1024` | Width of the model, the size of every token's vector. Has to be divisible by `heads`. |
| `d_ff` | `2048` | Hidden size of the feed-forward (SwiGLU) block inside every layer. |
| `seq_length` | `256` | How many tokens the model sees at once. It's the length of every training sample, and when generating it's how far back the model can look. |
| `heads` | `8` | Number of attention heads. Each head works on `d_model / heads` dimensions. |
| `layers` | `6` | Number of decoder layers stacked on top of each other. |
| `dropout` | `0.1` | Dropout applied to the token embeddings while training. |

#### The training

| Field | Default | Description |
|---|---|---|
| `tokenizer` | — | Path of the tokenizer from `phisch tokenizer`. |
| `dataset` | — | Path of the text file to train on. It's read line by line, so a document per line works best. |
| `batch` | `4` | How many sequences are processed per step. |
| `epochs` | `5` | How many times the whole dataset is seen. |
| `lr` | `0.001` | Peak learning rate of the AdamW optimizer. It's reached after `warmup` steps and stays there. |
| `valid` | `0.05` | Fraction of the dataset kept for validation, taken from the **end** of the file. |
| `artifact` | `"model"` | Name of the run. Every file it writes starts with it, see [What training writes](#what-training-writes). |
| `init` | `null` | A checkpoint to start from instead of random weights, see [Chaining trainings](#chaining-trainings). |
| `seed` | `50` | Seed for the weights initialization and the data shuffling, so runs are repeatable. |
| `warmup` | `200` | Steps it takes to ramp the learning rate from zero to `lr`. Keeps the first steps from blowing up. |
| `clip` | `1.0` | Maximum size (L2 norm) a gradient can have, anything bigger is scaled down. It's applied to every weight tensor on its own. |
| `checkpoint_minutes` | `30` | How often a step checkpoint is saved while training. `0` saves one every 25 steps. |
| `valid_batch` | `8` | Batch size used for validation. |

> [!NOTE]
> Since validation comes from the end of your dataset, shuffle the lines of your file beforehand. Otherwise, if your file is sorted (say, English first and Spanish last), you'd be validating on something the model has barely seen during training.

> [!NOTE]
> The validation part needs at least `seq_length + 1` tokens, otherwise training stops with an error. Raise `valid` or add more data.

### What training writes

Everything goes into the `--artifacts` directory, named after your `artifact`:

| File | When | What it is |
|---|---|---|
| `<artifact>-config.json` | At the start | The config of this run, so it can be loaded later. |
| `config.json` | At the start | A copy of the config of the **last** run. `phisch generate` uses it when you don't pick a checkpoint. |
| `<artifact>-latest.mpk` | End of every epoch | The most recent weights. |
| `<artifact>-best.mpk` | End of an epoch, if validation improved | The weights with the lowest validation loss so far. This is the one `phisch generate` loads by default. |
| `<artifact>-step.mpk` | Every `checkpoint_minutes` | Weights in the middle of an epoch, for resuming. |
| `<artifact>-step-optim.mpk` | Every `checkpoint_minutes` | The optimizer state, for resuming. |
| `<artifact>-step-progress.json` | Every `checkpoint_minutes` | Where the run was (epoch, step) and the best validation loss, for resuming. |

### Checkpoints and resuming

If a training dies halfway, you don't have to start again:

```sh
phisch train configs/my-config.json --resume
```

This loads the step checkpoint and carries on from the exact epoch and step it was saved at. The weights, the optimizer state, the best validation loss so far and even the order of the data are the same as if nothing happened.

> [!NOTE]
> Resume with the same config and the same `--artifacts` directory, that's where phisch looks for the step checkpoint. You lose at most `checkpoint_minutes` of training.

### When training stops by itself

If the loss ever turns into `NaN` or infinity, training stops right away with an error telling you where it happened and which checkpoint was the last good one. This usually means the learning rate is too high for your setup. Lower `lr`, or raise `warmup`, and resume or start again.

If only the **validation** loss comes out invalid, phisch warns you, still saves `-latest` and leaves `-best` untouched.

### Chaining trainings

`init` lets you train in stages, each one starting from the previous one. For example, first train on a big corpus of stories so the model learns the language, then continue on conversations so it learns to chat:

```json
{ "dataset": "data/stories.txt", "artifact": "phisch-stories", "init": null, ... }
```

```json
{ "dataset": "data/chats.txt", "artifact": "phisch", "init": "artifacts/phisch-stories-best", ... }
```

```sh
phisch train configs/stories.json && phisch train configs/chats.json
```

> [!NOTE]
> `init` is a path **without** the `.mpk` extension. Every stage has to use the same `d_model`, `d_ff`, `heads` and `layers`, and the same `tokenizer`, since the weights only fit into the same shape. `dropout`, `seq_length` and all the training settings can change between stages.

---

## Generate

Loads a trained model and continues your prompt.

```sh
Usage: phisch generate [OPTIONS] <PROMPT>

Arguments:
  <PROMPT>  Prompt to complete

Options:
  -a, --artifacts <ARTIFACTS>      Directory of a trained model [default: artifacts/]
      --checkpoint <CHECKPOINT>    Checkpoint to load, defaults to the config artifact plus -best
  -t, --tokens <TOKENS>            Amount of tokens to generate [default: 64]
  -k, --top-k <TOP_K>              Amount of candidates to sample from [default: 40]
      --temperature <TEMPERATURE>  Sampling temperature, 0 for greedy [default: 0.8]
  -s, --seed <SEED>                Seed of the sampler [default: 50]
      --max-context <MAX_CONTEXT>  Most tokens a single generation can reach, prompt included [default: 4096]
  -h, --help                       Print help
```

| Argument / Flag | Default | Description |
|---|---|---|
| `<PROMPT>` | *(required)* | The text to continue. |
| `-a`, `--artifacts` | `artifacts/` | The artifacts directory of a training, the same one you gave `phisch train -a`. |
| `--checkpoint` | `<artifact>-best` | Which checkpoint inside `--artifacts` to load, **without** the `.mpk` extension, for example `phisch-latest`. |
| `-t`, `--tokens` | `64` | Most tokens to generate. It can stop earlier, see below. |
| `-k`, `--top-k` | `40` | Only the `k` most likely next tokens can be picked. Lower is safer, higher is more varied. |
| `--temperature` | `0.8` | How adventurous the picks are. Lower sticks to the likely tokens, higher takes more risks. `0` always takes the most likely one. |
| `-s`, `--seed` | `50` | Seed of the sampling. Same prompt, same seed and same settings give the same text. |
| `--max-context` | `4096` | Most tokens a single generation can reach, prompt included, see [Context and memory](#context-and-memory). |

```sh
phisch generate "Once upon a time" -t 100
phisch generate "Once upon a time" --temperature 0
phisch generate "Once upon a time" -a runs/experiment-1 --checkpoint phisch-latest
```

Generation stops when the model writes `<|end|>`, when it reaches `--tokens`, or when it reaches `--max-context`, whatever comes first.

### Which model gets loaded

- **Without `--checkpoint`**, phisch reads `config.json` inside `--artifacts`, which belongs to the last training that wrote there, and loads the `-best`.
- **With `--checkpoint`**, it loads that checkpoint and looks for its own `<artifact>-config.json` next to it. If there isn't one (checkpoints from older versions), it falls back to `config.json`.

That way, you can keep several trainings in the same directory and still load any of them.

### Sampling

- `--temperature 0` is greedy, the model always takes its most likely token. Good for testing, since the output is always the same, but small models tend to repeat themselves this way.
- Otherwise, the scores are divided by `--temperature`, only the `--top-k` best are kept, and one is picked at random following their probabilities.

### Talking to it

`generate` just continues text, it doesn't format anything for you. To have a conversation, write the special tokens in the prompt yourself and end it with `<|assistant|>`, so the model knows it's its turn:

```sh
phisch generate "<|user|>hi, who are you?<|end|><|assistant|>"
```

With a system message and a longer conversation:

```sh
phisch generate "<|system|>You are Phisch, a helpful assistant.<|end|><|user|>hi<|end|><|assistant|>Hello! How can I help?<|end|><|user|>how long do I boil an egg?<|end|><|assistant|>"
```

The model answers and stops when it writes `<|end|>`.

### The output

phisch prints the **whole** text, your prompt plus what the model wrote, with the special tokens visible. The `<|end|>` that stopped the generation isn't printed.   
So for a conversation, the model's answer is everything after the last `<|assistant|>` of your prompt.

> [!NOTE]
> Logs are turned off while generating, and the welcome line goes to stderr, so the text is the only thing on stdout. That makes it safe to pipe the output into another program.

### Context and memory

When generating, phisch keeps a cache of what it has already read, so every new token is fast instead of re-reading the whole text each time.

- The model always looks back at most `seq_length` tokens, the same window it was trained with. Older tokens slide out as new ones come in, so the conversation can go on for longer than `seq_length`.
- A prompt longer than `seq_length` still works, only its last `seq_length` tokens are read.
- `--max-context` is the total number of tokens one generation can reach, prompt included. When it's reached, generation stops.

> [!NOTE]
> `--max-context` has to be bigger than your prompt (after the `seq_length` cut). If it isn't, phisch tells you instead of crashing.

---

## Export

Turns a trained checkpoint into a safetensors file, together with everything needed to use it again, so you can share or publish your model.

```sh
Usage: phisch export [OPTIONS]

Options:
      --checkpoint <CHECKPOINT>  Checkpoint to load, defaults to the config artifact plus -best
  -a, --artifacts <ARTIFACTS>    Directory of a trained model [default: artifacts/]
  -o, --output <OUTPUT>          Output directory for the safetensors file [default: out/]
  -h, --help                     Print help
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `<artifact>-best` | Which checkpoint inside `--artifacts` to export, **without** the `.mpk` extension. It's picked the same way as in `generate`, see [Which model gets loaded](#which-model-gets-loaded). |
| `-a`, `--artifacts` | `artifacts/` | The artifacts directory of a training. |
| `-o`, `--output` | `out/` | Where the exported files are written. Created if it doesn't exist. |

```sh
phisch export --checkpoint phisch-best -o out/phisch-v1
```

Export writes three files into `--output`:

| File | What it is |
|---|---|
| `<checkpoint>.safetensors` | The weights, in f32. Named after `--checkpoint`, or `model.safetensors` if you didn't pass one. |
| `config.json` | The model settings (`d_model`, `d_ff`, `seq_length`, `heads`, `layers`, `dropout`), so the network can be rebuilt around the weights. |
| `tokenizer.json` | The tokenizer the model was trained with. |

> [!NOTE]
> `config.json` and `tokenizer.json` always have the same name, so exporting two different models into the same `--output` overwrites them. Give every export its own directory.

> [!NOTE]
> The rotary position table isn't stored, it's rebuilt when the model is loaded: its values come from the head size in `config.json` (`d_model / heads`), and `phisch generate` makes it as long as `--max-context`. If you want to load the weights outside of phisch, keep in mind two differences with how PyTorch models are usually stored: linear weights are saved as `[in, out]`, and the rotary encoding pairs neighbouring dimensions instead of splitting each head in halves.

---

## Global options

| Flag | Description |
|---|---|
| `-h`, `--help` | Prints the help. Works on every command too, like `phisch train --help`. |
| `-V`, `--version` | Prints the version of phisch. |
| `phisch help <COMMAND>` | Same as `phisch <COMMAND> --help`. |
