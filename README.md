![Phisch banner](./assets/PhischBanner.png)

# Phisch ><> 
<sup><sub>pronounced like fish...</sub></sup>
Phisch is a Language Model fully made in Rust with Burn 🔥.   
It's made with the intention of being easy and fast to train on consumer hardware, including a CLI with training and generation steps built in.   
It also has configuration files that will let you customize model parameters to your liking, you could also define multiple files for more complex training pipelines.

## 💾 Installation

You'll need a recent Rust toolchain (1.85 or newer) and a GPU that supports Vulkan, Metal or DirectX 12.

```sh
git clone https://github.com/gg0074x/phisch.git
cd phisch
cargo install --path .
```

This installs the `phisch` command. If you'd rather not install it, `cargo build --release` leaves the binary at `target/release/phisch`.

## 🗒️ Usage

> [!NOTE]
> Check [USAGE.md](./USAGE.md) for a more detailed explanation of every command and flag.

A basic training pipeline goes as follows:   
First you can use `phisch config` to generate a default config file in the `configs/` directory, the default settings file will show as follows:
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

> [!NOTE]
> It's recommended that you consciously research about these settings and adjust them based on your training preferences, corpus size, and hardware.

Then you must train a tokenizer with `phisch tokenizer data/your-corpus.txt`, this will output the tokenizer at `data/tokenizer.json` (you can change where with `-o`). Remember to point the `tokenizer` and `dataset` fields of your config to your tokenizer and your corpus.

> [!NOTE]
> The tokenizer reserves custom tokens used for conversational pipelines + some other tokens that i will use in the future.

The next step is actually training the model, when you have your configuration prepared you can use `phisch train configs/your-config.json` to train the model with the specified configuration. Once a training has started, every epoch will save a checkpoint on the `artifacts/` folder.

Once you have a successfully trained Phisch you can generate a sequence with it by using `phisch generate <PROMPT>`, if a checkpoint isn't passed via argument it will load the `-best` checkpoint of the last model trained in the `artifacts/` directory.

Finally, if you want to share your Phisch you can use `phisch export` to turn a checkpoint into a safetensors file, it will write the weights together with the model config and the tokenizer in the `out/` directory.

You can also serve a Phisch over HTTP with `phisch serve`, then load a model and generate with it from any other program.

```sh
Usage: phisch <COMMAND>

Commands:
  config     Generate a default config
  tokenizer  Train the tokenizer
  train      Train a Phisch
  generate   Generate a sequence of text
  export     Export a checkpoint as safetensors
  info       Show sizes of a config or a corpus
  serve      Run a web server with the model in the specified directory
  help       Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```
