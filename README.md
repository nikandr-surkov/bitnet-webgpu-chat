# bitnet-webgpu-chat

https://github.com/user-attachments/assets/88fc8651-7559-41db-a3ec-8ffbcf0f1548

Run a 3B ternary AI model locally in Node.js using WebGPU. No cloud API, no Python, no CUDA.

## What this is

A minimal chat interface that runs TII's Falcon-E 3B Instruct model (native 1.58-bit ternary weights) entirely on your local GPU through WebGPU compute shaders.

## Requirements

- Node.js 20+
- A GPU with WebGPU support and enough VRAM (~1GB)
- npm

## Setup

npm install
npx tsx index.ts

The first run downloads ~1GB from Hugging Face. After that, everything runs offline.

## How it works

- **0xbitnet** handles ternary weight unpacking and WebGPU inference
- **webgpu** provides the Node.js WebGPU bindings
- The model streams responses token by token

Type `exit` to quit.

## Credits

- [Microsoft BitNet](https://github.com/microsoft/BitNet) — the 1-bit LLM framework
- [TII Falcon-Edge](https://falcon-lm.github.io/blog/falcon-edge/) — the Falcon-E model family
- [0xbitnet](https://www.npmjs.com/package/0xbitnet) — WebGPU inference runtime
