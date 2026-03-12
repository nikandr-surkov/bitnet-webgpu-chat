import { create, globals } from "webgpu";
import { BitNet } from "0xbitnet";
import { createInterface } from "readline/promises";

// 1. Inject WebGPU into Node.js
Object.assign(globalThis, globals);

const MAX_HISTORY = 20;

async function main() {
  console.log("Initializing WebGPU Engine...");
  const gpu = create([]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });

  if (!adapter) {
    console.error("Failed to find a WebGPU adapter.");
    process.exit(1);
  }

  let device;
  try {
    device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
        maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      },
    });
  } catch (err) {
    console.error("Failed to create WebGPU device:", err);
    process.exit(1);
  }

  console.log("Downloading Falcon-E 3B Instruct Model (~1 GB)...");

  // 2. Load the optimized Falcon 3B Instruct model
  const model = await BitNet.load(
    "https://huggingface.co/tiiuae/Falcon-E-3B-Instruct-GGUF/resolve/main/ggml-model-i2_s.gguf",
    {
      device,
      onProgress: (p) => {
        process.stdout.write(
          `\rLoading phase [${p.phase}]: ${(p.fraction * 100).toFixed(1)}%`.padEnd(60)
        );
      },
    }
  );

  console.log("\n\n✅ Model Ready!");

  const rl = createInterface({ input: process.stdin, output: process.stdout });

  // 3. Keep the clean array structure
  const history = [
    {
      role: "system",
      content:
        "You are an expert, concise AI assistant for PrivateDoc AI. You analyze documents carefully and answer directly.",
    },
  ];

  try {
    while (true) {
      const input = await rl.question("\nYou: ");

      if (!input.trim()) continue;
      if (input.toLowerCase() === "exit") break;

      history.push({ role: "user", content: input });

      // Trim history to prevent context overflow, always keep system prompt
      if (history.length > MAX_HISTORY) {
        history.splice(1, history.length - MAX_HISTORY);
      }

      process.stdout.write("AI: ");

      try {
        let fullResponse = "";

        // 4. Pass the ARRAY directly to the engine.
        const stream = model.generate(history, {
          maxTokens: 2048,
          temperature: 0.5,
        });

        for await (const token of stream) {
          const cleanToken = token.replace(/\r/g, "");
          process.stdout.write(cleanToken);
          fullResponse += cleanToken;
        }

        // Force a newline so readline doesn't overwrite single-line responses
        process.stdout.write("\n");

        history.push({ role: "assistant", content: fullResponse });
      } catch (err) {
        console.error("\nError during generation:", err);
      }
    }
  } finally {
    rl.close();
    device.destroy();
  }
}

main().catch(console.error);