# POD_INFO.md
**Runtime & Infrastructure Specification for the Rap Bot / Blacklight Project**

---

## 1. Pod Overview

| Field | Value |
|------|-------|
| **Provider** | RunPod |
| **Pod Name** | `rap_bot` |
| **Pod ID** | `t15c8yktneps16` |
| **Status** | Running |
| **Uptime (at capture)** | ~6 hours |
| **Networking** | Pod-only, secure cloud (CA-MTL-1) |
| **Web Terminal** | Enabled |

This pod hosts the entire Rap Bot / Blacklight training and generation pipeline, including fine-tuning, rhyme-scoring, and verse generation.

---

## 2. Compute Resources

### GPU
- **Model:** NVIDIA **RTX A6000**
- **Count:** 1 GPU
- **VRAM:** ~48 GB

### CPU
- **vCPUs:** 9 virtual CPUs

### Memory
- **RAM:** 50 GB system memory

### Storage
| Storage Type | Size | Cost |
|--------------|------|------|
| Container Disk | 30 GB | Included in compute |
| Volume Storage | 100 GB (mounted at `/workspace`) | $0.014/hr |
| Container Layer | 30 GB | $0.004/hr |

---

## 3. Pricing Summary

| Component | Cost |
|----------|------|
| Compute (A6000) | **consult RunPod plan (~$0.80–$1.00/hr on demand)** |
| Container Storage | $0.004/hr |
| Volume Storage | $0.014/hr |
| **Estimated Total** | **$0.86–$1.02/hr** |

(*Prices reflect RunPod's on-demand A5000 tier at the time captured.*)

---

## 4. Container Image

The pod uses a PyTorch-focused base image suitable for GPU-intensive ML workloads.

```
Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
Template: runpod-torch-v280
```

### Included Features
- CUDA 12.8.1  
- PyTorch 2.8.0  
- Ubuntu 24.04  
- Support for:
  - flash-attention  
  - bitsandbytes  
  - HuggingFace transformers  
  - all Qwen model variants  

This environment is fully compatible with:
- LoRA fine-tuning (QLoRA, PEFT)  
- Rhyme scoring (Siamese model)  
- Qwen model loading in FP16 or 4-bit quantized modes  

---

## 5. Networking & Access

### HTTP Services
- **Jupyter Lab:**  
  Accessible via proxied HTTPS through port **8888**.

### SSH Access
Two connection methods are available:

#### 1. Through RunPod-hosted SSH reverse tunnel:
```
ssh t15c8yktneps16-644113a5@ssh.runpod.io -i ~/.ssh/id_ed25519
```

#### 2. Direct TCP SSH:
```
ssh root@69.30.85.220 -p 22133 -i ~/.ssh/id_ed25519
```

### Web Terminal
- Enabled  
- Port: **19123**  
- Runs fully in-browser  
- Useful for quick execution inside the environment  

### Direct TCP Ports
- Port mapping:
```
69.30.85.220:22133 → pod port 22 (SSH)
```

---

## 6. Recommended Usage for Rap Bot

### ✔ LoRA fine-tuning (Stage 1 / Stage 2)
Ideal for:
- Qwen 7B–14B in FP16 or 4-bit quantization  
- Training with PEFT/QLoRA  
- Running rhyme-scoring models  

### ✔ Stage-3 critic-weighted refinement  
Handles large weighted corpora and multi-hour fine-tuning well.

### ✔ High-throughput verse generation  
Supports:
- Multiple sampling candidates  
- Rhyme-scoring loops  
- Structural generation (Qwen 7B–14B)  

### ✔ Rhyme-scoring workloads  
The GPU accelerates Siamese inference effectively.

---

## 7. Performance Notes

### Strengths
- Excellent cost-to-performance ratio for 48 GB class  
- Enough VRAM for Qwen 7B and 14B pipelines (FP16 or 4-bit)  
- Stable for long training runs  

### Limitations
- Cannot support 30B–70B-class models on a single card  
- Qwen 70B still requires multi-GPU or heavy CPU/NVMe offload  
- Best results achieved in FP16/4-bit modes  

### Upgrade Path
Move to **H100/A100 80GB** or multi-GPU setups for:
- Native 70B-class hosting  
- Larger sequences  
- Faster Stage-3 training with massive batches  

---

## 8. Pod Best Practices

### Monitor GPU
```
nvidia-smi
```

### Use tmux for long-running jobs
```
tmux new -s train
```

### Backup adapters before stopping
```
cp -r /workspace/rap-botV4/lora_elite_v2 /workspace/backups/
```

### Optional: Enable Cloud Sync
Prevents accidental data loss.

---

## 9. Summary

Your RunPod A5000 environment is perfectly matched to the Rap Bot project:

- Strong enough for all rhyme-aware generation  
- Efficient for multi-stage LoRA training  
- Cost-effective for long sessions  
- Stable for iterative experiments  

This pod acts as the **infrastructure backbone** of Rap Bot / Blacklight.

---

# END OF FILE
