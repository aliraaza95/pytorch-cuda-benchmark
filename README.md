# PyTorch CUDA Benchmark

A reproducible PyTorch benchmark comparing CPU and CUDA-enabled GPU training performance across batch sizes using CIFAR-10 and ResNet-18.

## Overview

This project is designed to evaluate how CUDA acceleration affects a practical image-classification training workload. It compares CPU and NVIDIA GPU execution under controlled PyTorch configurations, focusing on training throughput, epoch duration, and speedup across different batch sizes.

## Planned Benchmark

- **Framework:** PyTorch
- **Dataset:** CIFAR-10
- **Model:** ResNet-18 adapted for 32×32 images
- **Devices:** CPU and CUDA-enabled NVIDIA GPU
- **Batch sizes:** 32, 64, 128, and 256
- **Primary metric:** Training throughput in images per second
- **Supporting metrics:** Mean epoch duration, timing variability, and CPU-to-GPU speedup

## Project Goals

- Provide a reproducible CPU and CUDA benchmarking workflow.
- Capture relevant hardware and software information alongside benchmark results.
- Demonstrate how batch size affects CPU and GPU training performance.
- Generate clear performance visualizations for technical communication.
- Document benchmark assumptions, controls, and limitations.