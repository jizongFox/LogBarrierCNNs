#!/usr/bin/env bash
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0  python main.py --lamda 1 &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1  python main.py --lamda 10 &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2  python main.py --lamda 100 &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3  python main.py --lamda 1000