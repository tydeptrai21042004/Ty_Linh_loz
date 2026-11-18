"""Measured native-PyTorch inference profile for one restored-source YOLO checkpoint.

Use inside the same checked-out source tree as the checkpoint's implementation.
C2Former/MMRotate must be profiled separately in its own environment; never
invent a cross-framework speed or FLOPs number when source support is missing.
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from pathlib import Path


def profile(checkpoint, *, source, imgsz=512, batches=(1,4), repeats=40, warmup=10, device='cuda:0'):
    import torch
    source=Path(source).resolve();checkpoint=Path(checkpoint).resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    # Source mixing creates false baselines; imported Ultralytics must live in source.
    import ultralytics
    if not Path(ultralytics.__file__).resolve().is_relative_to(source):
        raise RuntimeError('Imported implementation does not belong to requested source')
    from ultralytics import YOLO
    model=YOLO(str(checkpoint),task='obb')
    module=model.model.to(device).eval()
    params=sum(p.numel() for p in module.parameters())
    results=[]
    def sync():
        if device.startswith('cuda'):torch.cuda.synchronize()
    for batch in batches:
        x=torch.zeros((batch,6,imgsz,imgsz),device=device)
        with torch.inference_mode():
            for _ in range(warmup):module(x)
            sync()
            if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(repeats):
                sync();start=time.perf_counter();module(x);sync()
                times.append((time.perf_counter()-start)*1000)
            peak=(torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)
        results.append({'batch':batch,'median_latency_ms':statistics.median(times),
                        'mean_latency_ms':statistics.mean(times),
                        'images_per_second':1000*batch/statistics.mean(times),
                        'peak_allocated_bytes':peak})
    return {'checkpoint':str(checkpoint),'source':str(source),'device':device,
            'batch_results':results,'parameters':params,'flops':None,
            'flops_note':'Not estimated without a validated counter for eigen/SVD/grid-sampling operators',
            'measurement_scope':'native PyTorch forward including prediction head, excluding loader and NMS'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--imgsz',type=int,default=512)
    p.add_argument('--repeats',type=int,default=40);p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    result=profile(a.checkpoint,source=a.source,imgsz=a.imgsz,repeats=a.repeats,warmup=a.warmup,device=a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
