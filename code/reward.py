# -*- coding: utf-8 -*-
import os, json, math
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer

class Rewarder:
    def __init__(self, mode="rule"):
        self.mode = mode
        self.pipe = None
        if mode == "sentiment":
            from transformers import pipeline
            self.pipe = pipeline("sentiment-analysis")

    @torch.no_grad()
    def __call__(self, prompts: List[str], responses: List[str]) -> List[float]:
        if self.mode == "rule":
            pos = ["great","正确","清晰","succinct","thank","helpful"]
            neg = ["错误","粗鲁","垃圾","toxic","stupid"]
            R = []
            for r in responses:
                s=0.0; low=r.lower()
                s += sum(k in low for k in pos)*0.5
                s -= sum(k in low for k in neg)*1.0
                if len(r)>512: s -= 0.2
                R.append(float(s))
            return R
        elif self.mode == "sentiment":
            preds = self.pipe(responses, truncation=True)
            R=[]
            for p in preds:
                lab=p["label"].lower(); sc=float(p["score"])
                if "pos" in lab: R.append(+sc)
                elif "neg" in lab: R.append(-sc)
                else: R.append(0.0)
            return R
        else:
            raise ValueError("unknown reward_mode")

