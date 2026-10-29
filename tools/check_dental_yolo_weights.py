"""CPU invariants without importing the GPU Ultralytics installation."""
import ast
from pathlib import Path

import torch


source=ast.parse((Path(__file__).parent/'dental_yolo_ablation.py').read_text())
function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='positive_weights')
scope={'torch':torch}
exec(compile(ast.Module(body=[function],type_ignores=[]),'weights-test','exec'),scope)
fn=scope['positive_weights']
fg=torch.tensor([[True,True,True,True,False],[False]*5])
indices=torch.tensor([[0,0,0,1,0],[0]*5])
logits=torch.zeros(2,5,1,requires_grad=True)
targets=torch.zeros_like(logits);targets[0,:4]=.8
for balance,hard in [(False,False),(True,False),(False,True),(True,True)]:
    w,_=fn(fg,indices,logits.detach(),targets,balance,hard)
    assert torch.isfinite(w).all() and (w>0).all() and not w.requires_grad
    torch.testing.assert_close(w[~fg],torch.ones_like(w[~fg]))
    torch.testing.assert_close((w*fg).sum(1),fg.sum(1).float())
    if balance: assert w[0,3]>w[0,0]
    elif not hard: torch.testing.assert_close(w,torch.ones_like(w))
    loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none').sum(-1)*w).sum()
    loss.backward(); assert torch.isfinite(logits.grad).all();logits.grad=None
print('PASS: neutral parity, no positives, instance balance, foreground mass, unchanged background, detached weights and finite backward')
