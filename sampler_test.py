from train import TransdimensionalFlowModule
from sampling import any_order_mask_insertion_euler_sampling, SamplingTraceDatapoint
import os


checkpoint_path = "./checkpoints/bracket/any-order/linear-Brian/epoch=89-val_loss=8.7507.ckpt"
model = TransdimensionalFlowModule.load_from_checkpoint(checkpoint_path)
print("model loaded")

steps = 2000
batch_size = 20
samples, trace = any_order_mask_insertion_euler_sampling(
    model,
    model.interpolant,
    steps=steps,
    mask=0,
    pad=3,
    batch_size=batch_size,
    max_length=64,
    return_trace=True,
)

print(samples)


