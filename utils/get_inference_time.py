import torch
import os

iterations = 1000   # 重复计算的轮次

model_sub_path = r'bone5_others3_class/2023_2_17_clavicle1.5_bone5_jsrt3_split_denseEn_resDe_two_decoder_1center_denseblock_4fold_bone_2CAS_jsrt_2AG_bone_AMFS_jsrt_AMFS137_deep_supervision_map2_map3_bone_0.6_0.4_1/fold_0/bone/bone/model_bone_149-0.910986.pth'
model_path = os.path.join(r'../../compare_experiment/OUT/no_contour', model_sub_path)

model = torch.load(model_path)
model.eval()
device = torch.device("cuda:0")
model.to(device)

random_input = torch.randn(1, 3, 512, 512).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input,  mode='test', task_type='bone')

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input,  mode='test', task_type='bone')
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))

