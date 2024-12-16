import torch
import os


'''
    计算推理时间
'''
iterations = 1000   # 重复计算的轮次

# model_sub_path = r'bone_jsrt_dense_class/2023_2_12_aug_bone4_jsrt3_denseEn_resDe_two_decoder_1center_denseblock_4fold_bone_2CAS_jsrt_2AG_bone_AMFS_jsrt_AMFS137_deep_supervision_map2_map3_weight_std/fold_0/bone/model_bone_46-0.908419.pth'
# model_sub_path = r'bone_jsrt_dense_class/2023_2_2_bone4_jsrt3_denseEn_resDe_two_decoder_1center_denseblock_bone_2CAS_jsrt_2AG/fold_0/bone/model_bone_51-0.894560.pth'
model_sub_path = r'bone_jsrt_dense_class/2023_2_3_bone4_jsrt3_denseEn_resDe_two_decoder_1center_denseblock_bone_2CAS_jsrt_2AG_bone_AMFS_JSRT_AMFS/fold_0/bone/model_bone_63-0.895962.pth'
model_path = os.path.join(r'../../compare_experiment/OUT/no_contour', model_sub_path)

model = torch.load(model_path)
model.eval()
device = torch.device("cuda:0")
model.to(device)

random_input = torch.randn(1, 3, 512, 512).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))

