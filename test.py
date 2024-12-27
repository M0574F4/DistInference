from PIL import Image
import torch
from DistInference.general_utils import analyze_input, grid_image_show, histogram_plotter, print_module_summary, set_module_grad, save_dict_pickle
from DistInference.tcm import TCM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def tcm_loader(checkpoint, net):
    checkpoint = torch.load(checkpoint, map_location=device)
    
    dictory = {}
    for k, v in checkpoint["state_dict"].items():
        dictory[k.replace("module.", "")] = v
    net.load_state_dict(dictory)
    return net


N = 64
lambda_value = "0_05" # 0_0025 0_05


device = 'cuda'

# TCM
checkpoint = f"/project_ghent/Mostafa/image/ImageTransmission/src/gen_comm/lic_tcm/LIC_TCM/mse_lambda_{lambda_value}_N{N}.pth"
# checkpoint = f"/project_ghent/Mostafa/image/ImageTransmission/src/gen_comm/lic_tcm/LIC_TCM/pths/20240926_063733.pth.tar"
# checkpoint = f"/project_ghent/Mostafa/image/ImageTransmission/src/gen_comm/lic_tcm/LIC_TCM/checkpointscheckpoint_latest.pth.tar"
tcm = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=N, M=320)
# tcm.num_slices=192
tcm = tcm_loader(checkpoint, tcm)
tcm.cuda()
tcm.update()

img = torch.rand(1, 3, 256, 256).cuda()

z = tcm.g_a(img)
img_hat = tcm.g_s(z).clamp_(0, 1)

# psnr = compute_metrics(img, img_hat)['psnr']
# print(psnr)
print(z.shape)
grid_image_show([img, img_hat], scale=.5)