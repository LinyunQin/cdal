import torch

def fgsm_attack(image, data_grad, epsilon):
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    #print(image)
    sign_data_grad = data_grad.sign()
    # 通过epsilon生成对抗样本
    perturbed_image = image - epsilon*sign_data_grad
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    #print(perturbed_image)
    # 返回对抗样本
    return perturbed_image
